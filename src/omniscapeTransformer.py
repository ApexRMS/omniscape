## omniscape

# Set up -----------------------------------------------------------------------

from osgeo import gdal
import pysyncrosim as ps
import pandas as pd
import sys
import os
import time
import rasterio
from rasterio import Affine
import numpy as np
import subprocess
import signal
import atexit

# Import helper functions
from helperFunctions import (
    load_tile_manifest,
    determine_execution_mode,
    crop_buffer_from_output,
    extend_tile_to_full_extent,
    merge_tile_outputs
)

# Global variable to track Julia process for cleanup
julia_process = None

def cleanup_julia_process():
    """Kill Julia process and all its children when script exits or is cancelled"""
    global julia_process
    if julia_process is not None:
        try:
            # On Windows, kill the entire process tree including child processes
            if os.name == 'nt':  # Windows
                import psutil
                try:
                    parent = psutil.Process(julia_process.pid)
                    children = parent.children(recursive=True)

                    # Kill children first
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass

                    # Then kill parent
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass

                except psutil.NoSuchProcess:
                    pass  # Process already terminated
            else:  # Unix-like systems
                import signal
                try:
                    os.killpg(os.getpgid(julia_process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already terminated
        except Exception as e:
            # Best effort cleanup - don't crash if cleanup fails
            pass

# Register cleanup function to run on exit or cancellation
atexit.register(cleanup_julia_process)
if os.name != 'nt':  # Unix-like systems support signal handlers
    signal.signal(signal.SIGTERM, lambda signum, frame: (cleanup_julia_process(), sys.exit(0)))


ps.environment.progress_bar(message="Setting up Scenario", report_type="message")
ps.environment.update_run_log("=== Omniscape Transformer Starting ===")

e = ps.environment._environment()
wrkDir = e.data_directory.item()
ps.environment.update_run_log(f"Working directory: {wrkDir}")

if os.path.exists(wrkDir) == False:
    os.mkdir(wrkDir)

myLibrary = ps.Library()
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)
ps.environment.update_run_log(f"Scenario ID: {myScenarioID}")

# Handle parent scenario (may be NaN in partial libraries during tiling)
if pd.isna(myScenario.parent_id):
    myParentScenario = myScenario  # Use self as parent if no parent exists
else:
    myScenarioParentID = int(myScenario.parent_id)
    myParentScenario = myLibrary.scenarios(sid = myScenarioParentID)

dataPath = os.path.join(wrkDir, "Scenario-" + repr(myScenarioID)) 



# Load input and settings from SyncroSim Library -------------------------------
ps.environment.update_run_log("Loading datasheets...")

requiredData = myScenario.datasheets(name = "omniscape_Required")
requiredDataValidation = myScenario.datasheets(name = "omniscape_Required", show_full_paths=True)
generalOptions = myScenario.datasheets(name = "omniscape_GeneralOptions")
resistanceOptions = myScenario.datasheets(name = "omniscape_ResistanceOptions")
reclassTable = myScenario.datasheets(name = "omniscape_ReclassTable")
condition1 = myScenario.datasheets(name = "omniscape_Condition1")
condition2 = myScenario.datasheets(name = "omniscape_Condition2")
conditionalOptions = myScenario.datasheets(name = "omniscape_ConditionalOptions")
futureConditions = myScenario.datasheets(name = "omniscape_FutureConditions")
outputOptions = myScenario.datasheets(name = "omniscape_OutputOptions")
multiprocessing = myScenario.datasheets(name = "core_Multiprocessing")
tilingOptions = myScenario.datasheets(name = "omniscape_TilingOptions")
juliaConfig = myLibrary.datasheets(name = "core_JlConfig")  # Julia config is now at library level

# ============================================================================
# LOAD TILE MANIFEST AND DETERMINE EXECUTION MODE
# ============================================================================

# Load tile manifest created by prep transformer
manifest = load_tile_manifest(myScenarioID, wrkDir)

if manifest is None:
    # No tiling - run on full extent (traditional mode)
    ps.environment.update_run_log("No tile manifest found - processing full extent")
    is_tiling = False
    mode = "full_extent"
    tiles_to_process = [None]
else:
    # Tiling enabled - determine execution mode
    ps.environment.update_run_log(f"Tile manifest loaded: {manifest['tile_count']} tiles, buffer={manifest['buffer_pixels']} pixels")

    is_tiling = True
    mode, assigned_tile_id, all_tile_ids = determine_execution_mode(myLibrary, manifest)

    if mode == "multiprocessing":
        # Process only the assigned tile
        tiles_to_process = [assigned_tile_id]
        ps.environment.update_run_log(f"MULTIPROCESSING MODE: This job will process tile {assigned_tile_id} of {manifest['tile_count']}")
        ps.environment.progress_bar(
            message=f"Processing tile {assigned_tile_id}/{manifest['tile_count']}",
            report_type="message"
        )
    else:
        # Loop mode: process all tiles sequentially
        tiles_to_process = all_tile_ids
        ps.environment.update_run_log(f"LOOP MODE: Will process {len(all_tile_ids)} tiles sequentially")
        ps.environment.progress_bar(
            message=f"Processing {len(all_tile_ids)} tiles sequentially",
            report_type="message"
        )

# ============================================================================
# CALCULATE HIERARCHICAL PARALLELIZATION
# ============================================================================

julia_workers_per_tile = 1  # Default: no Julia parallelization

if multiprocessing.EnableMultiprocessing.item() == "Yes":
    total_workers = int(multiprocessing.MaximumJobs.item())

    if is_tiling:
        num_tiles = manifest['tile_count']

        if mode == "loop":
            # Loop mode: tiles run sequentially, each tile gets ALL workers
            julia_workers_per_tile = total_workers
            ps.environment.update_run_log(
                f"Loop mode parallelization: {num_tiles} tiles (sequential) × {julia_workers_per_tile} Julia workers/tile"
            )
        else:
            # Multiprocessing mode: Each Job-X.ssim handles ONE tile
            # SyncroSim controls job-level parallelism, so each job should use all available workers
            julia_workers_per_tile = total_workers
            ps.environment.update_run_log(
                f"Multiprocessing mode: This job (tile {assigned_tile_id}/{num_tiles}) will use {julia_workers_per_tile} Julia workers"
            )
            ps.environment.update_run_log(
                f"Note: SyncroSim will spawn up to {total_workers} concurrent jobs (tiles) based on MaximumJobs setting"
            )

            # Check for over-subscription (if all tiles run simultaneously)
            import os
            physical_cores = os.cpu_count() or 1
            # In multiprocessing mode, SyncroSim may run multiple jobs concurrently
            # Worst case: all tiles run at once
            max_concurrent_jobs = min(num_tiles, total_workers)
            potential_workers = max_concurrent_jobs * julia_workers_per_tile

            if potential_workers > physical_cores * 1.5:
                ps.environment.update_run_log(
                    f"WARNING: Potential over-subscription detected! "
                    f"If {max_concurrent_jobs} tiles run simultaneously with {julia_workers_per_tile} Julia workers each, "
                    f"that's {potential_workers} total workers but only {physical_cores} physical cores available. "
                    f"Performance may degrade due to context switching."
                )
    else:
        # Non-tiling mode: use all available workers
        julia_workers_per_tile = total_workers
        ps.environment.update_run_log(f"Non-tiling parallelization: {julia_workers_per_tile} Julia workers")
else:
    # Multiprocessing disabled: single-threaded
    julia_workers_per_tile = 1
    ps.environment.update_run_log("Multiprocessing disabled: single-threaded execution")

# ============================================================================
# APPLY RAM-BASED CAP
# ============================================================================

# Get RAM per thread estimate from tiling options
ram_per_thread_gb = 16  # default
if not tilingOptions.empty and 'RamPerThreadGB' in tilingOptions.columns:
    if not pd.isna(tilingOptions.RamPerThreadGB.item()):
        ram_per_thread_gb = int(tilingOptions.RamPerThreadGB.item())

# Check available RAM and apply constraint
try:
    import psutil
    ramGB = psutil.virtual_memory().total / (1024**3)

    # Calculate maximum threads based on available RAM
    max_threads_by_ram = max(1, int(ramGB / ram_per_thread_gb))

    # Apply RAM constraint if necessary
    if julia_workers_per_tile > max_threads_by_ram:
        original_workers = julia_workers_per_tile
        julia_workers_per_tile = max_threads_by_ram
        ps.environment.update_run_log(" ")
        ps.environment.update_run_log("=" * 70)
        ps.environment.update_run_log("MEMORY CONSTRAINT APPLIED")
        ps.environment.update_run_log("=" * 70)
        ps.environment.update_run_log(
            f"Reducing Julia threads from {original_workers} to {julia_workers_per_tile} "
            f"based on available RAM ({ramGB:.1f} GB)"
        )
        ps.environment.update_run_log(
            f"Estimated RAM usage: {julia_workers_per_tile * ram_per_thread_gb:.1f} GB "
            f"({ram_per_thread_gb} GB/thread × {julia_workers_per_tile} threads)"
        )
        ps.environment.update_run_log(
            f"To use more threads, increase system RAM or reduce RamPerThreadGB setting"
        )
        ps.environment.update_run_log("=" * 70)
        ps.environment.update_run_log(" ")
    else:
        ps.environment.update_run_log(
            f"RAM check: {julia_workers_per_tile} threads × {ram_per_thread_gb} GB/thread "
            f"= {julia_workers_per_tile * ram_per_thread_gb:.1f} GB (within {ramGB:.1f} GB available)"
        )
except ImportError:
    ps.environment.update_run_log("WARNING: psutil not available, skipping RAM-based thread cap")

# Store original resistance path for output datasheet
original_resistance_path = requiredDataValidation.resistanceFile.item()

# ============================================================================
# SET DEFAULT VALUES FOR OPTIONS
# ============================================================================

if generalOptions.sourceFromResistance.item() == "Yes":
   requiredData.sourceFile = pd.Series("None")

if generalOptions.blockSize.empty:
    generalOptions.blockSize = pd.Series(1)

if generalOptions.sourceFromResistance.item() != generalOptions.sourceFromResistance.item():
    generalOptions.sourceFromResistance = pd.Series("No")

if generalOptions.resistanceIsConductance.item() != generalOptions.resistanceIsConductance.item():
    generalOptions.resistanceIsConductance = pd.Series("No")

if generalOptions.rCutoff.item() != generalOptions.rCutoff.item():
    generalOptions.rCutoff = pd.Series("Inf")

if generalOptions.buffer.item() != generalOptions.buffer.item():
    generalOptions.buffer = pd.Series(0)

if generalOptions.sourceThreshold.item() != generalOptions.sourceThreshold.item():
    generalOptions.sourceThreshold = pd.Series(0)

if generalOptions.calcNormalizedCurrent.item() != generalOptions.calcNormalizedCurrent.item():
    generalOptions.calcNormalizedCurrent = pd.Series("No")

if generalOptions.calcFlowPotential.item() != generalOptions.calcFlowPotential.item():
    generalOptions.calcFlowPotential = pd.Series("No")

if generalOptions.allowDifferentProjections.item() != generalOptions.allowDifferentProjections.item():
    generalOptions.allowDifferentProjections = pd.Series("No")

if generalOptions.connectFourNeighborsOnly.item() != generalOptions.connectFourNeighborsOnly.item():
    generalOptions.connectFourNeighborsOnly = pd.Series("No")

if generalOptions.solver.item() != generalOptions.solver.item():
    generalOptions.solver = pd.Series("cg+amg")

if resistanceOptions.reclassifyResistance.empty:
    resistanceOptions.reclassifyResistance = pd.Series("No")

if resistanceOptions.reclassifyResistance.item() == "No":
    resistanceOptions.reclassTable = pd.Series("None")

if resistanceOptions.writeReclassifiedResistance.item() != resistanceOptions.writeReclassifiedResistance.item():
    resistanceOptions.writeReclassifiedResistance = pd.Series("Yes")

if conditionalOptions.conditional.empty:
    conditionalOptions.conditional = pd.Series("No")

if conditionalOptions.nConditions.item() != conditionalOptions.nConditions.item():
    conditionalOptions.nConditions = pd.Series(1)

if conditionalOptions.conditional.item() == "No":
    condition1.condition1File = pd.Series("None")
    condition1.condition1Lower = pd.Series("NaN")
    condition1.condition1Upper = pd.Series("NaN")
    condition2.condition2File = pd.Series("None")
    condition2.condition2Lower = pd.Series("NaN")
    condition2.condition2Upper = pd.Series("NaN")

if condition1.comparison1.item() != condition1.comparison1.item():
    condition1.comparison1 = pd.Series("within")

if condition2.comparison2.item() != condition2.comparison2.item():
    condition2.comparison2 = pd.Series("within")

if futureConditions.compareToFuture.empty:
    futureConditions.compareToFuture = pd.Series("none")

if futureConditions.compareToFuture.item() == "none":
    futureConditions.condition1FutureFile = pd.Series("None")
    futureConditions.condition2FutureFile = pd.Series("None")

if outputOptions.writeRawCurrmap.empty:
    outputOptions.writeRawCurrmap = pd.Series("Yes")

if outputOptions.maskNodata.item() != outputOptions.maskNodata.item():
    outputOptions.maskNodata = pd.Series("Yes")

if outputOptions.writeAsTif.item() != outputOptions.writeAsTif.item():
    outputOptions.writeAsTif = pd.Series("Yes")




# Validation -------------------------------------------------------------------

if juliaConfig.ExePath.empty:
    sys.exit("A julia executable is required. Please configure it in Library > Options > Julia Tools.")

if not os.path.isfile(juliaConfig.ExePath.item()):
    sys.exit("The path to the julia executable is not valid or does not exist. Please check Library > Options > Julia Tools.")

if ' ' in juliaConfig.ExePath.item():
    sys.exit("The path to the julia executable may not contain spaces.")

if not 'julia.exe' in juliaConfig.ExePath.item():
    sys.exit("The path to the julia executable must contain the 'julia.exe' file.")

if multiprocessing.EnableMultiprocessing.item() == "Yes":
    if multiprocessing.MaximumJobs.empty or multiprocessing.MaximumJobs.item() < 1:
        sys.exit("'Maximum Jobs' must be at least 1 when multiprocessing is enabled.")

if requiredData.resistanceFile.item() != requiredData.resistanceFile.item():
    sys.exit("'Resistance file' is required.")

resistanceLayer = rasterio.open(requiredDataValidation.resistanceFile[0])
dataRaster = resistanceLayer.read()
unique, counts = np.unique(dataRaster, return_counts = True)
unique = pd.DataFrame(unique)
if (unique[0] <= 0).values.any():
    sys.exit("'Resistance file' may not contain 0 or negative values.")

if requiredData.radius[0] != requiredData.radius[0]:
    sys.exit("'Radius' is required.")

if generalOptions.sourceFromResistance.item() == "No" and requiredData.sourceFile.item() == "None":
    sys.exit("'Source from resistance' was set to 'No', therefore 'Source file' is required.")

if generalOptions.sourceFromResistance.item() == "No" and requiredDataValidation.sourceFile.item() == requiredDataValidation.sourceFile.item():
    resistanceLayer = rasterio.open(requiredDataValidation.resistanceFile.item())
    sourceLayer = rasterio.open(requiredDataValidation.sourceFile.item())
    if resistanceLayer.crs != sourceLayer.crs:
        sys.exit("'Resistance file' and 'Source file' must have the same Coordinate Reference System.")
    if resistanceLayer.bounds != sourceLayer.bounds:
        sys.exit("'Resistance file' and 'Source file' must have the same raster extent.")

if not resistanceOptions.empty:
    if resistanceOptions.reclassifyResistance.item() == "Yes":
        if reclassTable.empty:
            sys.exit("'Reclassify resistance' was set to 'Yes', therefore 'Reclass Table' is required.")
        if reclassTable['landCover'].isnull().values.any():
            sys.exit("'Reclass Table' has NaN values for 'Land cover class'.")
        if reclassTable['resistanceValue'].isnull().values.any():
            sys.exit("'Reclass Table' has NaN values for 'Resistance value'. If necessary, NaN values should be specified as -9999.")

if not conditionalOptions.empty:
    if conditionalOptions.conditional.item() == "Yes" and conditionalOptions.nConditions.item() == "1" and condition1.condition1File.item() != condition1.condition1File.item():
        sys.exit("'Conditional' was set to 'Yes' and 'Number of conditions' was set to 1, therefore 'Condition 1 file' is required.")
    if conditionalOptions.conditional.item() == "Yes" and conditionalOptions.nConditions.item() == 2 and (condition1.condition1File.item() != condition1.condition1File.item() or condition2.condition2File.item() != condition2.condition2File.item()):
        sys.exit("'Conditional' was set to 'Yes' and 'Number of conditions' was set to 2, therefore 'Condition 1 file' and 'Condition 2 file' are required.")
    if condition1.comparison1.item() == "within" and (condition1.condition1Lower.item() != condition1.condition1Lower.item() or condition1.condition1Upper.item() != condition1.condition1Upper.item()):
        sys.exit("'Comparison 1' was set to 'within', therefore 'Condition 1 lower' and 'Condition 1 upper' are required.")
    if condition2.comparison2.item() == "within" and (condition2.condition2Lower.item() != condition2.condition2Lower.item() or condition2.condition2Upper.item() != condition2.condition2Upper.item()):
        sys.exit("'Comparison 2' was set to 'within', therefore 'Condition 2 lower' and 'Condition 2 upper' are required.")

if not futureConditions.empty:
    if futureConditions.compareToFuture.item() == "1" and futureConditions.condition1FutureFile.item() != futureConditions.condition1FutureFile.item():
        sys.exit("'Compare to future' was set to 1, therefore 'Condition 1 future file' is required.")
    if futureConditions.compareToFuture.item() == "2" and futureConditions.condition2FutureFile.item() != futureConditions.condition2FutureFile.item():
        sys.exit("'Compare to future' was set to 2, therefore 'Condition 2 future file' is required.")
    if futureConditions.compareToFuture.item() == "both" and (futureConditions.condition1FutureFile.item() != futureConditions.condition1FutureFile.item() or futureConditions.condition2FutureFile.item() != futureConditions.condition2FutureFile.item()):
        sys.exit("'Compare to future' was set to 'both', therefore 'Condition 1 future file' and 'Condition 2 future file' are required.")



# Change "Yes" and "No" to "true" and "false" ----------------------------------

generalOptions = generalOptions.replace({'Yes': 'true', 'No': 'false'})
resistanceOptions = resistanceOptions.replace({'Yes': 'true', 'No': 'false'})
conditionalOptions = conditionalOptions.replace({'Yes': 'true', 'No': 'false'})
outputOptions = outputOptions.replace({'Yes': 'true', 'No': 'false'})
multiprocessing = multiprocessing.replace({'Yes': 'true', 'No': 'false'})



# Prepare reclass file ---------------------------------------------------------

ps.environment.progress_bar(message="Preparing for Omniscape run", report_type="message")

if not reclassTable.empty and (reclassTable != "None").values.any():
    reclassTablePath = os.path.join(dataPath, "omniscape_ResistanceOptions")
    if os.path.exists(reclassTablePath) == False:
        os.mkdir(reclassTablePath)
    #reclassTable.resistanceValue = reclassTable.resistanceValue.astype(str)
    reclassTable.loc[reclassTable["resistanceValue"] == -9999, "resistanceValue"] = "missing"
    with open(os.path.join(reclassTablePath, "reclass_table.txt"), "w") as f:
        file = reclassTable.to_string(header=False, index=False)
        f.write(file)
else:
    reclassTablePath = "None"


# NOTE: Tiling and buffering are now handled by prepMultiprocessingTransformer
# Pre-cropped and pre-buffered tiles are loaded from the manifest
# This eliminates ~130 lines of complex runtime processing


# ============================================================================
# MAIN EXECUTION: PROCESS TILES OR FULL EXTENT
# ============================================================================

# Prepare Julia executable
jlExe = juliaConfig.ExePath.item()

if ' ' in dataPath:
    sys.exit("Due to julia requirements, the path to the SyncroSim Library may not contain any spaces.")

# Storage for tile outputs (for merging in loop mode)
tile_output_paths = {}  # Dict: {output_name: [tile1_path, tile2_path, ...]}

# Process each tile (or single full-extent run if not tiling)
for tile_idx, tile_id in enumerate(tiles_to_process):

    if is_tiling:
        ps.environment.progress_bar(
            message=f"Processing tile {tile_idx+1}/{len(tiles_to_process)}",
            report_type="message"
        )
        ps.environment.update_run_log(f"=== Processing Tile {tile_id} ({tile_idx+1}/{len(tiles_to_process)}) ===")

        # Get tile info from manifest
        tile_info = next(t for t in manifest['tiles'] if t['tile_id'] == tile_id)

        # Load pre-processed tile paths
        resistance_file_path = tile_info['resistance_path']
        source_file_path = tile_info['source_path'] if tile_info['source_path'] else "None"

        # Set project name to tile-specific output directory
        project_name = os.path.join(dataPath, f"omniscape_tile_{tile_id}_output")

        ps.environment.update_run_log(f"[TILE {tile_id}] Resistance: {resistance_file_path}")
        ps.environment.update_run_log(f"[TILE {tile_id}] Source: {source_file_path}")
        ps.environment.update_run_log(f"[TILE {tile_id}] Output dir: {project_name}")
        ps.environment.update_run_log(f"[TILE {tile_id}] Buffered: {tile_info['is_buffered']}")

        # Verify tile files exist
        if not os.path.exists(resistance_file_path):
            sys.exit(f"ERROR: Tile resistance file not found: {resistance_file_path}")

        # Get tile file size to confirm it's not the full raster
        tile_size_mb = os.path.getsize(resistance_file_path) / (1024 * 1024)
        ps.environment.update_run_log(f"[TILE {tile_id}] Resistance file size: {tile_size_mb:.2f} MB")

    else:
        # Full extent mode (no tiling)
        ps.environment.progress_bar(message="Running Omniscape", report_type="message")
        ps.environment.update_run_log("=== Processing Full Extent ===")

        resistance_file_path = requiredDataValidation.resistanceFile.item()
        source_file_path = requiredDataValidation.sourceFile.item() if not pd.isna(requiredDataValidation.sourceFile.item()) else "None"
        project_name = os.path.join(dataPath, "omniscape_outputSpatial")

    # ========================================================================
    # GENERATE CONFIG.INI FOR THIS TILE
    # ========================================================================

    ps.environment.update_run_log("Generating config.ini...")

    config_file = open(os.path.join(dataPath, "omniscape_Required", "config.ini"), "w")

    config_file.write(
        "[Required]" + "\n"
        "resistance_file = " + str(resistance_file_path) + "\n"
        "radius = " + repr(requiredData.radius.item()) + "\n"
        "project_name = " + project_name + "\n"
        "source_file = " + str(source_file_path) + "\n"
        "\n"
        "[General Options]" + "\n"
        "block_size = " + repr(generalOptions.blockSize.item()) + "\n"
        "source_from_resistance = " + generalOptions.sourceFromResistance.item() + "\n"
        "resistance_is_conductance = " + generalOptions.resistanceIsConductance.item() + "\n"
        "r_cutoff = " + repr(generalOptions.rCutoff.item()) + "\n"
        "buffer = " + repr(generalOptions.buffer.item()) + "\n"
        "source_threshold = " + repr(generalOptions.sourceThreshold.item()) + "\n"
        "calc_normalized_current = " + generalOptions.calcNormalizedCurrent.item() + "\n"
        "calc_flow_potential = " + generalOptions.calcFlowPotential.item() + "\n"
        "allow_different_projections = " + generalOptions.allowDifferentProjections.item() + "\n"
        "connect_four_neighbors_only = " + generalOptions.connectFourNeighborsOnly.item() + "\n"
        "solver = " + generalOptions.solver.item() + "\n"
    )

    # Add parallelization parameters to General Options section
    if julia_workers_per_tile > 1:
        config_file.write(
            "parallelize = true" + "\n"
            f"parallel_batch_size = {julia_workers_per_tile}" + "\n"
        )
        ps.environment.update_run_log(f"Omniscape parallelization: enabled with {julia_workers_per_tile} workers")
    else:
        config_file.write(
            "parallelize = false" + "\n"
        )
        ps.environment.update_run_log("Omniscape parallelization: disabled (single-threaded)")

    config_file.write(
        "\n"
        "[Resistance Reclassification]" + "\n"
        "reclassify_resistance = " + resistanceOptions.reclassifyResistance.item() + "\n"
        "reclass_table = " + os.path.join(reclassTablePath, "reclass_table.txt") + "\n"
        "write_reclassified_resistance = " + resistanceOptions.writeReclassifiedResistance.item() + "\n"
        "\n"
        "[Conditional Connectivity]" + "\n"
        "conditional = " + conditionalOptions.conditional.item() + "\n"
        "n_conditions = " + repr(conditionalOptions.nConditions.item()) + "\n"
        "condition1_file = " + condition1.condition1File.item() + "\n"
        "comparison1 = " + condition1.comparison1.item() + "\n"
        "condition1_lower = " + condition1.condition1Lower.item() + "\n"
        "condition1_upper = " + condition1.condition1Upper.item() + "\n"
        "condition2_file = " + condition2.condition2File.item() + "\n"
        "comparison2 = " + condition2.comparison2.item() + "\n"
        "condition2_lower = " + condition2.condition2Lower.item() + "\n"
        "condition2_upper = " + condition2.condition2Upper.item() + "\n"
        "compare_to_future = " + futureConditions.compareToFuture.item() + "\n"
        "condition1_future_file = " + futureConditions.condition1FutureFile.item() + "\n"
        "condition2_future_file = " + futureConditions.condition2FutureFile.item() + "\n"
        "\n"
        "[Output Options]" + "\n"
        "write_raw_currmap = " + outputOptions.writeRawCurrmap.item() + "\n"
        "mask_nodata = " + outputOptions.maskNodata.item() + "\n"
        "write_as_tif = " + outputOptions.writeAsTif.item() + "\n"
        "\n"
    )
    config_file.close()

    # ========================================================================
    # GENERATE JULIA SCRIPT
    # ========================================================================

    julia_file = open(os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl"), "w")
    julia_file.write(
        "cd(raw\"" + os.path.join(dataPath, "omniscape_Required") + "\")" + "\n"
        "\n"
        "using Pkg; Pkg.add(name=\"GDAL\"); Pkg.add(name=\"Omniscape\")" + "\n"
        "using Omniscape" + "\n"
        "\n"
        "run_omniscape(\"config.ini\")"
    )
    julia_file.close()

    # ========================================================================
    # RUN JULIA
    # ========================================================================

    runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")

    # ========================================================================
    # CONFIGURE JULIA COMMAND
    # ========================================================================

    # Build command - parallelization is controlled via config.ini
    # We still set -t flag to ensure Julia has threads available for Omniscape to use
    if julia_workers_per_tile > 1:
        runOmniscape = f"{jlExe} -t {julia_workers_per_tile} {runFile}"
        ps.environment.update_run_log(f"Julia threading ENABLED with {julia_workers_per_tile} threads")
    else:
        runOmniscape = f"{jlExe} {runFile}"
        ps.environment.update_run_log(f"Julia threading DISABLED (single-threaded)")

    ps.environment.update_run_log(f"Executing: {runOmniscape}")
    ps.environment.update_run_log(">>> Starting Omniscape.jl <<<")

    start_time = time.time()

    # Use subprocess.Popen instead of os.system for proper process management
    # This allows us to kill Julia and all its child processes if the job is cancelled
    julia_process = subprocess.Popen(
        runOmniscape,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Stream output to SyncroSim log in real-time
    for line in julia_process.stdout:
        stripped_line = line.rstrip()
        if stripped_line:  # Only log non-empty lines
            ps.environment.update_run_log(stripped_line)

    # Wait for process to complete and get exit code
    exit_code = julia_process.wait()
    julia_process = None  # Clear the global reference after completion

    elapsed_time = time.time() - start_time

    ps.environment.update_run_log(f">>> Completed with exit code: {exit_code} (took {elapsed_time:.1f}s) <<<")

    if exit_code != 0:
        sys.exit(f"Omniscape.jl failed with exit code {exit_code}")

    # ========================================================================
    # POST-PROCESS TILE OUTPUTS
    # ========================================================================

    if is_tiling:
        tile_output_dir = os.path.join(dataPath, f"omniscape_tile_{tile_id}_output")

        # Remove buffer if tiles were buffered
        if tile_info['is_buffered']:
            ps.environment.update_run_log(f"Removing {manifest['buffer_pixels']}-pixel buffer from tile outputs...")

            output_files = ['cum_currmap.tif', 'normalized_cum_currmap.tif', 'flow_potential.tif']

            for output_file in output_files:
                buffered_path = os.path.join(tile_output_dir, output_file)

                if os.path.exists(buffered_path):
                    cropped_path = buffered_path.replace('.tif', '_cropped.tif')

                    crop_buffer_from_output(
                        buffered_path,
                        cropped_path,
                        tile_info['original_extent'],
                        manifest['buffer_pixels'],
                        tile_info.get('buffered_extent')
                    )

                    os.remove(buffered_path)
                    os.rename(cropped_path, buffered_path)
                    ps.environment.update_run_log(f"  Removed buffer from {output_file}")

        # Handle mode-specific post-processing
        if mode == "multiprocessing":
            # Extend tiles to full extent for SyncroSim merging
            ps.environment.update_run_log("Extending tile outputs to full extent for SyncroSim merging...")

            full_extent_info = manifest['full_extent']
            full_transform = Affine(*full_extent_info['transform'])

            output_files = ['cum_currmap.tif', 'normalized_cum_currmap.tif', 'flow_potential.tif', 'classified_resistance.tif']

            for output_file in output_files:
                tile_path = os.path.join(tile_output_dir, output_file)

                if os.path.exists(tile_path):
                    extended_path = tile_path.replace('.tif', '_extended.tif')

                    extend_tile_to_full_extent(
                        tile_path,
                        extended_path,
                        (full_extent_info['width'], full_extent_info['height']),
                        full_transform,
                        full_extent_info['crs']
                    )

                    os.remove(tile_path)
                    os.rename(extended_path, tile_path)
                    ps.environment.update_run_log(f"  Extended {output_file} to full extent")

        else:  # Loop mode
            # Collect tile outputs for later merging
            ps.environment.update_run_log(f"Collecting tile {tile_id} outputs for merging...")

            output_files = ['cum_currmap.tif', 'normalized_cum_currmap.tif', 'flow_potential.tif', 'classified_resistance.tif']

            for output_file in output_files:
                tile_path = os.path.join(tile_output_dir, output_file)

                if os.path.exists(tile_path):
                    if output_file not in tile_output_paths:
                        tile_output_paths[output_file] = []
                    tile_output_paths[output_file].append(tile_path)

# ============================================================================
# MERGE TILES IN LOOP MODE
# ============================================================================

if is_tiling and mode == "loop":
    ps.environment.progress_bar(message="Merging tile outputs", report_type="message")
    ps.environment.update_run_log("=== Merging Tile Outputs ===")

    # Create final output directory
    final_output_dir = os.path.join(dataPath, "omniscape_outputSpatial")
    os.makedirs(final_output_dir, exist_ok=True)

    # Merge each output type
    full_extent_info = manifest['full_extent']

    for output_name, tile_paths in tile_output_paths.items():
        final_path = os.path.join(final_output_dir, output_name)

        ps.environment.update_run_log(f"Merging {len(tile_paths)} tiles for {output_name}...")

        merge_tile_outputs(tile_paths, final_path, full_extent_info)

        ps.environment.update_run_log(f"  Created merged output: {output_name}")

    ps.environment.update_run_log("=== Tile Merging Complete ===")

# ============================================================================
# HANDLE NON-TILING MODE OUTPUT LOCATION
# ============================================================================

# In multiprocessing mode, outputs are already in correct location (extended to full extent)
# In non-tiling mode, outputs are already in omniscape_outputSpatial
# In loop mode, outputs have been merged to omniscape_outputSpatial
# So no additional copying needed!



# ============================================================================
# CREATE OUTPUT DATASHEETS
# ============================================================================

ps.environment.update_run_log("Creating output datasheets...")

myOutput = myScenario.datasheets(name = "omniscape_outputSpatial")

# Determine output directory based on execution mode
if is_tiling and mode == "multiprocessing":
    # Multiprocessing mode: outputs in tile-specific directory (extended to full extent)
    output_dir = os.path.join(dataPath, f"omniscape_tile_{tiles_to_process[0]}_output")
else:
    # Non-tiling or loop mode: outputs in standard directory
    output_dir = os.path.join(dataPath, "omniscape_outputSpatial")

ps.environment.update_run_log(f"Output directory: {output_dir}")

# Add outputs to datasheet
if outputOptions.writeRawCurrmap.item() == "true":
    cum_currmap_path = os.path.join(output_dir, "cum_currmap.tif")
    if os.path.exists(cum_currmap_path):
        myOutput.cumCurrmap = pd.Series(cum_currmap_path)
        ps.environment.update_run_log(f"  Added cumCurrmap output")
    else:
        ps.environment.update_run_log(f"  Warning: cum_currmap.tif not found")

if generalOptions.calcFlowPotential.item() == "true":
    flow_potential_path = os.path.join(output_dir, "flow_potential.tif")
    if os.path.exists(flow_potential_path):
        myOutput.flowPotential = pd.Series(flow_potential_path)
        ps.environment.update_run_log(f"  Added flowPotential output")
    else:
        ps.environment.update_run_log(f"  Warning: flow_potential.tif not found")

if generalOptions.calcNormalizedCurrent.item() == "true":
    normalized_path = os.path.join(output_dir, "normalized_cum_currmap.tif")
    if os.path.exists(normalized_path):
        myOutput.normalizedCumCurrmap = pd.Series(normalized_path)
        ps.environment.update_run_log(f"  Added normalizedCumCurrmap output")
    else:
        ps.environment.update_run_log(f"  Warning: normalized_cum_currmap.tif not found")

# Check for classified resistance
classified_path = os.path.join(output_dir, "classified_resistance.tif")
if os.path.exists(classified_path) and resistanceOptions.writeReclassifiedResistance.item() == "true":
    myOutput.classifiedResistance = pd.Series(classified_path)
    ps.environment.update_run_log(f"  Added classifiedResistance output (reclassified)")
elif is_tiling and mode == "multiprocessing":
    # In multiprocessing/tiling mode, skip adding classifiedResistance if it doesn't exist
    # (the full-extent resistance raster may have Byte data type that SyncroSim merger can't handle)
    ps.environment.update_run_log(f"  Skipping classifiedResistance in multiprocessing mode (not generated by Omniscape)")
    # Don't add classifiedResistance to avoid Byte data type merge errors
else:
    # Non-tiling mode or loop mode: add original resistance as fallback
    myOutput.classifiedResistance = pd.Series(original_resistance_path)
    ps.environment.update_run_log(f"  Added classifiedResistance output (original)")



# Save outputs to SyncroSim ---------------------------------------------------------------------

ps.environment.update_run_log("Saving outputs to SyncroSim datasheet...")
myParentScenario.save_datasheet(name = "omniscape_outputSpatial", data = myOutput)
ps.environment.update_run_log("=== Omniscape Transformer Completed Successfully ===")
ps.environment.progress_bar(message="Scenario complete", report_type="message")


