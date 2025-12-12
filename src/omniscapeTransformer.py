## omniscape

# Set up -----------------------------------------------------------------------

from osgeo import gdal
import pysyncrosim as ps
import pandas as pd
import sys
import os
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
from rasterio.windows import Window


# ============================================================================
# SPATIAL TILING HELPER FUNCTIONS
# ============================================================================

def detect_multiprocessing(myLibrary):
    """
    Detect if running in spatial tiling mode.

    Returns:
        tuple: (is_tiling, tile_id, all_tile_ids, max_jobs)
    """
    lib_path = myLibrary.name
    lib_filename = os.path.basename(lib_path)
    lib_name = lib_filename  # Use filename as library name

    ps.environment.update_run_log(f"Checking for tiling mode: lib_name='{lib_name}', lib_filename='{lib_filename}'")

    # Pattern 1: Library name is "Partial" and filename is "library-X.ssim"
    if lib_name == "Partial" and lib_filename.startswith("library-"):
        try:
            tile_id = int(lib_filename.replace("library-", "").replace(".ssim", ""))
        except ValueError:
            ps.environment.update_run_log(f"Warning: Could not extract tile ID from filename: {lib_filename}")
            return (False, None, None, 1)

    # Pattern 2: Library name is "Job-X.ssim" (SyncroSim spatial multiprocessing)
    elif lib_name.startswith("Job-") and lib_name.endswith(".ssim"):
        try:
            tile_id = int(lib_name.replace("Job-", "").replace(".ssim", ""))
        except ValueError:
            ps.environment.update_run_log(f"Warning: Could not extract tile ID from library name: {lib_name}")
            return (False, None, None, 1)

    # No tiling pattern detected
    else:
        ps.environment.update_run_log(f"No tiling pattern detected - using full raster extent")
        return (False, None, None, 1)

    # Parse Jobs.xml to get all tile IDs
    jobs_xml_path = os.path.join(os.path.dirname(lib_path), "Jobs.xml")

    if not os.path.exists(jobs_xml_path):
        ps.environment.update_run_log(f"Warning: Jobs.xml not found at {jobs_xml_path}")
        return (False, None, None, 1)

    tree = ET.parse(jobs_xml_path)
    root = tree.getroot()

    all_tile_ids = []
    for job in root.findall("Job"):
        tid = int(job.get("TileID"))
        all_tile_ids.append(tid)

    max_jobs = len(all_tile_ids)

    ps.environment.update_run_log(
        f"Detected tiling mode: Tile {tile_id} of {max_jobs} tiles"
    )

    return (True, tile_id, all_tile_ids, max_jobs)


def isolate_tile(grid_raster, tile_id, other_tile_ids):
    """
    Mask grid to isolate current tile.

    Args:
        grid_raster: rasterio dataset of tile grid
        tile_id: Current tile ID to isolate
        other_tile_ids: List of all other tile IDs

    Returns:
        tuple: (tile_mask, tile_extent)
            - tile_mask: Boolean array where True = current tile
            - tile_extent: Bounding box (row_start, row_end, col_start, col_end)
    """
    ps.environment.update_run_log(f"[isolate_tile] Reading grid data for tile {tile_id}...")
    grid_data = grid_raster.read(1)

    ps.environment.update_run_log(f"[isolate_tile] Creating tile mask...")
    # Create mask: True for current tile, False elsewhere
    tile_mask = (grid_data == tile_id)

    ps.environment.update_run_log(f"[isolate_tile] Finding bounding box...")
    # Find bounding box of non-masked pixels
    rows, cols = np.where(tile_mask)

    if len(rows) == 0:
        raise ValueError(f"Tile {tile_id} has no valid pixels")

    row_start = rows.min()
    row_end = rows.max() + 1
    col_start = cols.min()
    col_end = cols.max() + 1

    ps.environment.update_run_log(
        f"[isolate_tile] Tile extent: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]"
    )

    return tile_mask, (row_start, row_end, col_start, col_end)


def crop_raster_to_tile(input_path, output_path, tile_extent, grid_transform, grid_crs):
    """
    Crop raster to tile extent.

    Args:
        input_path: Path to input raster
        output_path: Path for cropped raster
        tile_extent: Tuple (row_start, row_end, col_start, col_end)
        grid_transform: Affine transform from grid raster
        grid_crs: CRS from grid raster
    """
    ps.environment.update_run_log(f"[crop_raster_to_tile] Cropping {os.path.basename(input_path)}...")
    row_start, row_end, col_start, col_end = tile_extent

    ps.environment.update_run_log(f"[crop_raster_to_tile] Opening source raster...")
    with rasterio.open(input_path) as src:
        # Create window
        window = Window(col_start, row_start,
                       col_end - col_start,
                       row_end - row_start)

        ps.environment.update_run_log(f"[crop_raster_to_tile] Reading windowed data ({window.width}x{window.height})...")
        # Read windowed data
        data = src.read(1, window=window)

        # Calculate new transform
        new_transform = rasterio.windows.transform(window, src.transform)

        ps.environment.update_run_log(f"[crop_raster_to_tile] Writing cropped raster to {os.path.basename(output_path)}...")
        # Write cropped raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=new_transform,
            nodata=src.nodata,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)

    ps.environment.update_run_log(f"[crop_raster_to_tile] Completed cropping {os.path.basename(input_path)}")


def extend_tile_to_full_extent(tile_raster_path, output_path, full_extent, full_transform, full_crs):
    """
    Extend tile raster to full analysis extent (critical for SyncroSim merging).

    Args:
        tile_raster_path: Path to cropped tile raster
        output_path: Path for extended raster
        full_extent: Tuple (width, height) of full analysis area
        full_transform: Affine transform of full analysis area
        full_crs: CRS of full analysis area
    """
    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Extending {os.path.basename(tile_raster_path)}...")
    full_width, full_height = full_extent

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Reading tile data...")
    with rasterio.open(tile_raster_path) as src:
        tile_data = src.read(1)
        tile_transform = src.transform
        nodata = src.nodata
        dtype = src.dtypes[0]

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Creating full-extent array ({full_width}x{full_height})...")
    # Create full-extent array filled with nodata
    if nodata is not None:
        full_array = np.full((full_height, full_width), nodata, dtype=dtype)
    else:
        full_array = np.full((full_height, full_width), np.nan, dtype=dtype)

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Calculating tile position...")
    # Calculate tile position in full extent
    # Convert tile origin to row/col in full extent
    tile_origin_x = tile_transform.c
    tile_origin_y = tile_transform.f

    full_origin_x = full_transform.c
    full_origin_y = full_transform.f

    pixel_width = full_transform.a
    pixel_height = abs(full_transform.e)

    col_offset = int(round((tile_origin_x - full_origin_x) / pixel_width))
    row_offset = int(round((full_origin_y - tile_origin_y) / pixel_height))

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Placing tile at offset row={row_offset}, col={col_offset}...")
    # Place tile data into full array
    tile_height, tile_width = tile_data.shape

    # Bounds checking
    row_end = min(row_offset + tile_height, full_height)
    col_end = min(col_offset + tile_width, full_width)

    full_array[row_offset:row_end, col_offset:col_end] = \
        tile_data[0:(row_end - row_offset), 0:(col_end - col_offset)]

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Writing extended raster...")
    # Write extended raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=full_height,
        width=full_width,
        count=1,
        dtype=dtype,
        crs=full_crs,
        transform=full_transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(full_array, 1)

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Completed extending {os.path.basename(tile_raster_path)}")


def calculate_memory_per_job(max_jobs):
    """
    Calculate memory allocation per job (WISDM pattern).

    Formula: max(0.5, min(12, 0.6 * totalRAM / maxJobs))

    Args:
        max_jobs: Number of concurrent jobs

    Returns:
        float: Memory in GB per job
    """
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except:
        # Fallback: assume 16GB if psutil not available
        total_ram_gb = 16.0
        ps.environment.update_run_log("Warning: Could not detect system RAM, assuming 16GB")

    mem_per_job = max(0.5, min(12.0, 0.6 * total_ram_gb / max_jobs))

    ps.environment.update_run_log(
        f"Memory allocation: {mem_per_job:.1f} GB per job "
        f"(Total RAM: {total_ram_gb:.1f} GB, Jobs: {max_jobs})"
    )

    return mem_per_job


def buffer_raster(input_path, output_path, buffer_pixels):
    """
    Create buffered version of raster (from buffer_implementation_spec.md).

    Args:
        input_path: Path to input raster
        output_path: Path for buffered raster
        buffer_pixels: Number of pixels to buffer on each side

    Returns:
        dict: Original bounds metadata for later cropping
            - 'height': Original raster height
            - 'width': Original raster width
            - 'transform': Original affine transform
            - 'bounds': Original geographic bounds
            - 'crs': Coordinate reference system
    """
    ps.environment.update_run_log(f"[buffer_raster] Buffering {os.path.basename(input_path)} by {buffer_pixels} pixels...")
    with rasterio.open(input_path) as src:
        ps.environment.update_run_log(f"[buffer_raster] Reading source data ({src.width}x{src.height})...")
        data = src.read(1)

        ps.environment.update_run_log(f"[buffer_raster] Padding array with edge replication...")
        # Pad array with edge replication
        buffered_data = np.pad(data, buffer_pixels, mode='edge')

        ps.environment.update_run_log(f"[buffer_raster] Calculating new transform...")
        # Calculate new transform (shift origin)
        old_transform = src.transform
        new_transform = rasterio.Affine(
            old_transform.a,  # pixel width
            old_transform.b,  # rotation
            old_transform.c - (buffer_pixels * old_transform.a),  # shift left
            old_transform.d,  # rotation
            old_transform.e,  # pixel height (negative)
            old_transform.f - (buffer_pixels * old_transform.e)   # shift up
        )

        ps.environment.update_run_log(f"[buffer_raster] Writing buffered raster ({buffered_data.shape[1]}x{buffered_data.shape[0]})...")
        # Write buffered raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=buffered_data.shape[0],
            width=buffered_data.shape[1],
            count=1,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=new_transform,
            nodata=src.nodata,
            compress='lzw'
        ) as dst:
            dst.write(buffered_data, 1)

        ps.environment.update_run_log(f"[buffer_raster] Completed buffering {os.path.basename(input_path)}")

        # Return original bounds for later cropping
        return {
            'height': src.height,
            'width': src.width,
            'transform': old_transform,
            'bounds': src.bounds,
            'crs': src.crs
        }


def crop_to_original(buffered_path, output_path, original_bounds, buffer_pixels):
    """
    Crop buffered output back to original extent.

    Args:
        buffered_path: Path to buffered raster
        output_path: Path for cropped output
        original_bounds: Bounds dict from buffer_raster()
        buffer_pixels: Buffer size used
    """
    ps.environment.update_run_log(f"[crop_to_original] Cropping buffer from {os.path.basename(buffered_path)}...")
    with rasterio.open(buffered_path) as src:
        # Create window to extract original extent
        window = Window(buffer_pixels, buffer_pixels,
                       original_bounds['width'],
                       original_bounds['height'])

        ps.environment.update_run_log(f"[crop_to_original] Reading windowed data...")
        data = src.read(1, window=window)

        ps.environment.update_run_log(f"[crop_to_original] Writing cropped raster ({original_bounds['width']}x{original_bounds['height']})...")
        # Write cropped raster with original transform
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=original_bounds['height'],
            width=original_bounds['width'],
            count=1,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=original_bounds['transform'],
            nodata=src.nodata,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)

    ps.environment.update_run_log(f"[crop_to_original] Completed cropping {os.path.basename(buffered_path)}")

# ============================================================================
# END SPATIAL TILING HELPER FUNCTIONS
# ============================================================================


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
juliaConfig = myScenario.datasheets(name = "omniscape_juliaConfiguration")

# Load tiling configuration
tilingOptions = myScenario.datasheets(name="omniscape_TilingOptions")

# Detect if running in tiling mode
is_tiling, tile_id, all_tile_ids, max_jobs = detect_multiprocessing(myLibrary)

# Load spatial multiprocessing grid if tiling is enabled
tile_grid_raster = None
full_extent_info = None

ps.environment.update_run_log(f"Tiling mode: {is_tiling}, Library name: {os.path.basename(myLibrary.name)}")

if is_tiling:
    ps.environment.progress_bar(message=f"Processing tile {tile_id} of {max_jobs}", report_type="message")

    # Load tile grid
    smp_datasheet = myScenario.datasheets(name="core_SpatialMultiprocessing", show_full_paths=True)

    if smp_datasheet.empty or pd.isna(smp_datasheet.MaskFileName.item()):
        sys.exit(
            "Tiling mode detected but core_SpatialMultiprocessing datasheet is empty. "
            "Run PrepMultiprocessing transformer first."
        )

    grid_path = smp_datasheet.MaskFileName.item()

    if not os.path.exists(grid_path):
        sys.exit(f"Tile grid file not found: {grid_path}")

    tile_grid_raster = rasterio.open(grid_path)

    # Store full extent info for later extension
    full_extent_info = {
        'width': tile_grid_raster.width,
        'height': tile_grid_raster.height,
        'transform': tile_grid_raster.transform,
        'crs': tile_grid_raster.crs
    }



# If not provided, set default values  -----------------------------------------

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

if juliaConfig.juliaPath.empty:
    sys.exit("A julia executable is required.")

if not os.path.isfile(juliaConfig.juliaPath.item()):
    sys.exit("The path to the julia executable is not valid or does not exist.")

if ' ' in juliaConfig.juliaPath.item():
    sys.exit("The path to the julia executable may not contains spaces.")

if not 'julia.exe' in juliaConfig.juliaPath.item():
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



# ============================================================================
# SPATIAL TILING: CROP INPUTS TO TILE EXTENT
# ============================================================================

original_resistance_path = requiredDataValidation.resistanceFile.item()
original_source_path = requiredDataValidation.sourceFile.item() if requiredData.sourceFile.item() != "None" else None

if is_tiling:
    ps.environment.progress_bar(message="Cropping rasters to tile extent", report_type="message")
    ps.environment.update_run_log(f"=== TILING: Processing tile {tile_id} of {max_jobs} ===")

    # Get other tile IDs (for masking)
    ps.environment.update_run_log(f"Getting other tile IDs for masking...")
    other_tile_ids = [tid for tid in all_tile_ids if tid != tile_id]
    ps.environment.update_run_log(f"Other tile IDs: {other_tile_ids}")

    # Isolate current tile
    ps.environment.update_run_log(f"Isolating tile {tile_id}...")
    tile_mask, tile_extent = isolate_tile(tile_grid_raster, tile_id, other_tile_ids)

    # Create tile-specific directory
    ps.environment.update_run_log(f"Creating tile directory...")
    tileDataPath = os.path.join(dataPath, f"tile-{tile_id}")
    os.makedirs(tileDataPath, exist_ok=True)
    ps.environment.update_run_log(f"Tile directory: {tileDataPath}")

    # Crop resistance raster
    ps.environment.update_run_log(f"Cropping resistance raster to tile extent...")
    cropped_resistance = os.path.join(tileDataPath, "tile_resistance.tif")
    crop_raster_to_tile(
        original_resistance_path,
        cropped_resistance,
        tile_extent,
        tile_grid_raster.transform,
        tile_grid_raster.crs
    )
    ps.environment.update_run_log(f"Resistance cropped to: {cropped_resistance}")

    # Crop source raster if provided
    if original_source_path is not None:
        ps.environment.update_run_log(f"Cropping source raster to tile extent...")
        cropped_source = os.path.join(tileDataPath, "tile_source.tif")
        crop_raster_to_tile(
            original_source_path,
            cropped_source,
            tile_extent,
            tile_grid_raster.transform,
            tile_grid_raster.crs
        )
        ps.environment.update_run_log(f"Source cropped to: {cropped_source}")

        # Update paths for config generation
        requiredDataValidation.at[0, 'sourceFile'] = cropped_source

    # Update resistance path for config generation
    requiredDataValidation.at[0, 'resistanceFile'] = cropped_resistance

    # Close tile grid raster now that we're done with it (prevents file locking)
    ps.environment.update_run_log(f"Closing tile grid raster...")
    if tile_grid_raster is not None:
        tile_grid_raster.close()
        tile_grid_raster = None
    ps.environment.update_run_log(f"=== TILING: Tile cropping complete ===")
else:
    ps.environment.update_run_log(f"Tiling mode disabled - using full raster extent")

# ============================================================================
# END TILING SETUP
# ============================================================================



# ============================================================================
# BUFFERING: APPLY BUFFER TO TILE (OR FULL RASTER IF NOT TILING)
# ============================================================================

# Get buffer configuration
buffer_pixels = 0
if not tilingOptions.empty and 'BufferPixels' in tilingOptions.columns:
    if not pd.isna(tilingOptions.BufferPixels.item()):
        buffer_pixels = int(tilingOptions.BufferPixels.item())

ps.environment.update_run_log(f"Buffer pixels configured: {buffer_pixels}")

original_bounds = None

if buffer_pixels > 0:
    ps.environment.progress_bar(message=f"Applying {buffer_pixels}-pixel buffer", report_type="message")
    ps.environment.update_run_log(f"=== BUFFERING: Applying {buffer_pixels}-pixel buffer ===")

    # Determine which rasters to buffer (tile-cropped or original)
    resistance_to_buffer = requiredDataValidation.resistanceFile.item()
    source_to_buffer = requiredDataValidation.sourceFile.item() if requiredData.sourceFile.item() != "None" else None
    ps.environment.update_run_log(f"Resistance to buffer: {resistance_to_buffer}")
    ps.environment.update_run_log(f"Source to buffer: {source_to_buffer}")

    # Create buffered directory
    if is_tiling:
        bufferedPath = os.path.join(dataPath, f"tile-{tile_id}", "buffered")
    else:
        bufferedPath = os.path.join(dataPath, "buffered")
    os.makedirs(bufferedPath, exist_ok=True)
    ps.environment.update_run_log(f"Buffer directory: {bufferedPath}")

    # Buffer resistance
    ps.environment.update_run_log(f"Buffering resistance raster...")
    buffered_resistance = os.path.join(bufferedPath, "buffered_resistance.tif")
    original_bounds = buffer_raster(resistance_to_buffer, buffered_resistance, buffer_pixels)
    ps.environment.update_run_log(f"Buffered resistance: {buffered_resistance}")

    # Buffer source if provided
    buffered_source = None
    if source_to_buffer is not None and source_to_buffer != "None":
        ps.environment.update_run_log(f"Buffering source raster...")
        buffered_source = os.path.join(bufferedPath, "buffered_source.tif")
        buffer_raster(source_to_buffer, buffered_source, buffer_pixels)
        ps.environment.update_run_log(f"Buffered source: {buffered_source}")

        # Update paths for config generation
        requiredDataValidation.at[0, 'sourceFile'] = buffered_source

    # Update resistance path for config generation
    requiredDataValidation.at[0, 'resistanceFile'] = buffered_resistance

    ps.environment.update_run_log(f"=== BUFFERING: Buffer application complete ===")
else:
    ps.environment.update_run_log(f"Buffering disabled (buffer_pixels = 0)")

# ============================================================================
# END BUFFERING SETUP
# ============================================================================



# Prepare configuration file (.ini) --------------------------------------------
ps.environment.update_run_log("Generating config.ini...")

file = open(os.path.join(dataPath, "omniscape_Required", "config.ini"), "w")

# Get file paths - use updated paths from tiling/buffering if they exist, otherwise use originals
resistance_file_path = requiredDataValidation.resistanceFile.item()
source_file_path = requiredDataValidation.sourceFile.item()

# Handle NaN values (convert to string)
if pd.isna(resistance_file_path):
    resistance_file_path = os.path.join(dataPath, "omniscape_Required", requiredData.resistanceFile.item())
if pd.isna(source_file_path):
    source_file_path = requiredData.sourceFile.item()

file.write(
    "[Required]" + "\n"
    "resistance_file = " + str(resistance_file_path) + "\n"
    "radius = " + repr(requiredData.radius.item()) + "\n"
    "project_name = " + os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial") + "\n"
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
    "[Multiprocessing]" + "\n"
    "parallelize = " + multiprocessing.EnableMultiprocessing.item() + "\n"
    "parallel_batch_size = " + repr(multiprocessing.MaximumJobs.item()) + "\n"
)
file.close()



# Prepare julia script ---------------------------------------------------------

configName = "config.ini"

file = open(os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl"), "w")
file.write(
    "cd(raw\"" + os.path.join(dataPath, "omniscape_Required") + "\")" + "\n"
    "\n"
    "using Pkg; Pkg.add(name=\"GDAL\"); Pkg.add(name=\"Omniscape\")" + "\n"
    "using Omniscape" + "\n"
    "run_omniscape(\"" + configName + "\")"
)
file.close()
ps.environment.update_run_log("Config.ini written successfully")



# Run julia script with system call -------------------------------------------------------------

ps.environment.progress_bar(message="Running Omniscape", report_type="message")
ps.environment.update_run_log("Preparing to execute Julia...")

jlExe = juliaConfig.juliaPath.item()
runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")
ps.environment.update_run_log(f"Julia executable: {jlExe}")
ps.environment.update_run_log(f"Julia script: {runFile}")

if ' ' in dataPath:
    sys.exit("Due to julia requirements, the path to the SyncroSim Library may not contain any spaces.")

# Add thread specification if multiprocessing is enabled
# NOTE: For tiling mode, use threads per tile, not total threads
if multiprocessing.EnableMultiprocessing.item() == "true":
    if is_tiling:
        # Calculate threads per tile
        total_threads = int(multiprocessing.MaximumJobs.item())
        threads_per_tile = max(1, total_threads // max_jobs)
        numThreads = threads_per_tile
        ps.environment.update_run_log(f"Using {numThreads} Julia threads for this tile (total threads: {total_threads}, tiles: {max_jobs})")

        # Calculate memory per job
        mem_per_job_gb = calculate_memory_per_job(max_jobs)
    else:
        numThreads = int(multiprocessing.MaximumJobs.item())

    runOmniscape = f"{jlExe} -t {numThreads} {runFile}"
else:
    runOmniscape = f"{jlExe} {runFile}"

ps.environment.update_run_log(f"Executing command: {runOmniscape}")
ps.environment.update_run_log(">>> Starting Omniscape.jl execution (this may take several minutes) <<<")

exit_code = os.system(runOmniscape)

ps.environment.update_run_log(f">>> Omniscape.jl execution completed with exit code: {exit_code} <<<")

if exit_code != 0:
    sys.exit(f"Omniscape.jl failed with exit code {exit_code}. Check Julia output above for errors.")



# ============================================================================
# POST-PROCESSING: REMOVE BUFFER FROM OUTPUTS
# ============================================================================

if buffer_pixels > 0:
    ps.environment.progress_bar(message="Removing buffer from outputs", report_type="message")
    ps.environment.update_run_log(f"=== POST-PROCESSING: Removing {buffer_pixels}-pixel buffer from outputs ===")

    # List of possible output files
    output_files = ['cum_currmap.tif', 'normalized_cum_currmap.tif', 'flow_potential.tif']

    for output_file in output_files:
        buffered_output = os.path.join(dataPath, "omniscape_outputSpatial", output_file)
        ps.environment.update_run_log(f"Checking for output: {output_file}")

        if os.path.exists(buffered_output):
            ps.environment.update_run_log(f"  Found {output_file}, removing buffer...")
            # Crop back to original (pre-buffer) extent
            temp_cropped = buffered_output.replace('.tif', '_cropped.tif')
            crop_to_original(buffered_output, temp_cropped, original_bounds, buffer_pixels)

            # Replace buffered with cropped
            ps.environment.update_run_log(f"  Replacing buffered output with cropped version...")
            os.remove(buffered_output)
            os.rename(temp_cropped, buffered_output)

            ps.environment.update_run_log(f"  Removed buffer from {output_file}")
        else:
            ps.environment.update_run_log(f"  {output_file} not found (may not have been generated)")

    ps.environment.update_run_log(f"=== POST-PROCESSING: Buffer removal complete ===")

# ============================================================================
# END BUFFER REMOVAL
# ============================================================================



# ============================================================================
# POST-PROCESSING: EXTEND TILE OUTPUTS TO FULL EXTENT
# ============================================================================

if is_tiling:
    ps.environment.progress_bar(message="Extending tile outputs to full extent", report_type="message")
    ps.environment.update_run_log(f"=== POST-PROCESSING: Extending tile outputs to full extent ===")

    # List of possible output files
    output_files = ['cum_currmap.tif', 'normalized_cum_currmap.tif', 'flow_potential.tif', 'classified_resistance.tif']

    for output_file in output_files:
        tile_output = os.path.join(dataPath, "omniscape_outputSpatial", output_file)
        ps.environment.update_run_log(f"Checking for output: {output_file}")

        if os.path.exists(tile_output):
            ps.environment.update_run_log(f"  Found {output_file}, extending to full extent...")
            # Extend to full extent (critical for SyncroSim merging)
            temp_extended = tile_output.replace('.tif', '_extended.tif')
            extend_tile_to_full_extent(
                tile_output,
                temp_extended,
                (full_extent_info['width'], full_extent_info['height']),
                full_extent_info['transform'],
                full_extent_info['crs']
            )

            # Replace tile with extended
            ps.environment.update_run_log(f"  Replacing tile output with extended version...")
            os.remove(tile_output)
            os.rename(temp_extended, tile_output)

            ps.environment.update_run_log(f"  Extended {output_file} to full extent")
        else:
            ps.environment.update_run_log(f"  {output_file} not found (may not have been generated)")

    ps.environment.update_run_log(f"=== POST-PROCESSING: Extent extension complete ===")

# ============================================================================
# END POST-PROCESSING
# ============================================================================



# Create output datasheets ----------------------------------------------------------------------

myOutput = myScenario.datasheets(name = "omniscape_outputSpatial")

if outputOptions.writeRawCurrmap.item() == "true":
    myOutput.cumCurrmap = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "cum_currmap.tif"))

if generalOptions.calcFlowPotential.item() == "true":
    myOutput.flowPotential = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "flow_potential.tif"))

if generalOptions.calcNormalizedCurrent.item() == "true":
    myOutput.normalizedCumCurrmap = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "normalized_cum_currmap.tif"))

if (os.path.isfile(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "classified_resistance.tif"))) & (resistanceOptions.writeReclassifiedResistance.item() == "true"):
    myOutput.classifiedResistance = pd.Series(os.path.join(wrkDir, "Scenario-" + repr(myScenarioID), "omniscape_outputSpatial", "classified_resistance.tif"))
else:
    myOutput.classifiedResistance = pd.Series(original_resistance_path)



# Save outputs to SyncroSim ---------------------------------------------------------------------

myParentScenario.save_datasheet(name = "omniscape_outputSpatial", data = myOutput)


