## omniscape - PrepMultiprocessing Transformer
## Creates spatial tile grid for multiprocessing

import pysyncrosim as ps
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
import os
import sys
import math
import json
import shutil
from datetime import datetime

ps.environment.progress_bar(message="Initializing PrepMultiprocessing", report_type="message")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def crop_raster_to_extent(input_path, output_path, extent, transform, crs):
    """
    Crop raster to specified extent.

    Args:
        input_path: Path to input raster
        output_path: Path for cropped output
        extent: Tuple (row_start, row_end, col_start, col_end)
        transform: Affine transform from source
        crs: CRS from source
    """
    row_start, row_end, col_start, col_end = extent

    with rasterio.open(input_path) as src:
        # Create window
        window = Window(col_start, row_start,
                       col_end - col_start,
                       row_end - row_start)

        # Read windowed data
        data = src.read(1, window=window)

        # Calculate new transform
        new_transform = rasterio.windows.transform(window, src.transform)

        # Write cropped raster
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=data.dtype,
            crs=crs, transform=new_transform,
            nodata=src.nodata, compress='lzw'
        ) as dst:
            dst.write(data, 1)


def calculate_optimal_grid(width, height, desired_tile_count):
    """
    Calculate optimal grid dimensions that match raster aspect ratio.

    Args:
        width: Raster width in pixels
        height: Raster height in pixels
        desired_tile_count: User-specified or auto-calculated tile count

    Returns:
        tuple: (rows, cols, actual_tile_count, description)
    """
    aspect_ratio = width / height

    # Find best grid by testing combinations near desired count
    best_grid = None
    best_score = float('inf')

    # Test grid dimensions from 1x1 up to reasonable limits
    max_dim = min(int(math.sqrt(desired_tile_count * 4)), 20)  # Don't go crazy

    for rows in range(1, max_dim + 1):
        for cols in range(1, max_dim + 1):
            actual_count = rows * cols

            # Skip grids that are too far from desired count
            if abs(actual_count - desired_tile_count) > max(3, desired_tile_count * 0.3):
                continue

            grid_aspect = cols / rows

            # Score based on:
            # 1. How close to desired tile count (weight: 1.0)
            # 2. How well grid aspect matches raster aspect (weight: 3.0)
            # Higher aspect weight ensures we don't create long thin strips on square rasters
            count_penalty = abs(actual_count - desired_tile_count)
            aspect_penalty = abs(grid_aspect - aspect_ratio) * 3.0
            score = count_penalty + aspect_penalty

            if score < best_score:
                best_score = score
                best_grid = (rows, cols, actual_count)

    if best_grid is None:
        # Fallback to square grid
        dim = int(math.ceil(math.sqrt(desired_tile_count)))
        best_grid = (dim, dim, dim * dim)

    rows, cols, actual_count = best_grid
    description = f"{rows}×{cols} grid"

    return rows, cols, actual_count, description


def crop_and_buffer_raster(input_path, output_path, tile_extent, buffer_pixels,
                           raster_width, raster_height, transform, crs):
    """
    Crop raster with buffer using real overlapping data only.
    - Use real overlapping data where available (internal boundaries)
    - No padding beyond raster boundaries (only real data)

    Args:
        input_path: Path to input raster
        output_path: Path for output
        tile_extent: Tuple (row_start, row_end, col_start, col_end) - original tile extent
        buffer_pixels: Number of pixels to buffer on each side
        raster_width: Full raster width
        raster_height: Full raster height
        transform: Affine transform from full raster
        crs: CRS from full raster

    Returns:
        tuple: (actual_extent, buffer_info)
    """
    row_start, row_end, col_start, col_end = tile_extent

    # Calculate ideal buffered extent
    row_start_buffered = row_start - buffer_pixels
    row_end_buffered = row_end + buffer_pixels
    col_start_buffered = col_start - buffer_pixels
    col_end_buffered = col_end + buffer_pixels

    # Clip to raster bounds (only crop what exists in full raster)
    row_start_clipped = max(0, row_start_buffered)
    row_end_clipped = min(raster_height, row_end_buffered)
    col_start_clipped = max(0, col_start_buffered)
    col_end_clipped = min(raster_width, col_end_buffered)

    # Crop real data from full raster
    with rasterio.open(input_path) as src:
        # Create window for cropping
        window = Window(col_start_clipped, row_start_clipped,
                       col_end_clipped - col_start_clipped,
                       row_end_clipped - row_start_clipped)

        # Read data
        data = src.read(1, window=window)

        # Calculate transform for cropped area
        cropped_transform = rasterio.windows.transform(window, src.transform)

        # Check if we hit raster boundaries (for logging purposes)
        hit_top = row_start_buffered < 0
        hit_bottom = row_end_buffered > raster_height
        hit_left = col_start_buffered < 0
        hit_right = col_end_buffered > raster_width

        # Use only real data (no padding)
        final_data = data
        final_transform = cropped_transform

        if hit_top or hit_bottom or hit_left or hit_right:
            boundaries_hit = []
            if hit_top: boundaries_hit.append("top")
            if hit_bottom: boundaries_hit.append("bottom")
            if hit_left: boundaries_hit.append("left")
            if hit_right: boundaries_hit.append("right")
            buffer_info = f"real data only (clipped at {', '.join(boundaries_hit)})"
        else:
            buffer_info = "all real overlapping data"

        # Write output
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=final_data.shape[0],
            width=final_data.shape[1],
            count=1,
            dtype=src.dtypes[0],
            crs=crs,
            transform=final_transform,
            nodata=src.nodata,
            compress='lzw'
        ) as dst:
            dst.write(final_data, 1)

        # Return the actual clipped extent (what was really written)
        # NOT the buffered extent (which may be negative or beyond bounds)
        actual_extent = {
            "row_start": row_start_clipped,
            "row_end": row_end_clipped,
            "col_start": col_start_clipped,
            "col_end": col_end_clipped
        }

        return actual_extent, buffer_info


# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Initialize environment
e = ps.environment._environment()
wrkDir = e.data_directory.item()

myLibrary = ps.Library()
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)

# Load configuration datasheets
requiredData = myScenario.datasheets(name="omniscape_Required", show_full_paths=True)
tilingOptions = myScenario.datasheets(name="omniscape_TilingOptions")
generalOptions = myScenario.datasheets(name="omniscape_GeneralOptions")
multiprocessing = myScenario.datasheets(name="core_Multiprocessing")

# Validation: Check if resistance file exists (template raster)
if requiredData.empty or requiredData.resistanceFile.empty or pd.isna(requiredData.resistanceFile.item()):
    sys.exit("PrepMultiprocessing requires 'Resistance file' to be specified first. Please configure inputs before running this transformer.")

resistancePath = requiredData.resistanceFile.item()
if not os.path.isfile(resistancePath):
    sys.exit(f"Resistance file does not exist: {resistancePath}")

# Get source file path if it exists (will be None if not specified)
sourcePath = None
if not requiredData.empty and 'sourceFile' in requiredData.columns:
    if not pd.isna(requiredData.sourceFile.item()):
        source_file = requiredData.sourceFile.item()
        if os.path.isfile(source_file):
            sourcePath = source_file

# Get Omniscape analysis parameters
# Radius (required parameter)
if requiredData.empty or 'radius' not in requiredData.columns or pd.isna(requiredData.radius.item()):
    sys.exit("PrepMultiprocessing requires 'Radius' to be specified first. Please configure Omniscape parameters before running tiling.")
radius = int(requiredData.radius.item())
ps.environment.update_run_log(f"Omniscape radius: {radius} pixels")

# Block size (optional parameter, default=1)
block_size = 1
if not generalOptions.empty and 'blockSize' in generalOptions.columns:
    if not pd.isna(generalOptions.blockSize.item()):
        block_size = int(generalOptions.blockSize.item())
ps.environment.update_run_log(f"Omniscape block_size: {block_size}")

# Get available cores for parallelization
available_cores = 1
if not multiprocessing.empty and 'MaximumJobs' in multiprocessing.columns:
    if not pd.isna(multiprocessing.MaximumJobs.item()):
        available_cores = int(multiprocessing.MaximumJobs.item())
ps.environment.update_run_log(f"Available cores (MaximumJobs): {available_cores}")

# Get parallelization intensity
intensity = "Auto"  # default
if not tilingOptions.empty and 'ParallelizationIntensity' in tilingOptions.columns:
    if not pd.isna(tilingOptions.ParallelizationIntensity.item()):
        user_intensity = tilingOptions.ParallelizationIntensity.item().strip()
        # Validate intensity value
        valid_intensities = ["Auto", "Conservative", "Balanced", "Aggressive"]
        if user_intensity in valid_intensities:
            intensity = user_intensity
        else:
            ps.environment.update_run_log(
                f"WARNING: Invalid ParallelizationIntensity '{user_intensity}'. "
                f"Valid values: {', '.join(valid_intensities)}. Using default 'Auto'."
            )
ps.environment.update_run_log(f"Parallelization intensity: {intensity}")

# Get buffer multiplier
buffer_multiplier = 1.0  # default
if not tilingOptions.empty and 'BufferMultiplier' in tilingOptions.columns:
    if not pd.isna(tilingOptions.BufferMultiplier.item()):
        buffer_multiplier = float(tilingOptions.BufferMultiplier.item())

buffer_pixels = int(radius * buffer_multiplier)

ps.environment.update_run_log(
    f"Buffer configuration: {buffer_pixels} pixels ({buffer_multiplier:.1f}× radius of {radius})"
)

# Validate buffer vs radius
if buffer_multiplier < 1.0:
    ps.environment.update_run_log(
        f"WARNING: Buffer multiplier ({buffer_multiplier:.1f}) < 1.0. "
        f"This may cause edge effects at tile boundaries. "
        f"Recommend setting to 1.0 or higher."
    )

ps.environment.update_run_log(f"Buffer strategy: real overlapping data only (no padding beyond raster boundary)")

ps.environment.progress_bar(message="Analyzing raster dimensions", report_type="message")

# Load raster to analyze dimensions
with rasterio.open(resistancePath) as src:
    width = src.width
    height = src.height
    crs = src.crs
    transform = src.transform
    bounds = src.bounds
    nodata = src.nodata

    # Read data to calculate actual analysis area (non-nodata pixels)
    data = src.read(1)

    # Count valid pixels
    if nodata is not None:
        valid_pixels = np.sum(data != nodata)
    else:
        valid_pixels = np.sum(~np.isnan(data))

    total_pixels = width * height

ps.environment.update_run_log(f"Raster dimensions: {width} x {height} = {total_pixels:,} pixels ({valid_pixels:,} valid)")

# ============================================================================
# INTELLIGENT TILE COUNT CALCULATION
# Based on Omniscape parameters (block_size, radius) and system resources
# ============================================================================

ps.environment.update_run_log("Calculating optimal tile configuration based on analysis parameters...")

# Minimum tile dimensions based on block_size
# Tiles should be at least 7x block_size in each dimension for efficient processing
MIN_TILE_DIMENSION_MULTIPLIER = 7
min_tile_dimension = max(block_size * MIN_TILE_DIMENSION_MULTIPLIER, 100)  # At least 100 pixels
min_tile_pixels = min_tile_dimension ** 2

# Also need to account for buffer - effective tile size after buffering
# Effective dimension = tile_dimension - (2 * buffer_pixels)
# Want: effective_dimension >= min_tile_dimension
# So: tile_dimension >= min_tile_dimension + (2 * buffer_pixels)
min_tile_dimension_with_buffer = min_tile_dimension + (2 * buffer_pixels)
min_tile_pixels_with_buffer = min_tile_dimension_with_buffer ** 2

# Don't tile if raster is too small
MIN_PIXELS_FOR_TILING = min_tile_pixels_with_buffer * 2  # Need at least 2 tiles to be worthwhile

if valid_pixels < MIN_PIXELS_FOR_TILING:
    ps.environment.update_run_log(
        f"Raster too small for multiprocessing ({valid_pixels:,} pixels < {MIN_PIXELS_FOR_TILING:,}). "
        f"Minimum tile size based on block_size={block_size} and buffer={buffer_pixels} "
        f"requires {min_tile_pixels_with_buffer:,} pixels per tile. "
        "Skipping tile generation. Omniscape will run in single-process mode."
    )
    sys.exit(0)

# Calculate optimal tile count based on multiple constraints

# 1. Maximum tiles based on minimum tile size constraint
max_tile_count_size = int(valid_pixels // min_tile_pixels_with_buffer)

# 2. Calculate target tiles based on intensity setting
intensity_multipliers = {
    "Conservative": 1.0,  # Match available cores
    "Balanced": 2.0,      # 2× cores for load balancing
    "Aggressive": 4.0,    # 4× cores for maximum parallelization
    "Auto": 2.0           # Smart default (same as Balanced)
}

multiplier = intensity_multipliers.get(intensity, 2.0)
target_tile_count_cores = int(available_cores * multiplier)

# 3. Also consider a reasonable target tile size for efficiency
# Aim for tiles that are at least 2x minimum size, or 250K pixels
TARGET_TILE_SIZE = max(min_tile_pixels_with_buffer * 2, 250000)
target_tile_count_size = max(2, math.ceil(valid_pixels / TARGET_TILE_SIZE))

# Choose the most conservative (smallest) tile count from all constraints
# This ensures tiles are large enough and we don't over-parallelize
desired_tile_count = min(
    max_tile_count_size,      # Size constraint
    target_tile_count_cores,  # Intensity/cores constraint
    target_tile_count_size,   # Efficiency constraint
    100                        # Absolute maximum cap
)
desired_tile_count = max(2, desired_tile_count)  # At least 2 tiles

# Log the decision process
ps.environment.update_run_log("Tile count calculation constraints:")
ps.environment.update_run_log(f"  - Minimum tile dimension: {min_tile_dimension} px (7× block_size={block_size})")
ps.environment.update_run_log(f"  - Buffer requirement: {buffer_pixels} px on each side")
ps.environment.update_run_log(f"  - Minimum tile dimension with buffer: {min_tile_dimension_with_buffer} px")
ps.environment.update_run_log(f"  - Minimum tile area: {min_tile_pixels_with_buffer:,} pixels")
ps.environment.update_run_log(f"  - Maximum tiles (size constraint): {max_tile_count_size}")
ps.environment.update_run_log(f"  - Target tiles ({intensity} intensity, {available_cores} cores × {multiplier:.1f}): {target_tile_count_cores}")
ps.environment.update_run_log(f"  - Target tiles (based on efficiency): {target_tile_count_size}")
ps.environment.update_run_log(f"  - Final calculated tile count: {desired_tile_count}")

# Calculate optimal grid based on raster aspect ratio
tile_dim_rows, tile_dim_cols, tile_count, grid_desc = calculate_optimal_grid(
    width, height, desired_tile_count
)

# Log adjustment if needed (grid optimization may change tile count slightly)
if tile_count != desired_tile_count:
    ps.environment.update_run_log(
        f"Adjusted tile count from {desired_tile_count} to {tile_count} ({grid_desc}) "
        f"for optimal aspect ratio and even distribution"
    )
else:
    ps.environment.update_run_log(f"Using {tile_count} tiles ({grid_desc})")

# Calculate actual tile dimensions for validation
tile_height = int(math.ceil(height / tile_dim_rows))
tile_width = int(math.ceil(width / tile_dim_cols))
actual_tile_pixels = tile_height * tile_width

# Validate tile size meets minimum requirements
effective_tile_height = tile_height - (2 * buffer_pixels)
effective_tile_width = tile_width - (2 * buffer_pixels)
effective_tile_pixels = effective_tile_height * effective_tile_width

ps.environment.update_run_log(
    f"Tile dimensions: {tile_width} × {tile_height} pixels ({actual_tile_pixels:,} total)"
)
ps.environment.update_run_log(
    f"Effective tile dimensions after buffering: {effective_tile_width} × {effective_tile_height} pixels "
    f"({effective_tile_pixels:,} total)"
)

# Warn if tiles might be too small
if effective_tile_pixels < min_tile_pixels:
    ps.environment.update_run_log(
        f"WARNING: Effective tile size ({effective_tile_pixels:,}) is smaller than recommended minimum "
        f"({min_tile_pixels:,}) for block_size={block_size}. Consider reducing buffer or tile count."
    )

ps.environment.progress_bar(message=f"Generating {tile_count}-tile grid", report_type="message")

# Create grid array initialized to -9999 (NoData)
grid = np.full((height, width), -9999, dtype=np.int32)

# Assign tile IDs (all tiles will be filled since grid matches tile_count exactly)
tile_id = 1
for row in range(tile_dim_rows):
    for col in range(tile_dim_cols):
        row_start = row * tile_height
        row_end = min((row + 1) * tile_height, height)
        col_start = col * tile_width
        col_end = min((col + 1) * tile_width, width)

        grid[row_start:row_end, col_start:col_end] = tile_id
        tile_id += 1

# Mask to analysis area (set nodata areas to -9999)
if nodata is not None:
    grid[data == nodata] = -9999
else:
    grid[np.isnan(data)] = -9999

# Save tile grid
dataPath = os.path.join(wrkDir, f"Scenario-{myScenarioID}")
os.makedirs(dataPath, exist_ok=True)

tile_size_k = int(valid_pixels / tile_count / 1000)
grid_filename = f"smpGrid-{tile_count}-{tile_size_k}K.tif"
grid_path = os.path.join(dataPath, grid_filename)

ps.environment.progress_bar(message="Writing tile grid raster", report_type="message")

# Write grid raster
with rasterio.open(
    grid_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=rasterio.int32,
    crs=crs,
    transform=transform,
    nodata=-9999,
    compress='lzw'
) as dst:
    dst.write(grid, 1)

ps.environment.update_run_log(f"Tile grid created: {grid_filename}")
ps.environment.update_run_log(f"Grid saved to: {grid_path}")

# Save to core_SpatialMultiprocessing datasheet
smp_data = pd.DataFrame({
    'MaskFileName': [grid_path]
})
myScenario.save_datasheet(name="core_SpatialMultiprocessing", data=smp_data)


# ============================================================================
# CREATE PRE-PROCESSED TILES
# ============================================================================

ps.environment.progress_bar(message="Creating pre-processed tiles", report_type="message")

# Create tiles directory
tiles_dir = os.path.join(dataPath, "OmniscapeTiles")
os.makedirs(tiles_dir, exist_ok=True)
ps.environment.update_run_log(f"Tiles directory: {tiles_dir}")

# Copy grid to tiles directory
tiles_grid_path = os.path.join(tiles_dir, os.path.basename(grid_path))
shutil.copy(grid_path, tiles_grid_path)

# Prepare tile manifest
manifest = {
    "version": "1.0",
    "created_at": datetime.now().isoformat(),
    "tile_count": tile_count,
    "buffer_pixels": buffer_pixels,
    "grid_raster": tiles_grid_path,
    "full_extent": {
        "width": width,
        "height": height,
        "transform": list(transform),
        "crs": str(crs)
    },
    "tiles": []
}

# Process each tile
tile_id = 1
for row in range(tile_dim_rows):
    for col in range(tile_dim_cols):
        ps.environment.progress_bar(
            message=f"Processing tile {tile_id}/{tile_count}",
            report_type="message"
        )

        # Calculate tile extent
        row_start = row * tile_height
        row_end = min((row + 1) * tile_height, height)
        col_start = col * tile_width
        col_end = min((col + 1) * tile_width, width)

        tile_pixels = (row_end - row_start) * (col_end - col_start)
        ps.environment.update_run_log(
            f"Tile {tile_id}: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}] "
            f"({tile_pixels:,} pixels)"
        )

        # Crop and buffer resistance using HYBRID approach
        tile_resistance_path = os.path.join(tiles_dir, f"tile-{tile_id}-resistance.tif")

        if buffer_pixels > 0:
            # Use real overlapping data only (no padding beyond raster boundary)
            ps.environment.update_run_log(f"  Applying {buffer_pixels}-pixel buffer (real overlapping data only)")

            buffered_extent, buffer_info = crop_and_buffer_raster(
                resistancePath,
                tile_resistance_path,
                (row_start, row_end, col_start, col_end),
                buffer_pixels,
                width,
                height,
                transform,
                crs
            )

            ps.environment.update_run_log(f"    Resistance: {buffer_info}")

            # Crop and buffer source (if exists)
            tile_source_path = None
            if sourcePath:
                tile_source_path = os.path.join(tiles_dir, f"tile-{tile_id}-source.tif")

                _, source_buffer_info = crop_and_buffer_raster(
                    sourcePath,
                    tile_source_path,
                    (row_start, row_end, col_start, col_end),
                    buffer_pixels,
                    width,
                    height,
                    transform,
                    crs
                )

                ps.environment.update_run_log(f"    Source: {source_buffer_info}")
        else:
            # No buffer - just crop
            crop_raster_to_extent(
                resistancePath,
                tile_resistance_path,
                (row_start, row_end, col_start, col_end),
                transform,
                crs
            )

            tile_source_path = None
            if sourcePath:
                tile_source_path = os.path.join(tiles_dir, f"tile-{tile_id}-source.tif")
                crop_raster_to_extent(
                    sourcePath,
                    tile_source_path,
                    (row_start, row_end, col_start, col_end),
                    transform,
                    crs
                )

            buffered_extent = None

        # Add to manifest
        manifest["tiles"].append({
            "tile_id": tile_id,
            "resistance_path": tile_resistance_path,
            "source_path": tile_source_path,
            "original_extent": {
                "row_start": row_start, "row_end": row_end,
                "col_start": col_start, "col_end": col_end
            },
            "buffered_extent": buffered_extent,
            "is_buffered": buffer_pixels > 0
        })

        tile_id += 1

# Save manifest
manifest_path = os.path.join(tiles_dir, "tile_manifest.json")
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

ps.environment.update_run_log(f"Tile manifest saved: {manifest_path}")

# ============================================================================
# TILING CONFIGURATION SUMMARY
# ============================================================================

ps.environment.update_run_log("")
ps.environment.update_run_log("=" * 70)
ps.environment.update_run_log("TILING CONFIGURATION SUMMARY")
ps.environment.update_run_log("=" * 70)
ps.environment.update_run_log(f"Raster: {width} × {height} pixels ({valid_pixels:,} valid)")
ps.environment.update_run_log(f"")
ps.environment.update_run_log(f"Tile Configuration:")
ps.environment.update_run_log(f"  Grid layout:           {grid_desc} = {tile_count} tiles")
ps.environment.update_run_log(f"  Tile dimensions:       {tile_width} × {tile_height} pixels")
ps.environment.update_run_log(f"  Buffer applied:        {buffer_pixels} pixels ({buffer_multiplier:.1f}× radius)")
ps.environment.update_run_log(f"  Effective tile size:   {effective_tile_width} × {effective_tile_height} pixels")
ps.environment.update_run_log(f"")
ps.environment.update_run_log(f"Parallelization:")
ps.environment.update_run_log(f"  Available cores:       {available_cores}")
ps.environment.update_run_log(f"  Intensity setting:     {intensity}")
ps.environment.update_run_log(f"  Simultaneous tiles:    {min(tile_count, available_cores)}")

# Calculate estimated speedup
# Account for overhead and diminishing returns
tiles_per_batch = math.ceil(tile_count / available_cores)
ideal_speedup = available_cores
overhead_factor = 0.85  # 15% overhead for I/O, merging, etc.
actual_speedup = min(tile_count, ideal_speedup) * overhead_factor

ps.environment.update_run_log(f"  Estimated speedup:     {actual_speedup:.1f}× vs single-core")
ps.environment.update_run_log(f"")

# Memory estimate (rough calculation)
bytes_per_pixel = 8  # Assume float64 for safety
tile_memory_mb = (tile_width * tile_height * bytes_per_pixel) / (1024 * 1024)
peak_memory_mb = tile_memory_mb * min(tile_count, available_cores)

ps.environment.update_run_log(f"Memory Estimates:")
ps.environment.update_run_log(f"  Per tile:              ~{tile_memory_mb:.1f} MB")
ps.environment.update_run_log(f"  Peak (all cores):      ~{peak_memory_mb:.1f} MB")
ps.environment.update_run_log("=" * 70)
ps.environment.update_run_log("")

ps.environment.progress_bar(message="PrepMultiprocessing complete", report_type="message")
ps.environment.update_run_log("PrepMultiprocessing => Complete")
