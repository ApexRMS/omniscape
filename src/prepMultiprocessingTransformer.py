## omniscape - PrepMultiprocessing Transformer
## Creates spatial tile grid for multiprocessing

import pysyncrosim as ps
import pandas as pd
import numpy as np
import rasterio
import os
import sys
import math

ps.environment.progress_bar(message="Initializing PrepMultiprocessing", report_type="message")

# Initialize environment
e = ps.environment._environment()
wrkDir = e.data_directory.item()

myLibrary = ps.Library()
myScenarioID = e.scenario_id.item()
myScenario = myLibrary.scenarios(myScenarioID)

# Load configuration datasheets
requiredData = myScenario.datasheets(name="omniscape_Required", show_full_paths=True)
tilingOptions = myScenario.datasheets(name="omniscape_TilingOptions")

# Validation: Check if resistance file exists (template raster)
if requiredData.empty or requiredData.resistanceFile.empty or pd.isna(requiredData.resistanceFile.item()):
    sys.exit("PrepMultiprocessing requires 'Resistance file' to be specified first. Please configure inputs before running this transformer.")

resistancePath = requiredData.resistanceFile.item()
if not os.path.isfile(resistancePath):
    sys.exit(f"Resistance file does not exist: {resistancePath}")

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

# Determine tile count
MIN_PIXELS_FOR_TILING = 10000  # Don't tile small rasters

if valid_pixels < MIN_PIXELS_FOR_TILING:
    ps.environment.update_run_log(
        f"Raster too small for multiprocessing ({valid_pixels:,} pixels < {MIN_PIXELS_FOR_TILING:,}). "
        "Skipping tile generation. Omniscape will run in single-process mode.",
        report_type="message"
    )
    sys.exit(0)  # Exit gracefully without creating grid

# Check if user specified manual tile count
manual_tile_count = None
if not tilingOptions.empty and 'TileCount' in tilingOptions.columns:
    if not pd.isna(tilingOptions.TileCount.item()):
        manual_tile_count = int(tilingOptions.TileCount.item())
        if manual_tile_count < 1:
            sys.exit("TileCount must be at least 1")
        elif manual_tile_count == 1:
            ps.environment.update_run_log("TileCount=1 specified. Disabling multiprocessing. Omniscape will run in single-process mode.", report_type="message")
            sys.exit(0)

# Auto-calculate optimal tile count
if manual_tile_count is None:
    # Target: Keep tiles around 100K to 1M pixels each
    TARGET_TILE_SIZE = 100000  # 100K pixels per tile

    # Calculate required tiles
    num_tiles_needed = max(2, math.ceil(valid_pixels / TARGET_TILE_SIZE))

    # Round to nearest perfect square for square grids
    # Prefer: 4, 9, 16, 25, 36, 49, 64, 81, 100
    sqrt_tiles = math.sqrt(num_tiles_needed)
    rounded_sqrt = max(2, round(sqrt_tiles))
    tile_count = rounded_sqrt ** 2

    # Cap at 100 tiles maximum
    tile_count = min(tile_count, 100)

    ps.environment.update_run_log(f"Auto-calculated tile count: {tile_count} ({rounded_sqrt}x{rounded_sqrt} grid)")
else:
    tile_count = manual_tile_count
    ps.environment.update_run_log(f"Using manual tile count: {tile_count}")

ps.environment.progress_bar(message=f"Generating {tile_count}-tile grid", report_type="message")

# Create tile grid raster
tile_dim_rows = int(math.ceil(math.sqrt(tile_count)))
tile_dim_cols = int(math.ceil(tile_count / tile_dim_rows))

# Calculate tile dimensions in pixels
tile_height = int(math.ceil(height / tile_dim_rows))
tile_width = int(math.ceil(width / tile_dim_cols))

# Create grid array initialized to -9999 (NoData)
grid = np.full((height, width), -9999, dtype=np.int32)

# Assign tile IDs
tile_id = 1
for row in range(tile_dim_rows):
    for col in range(tile_dim_cols):
        if tile_id > tile_count:
            break

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

ps.environment.progress_bar(message="PrepMultiprocessing complete", report_type="message")
ps.environment.update_run_log("PrepMultiprocessing => Complete")
