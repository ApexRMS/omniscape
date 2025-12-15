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


def buffer_raster(input_path, output_path, buffer_pixels):
    """
    Create buffered version of raster by extending edges.

    Args:
        input_path: Path to input raster
        output_path: Path for buffered output
        buffer_pixels: Number of pixels to buffer on each side
    """
    with rasterio.open(input_path) as src:
        data = src.read(1)

        # Pad with edge replication
        buffered_data = np.pad(data, buffer_pixels, mode='edge')

        # Calculate new transform (shift origin)
        old_transform = src.transform
        new_transform = rasterio.Affine(
            old_transform.a,
            old_transform.b,
            old_transform.c - (buffer_pixels * old_transform.a),
            old_transform.d,
            old_transform.e,
            old_transform.f - (buffer_pixels * old_transform.e)
        )

        # Write buffered raster
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=buffered_data.shape[0], width=buffered_data.shape[1],
            count=1, dtype=src.dtypes[0],
            crs=src.crs, transform=new_transform,
            nodata=src.nodata, compress='lzw'
        ) as dst:
            dst.write(buffered_data, 1)


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

# Get buffer configuration
buffer_pixels = 0
if not tilingOptions.empty and 'BufferPixels' in tilingOptions.columns:
    if not pd.isna(tilingOptions.BufferPixels.item()):
        buffer_pixels = int(tilingOptions.BufferPixels.item())

ps.environment.update_run_log(f"Buffer configuration: {buffer_pixels} pixels")

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
        if tile_id > tile_count:
            break

        ps.environment.progress_bar(
            message=f"Processing tile {tile_id}/{tile_count}",
            report_type="message"
        )

        # Calculate tile extent
        row_start = row * tile_height
        row_end = min((row + 1) * tile_height, height)
        col_start = col * tile_width
        col_end = min((col + 1) * tile_width, width)

        # Extend last tile in each row to cover full width (prevents gaps)
        # Only extend if this is truly the last column OR if it's the last tile overall
        if col == tile_dim_cols - 1:
            # This tile is in the last column - extend to full width
            col_end = width

        # Extend last tile in each column to cover full height
        if row == tile_dim_rows - 1:
            # This tile is in the last row - extend to full height
            row_end = height

        # Special case: if this is the last tile and it's not in the last column,
        # extend it to cover the remaining width to prevent gaps
        if tile_id == tile_count and col < tile_dim_cols - 1:
            col_end = width
            ps.environment.update_run_log(
                f"  Extending tile {tile_id} to full width to prevent gap "
                f"(last tile in incomplete grid)"
            )

        ps.environment.update_run_log(
            f"Tile {tile_id}: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]"
        )

        # Crop resistance
        tile_resistance_path = os.path.join(tiles_dir, f"tile-{tile_id}-resistance.tif")
        crop_raster_to_extent(
            resistancePath,
            tile_resistance_path,
            (row_start, row_end, col_start, col_end),
            transform,
            crs
        )

        # Crop source (if exists)
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

        # Apply buffer if configured
        buffered_extent = None
        if buffer_pixels > 0:
            ps.environment.update_run_log(f"  Applying {buffer_pixels}-pixel buffer to tile {tile_id}")

            # Buffer resistance
            temp_resistance = tile_resistance_path.replace('.tif', '_unbuffered.tif')
            os.rename(tile_resistance_path, temp_resistance)
            buffer_raster(temp_resistance, tile_resistance_path, buffer_pixels)
            os.remove(temp_resistance)

            # Buffer source
            if tile_source_path:
                temp_source = tile_source_path.replace('.tif', '_unbuffered.tif')
                os.rename(tile_source_path, temp_source)
                buffer_raster(temp_source, tile_source_path, buffer_pixels)
                os.remove(temp_source)

            buffered_extent = {
                "row_start": row_start - buffer_pixels,
                "row_end": row_end + buffer_pixels,
                "col_start": col_start - buffer_pixels,
                "col_end": col_end + buffer_pixels
            }

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
ps.environment.update_run_log(f"Created {tile_count} pre-processed tiles")
if buffer_pixels > 0:
    ps.environment.update_run_log(f"  Each tile buffered by {buffer_pixels} pixels")

ps.environment.progress_bar(message="PrepMultiprocessing complete", report_type="message")
ps.environment.update_run_log("PrepMultiprocessing => Complete")
