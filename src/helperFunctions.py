"""
Helper functions for Omniscape SyncroSim package.
Handles tiling, buffering, and raster processing operations.
"""

import os
import re
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import pysyncrosim as ps


# ============================================================================
# TILING AND MANIFEST FUNCTIONS
# ============================================================================

def load_tile_manifest(scenario_id, wrkDir):
    """
    Load tile manifest created by prep transformer.

    Handles both regular and multiprocessing (Job-X.ssim) scenarios.
    In multiprocessing mode, looks for manifest in parent library's data directory.

    Returns:
        dict: Manifest data, or None if no manifest exists
    """
    # Check if we're in a Job-X.ssim library (multiprocessing mode)
    if "MultiProc" in wrkDir and "Job-" in wrkDir:
        # Extract parent scenario ID and library name from path
        # Example path: D:\SyncroSim\omniscape-tiling.ssim.temp\MultiProc\Scenario-20\t10\Job-4.ssim.data

        # Find parent scenario ID in path
        scenario_match = re.search(r'Scenario-(\d+)', wrkDir)
        if scenario_match:
            parent_scenario_id = scenario_match.group(1)

            # Reconstruct parent library data directory
            # Replace .ssim.temp/MultiProc/... with .ssim.data
            lib_base = wrkDir.split('.ssim.temp')[0]
            parent_wrkDir = f"{lib_base}.ssim.data"

            manifest_path = os.path.join(
                parent_wrkDir, f"Scenario-{parent_scenario_id}", "OmniscapeTiles", "tile_manifest.json"
            )

            ps.environment.update_run_log(f"Multiprocessing mode: Looking for manifest in parent library at {manifest_path}")
        else:
            # Fallback to local path
            manifest_path = os.path.join(
                wrkDir, f"Scenario-{scenario_id}", "OmniscapeTiles", "tile_manifest.json"
            )
    else:
        # Regular mode: look in current working directory
        manifest_path = os.path.join(
            wrkDir, f"Scenario-{scenario_id}", "OmniscapeTiles", "tile_manifest.json"
        )

    if not os.path.exists(manifest_path):
        ps.environment.update_run_log(f"Manifest not found at: {manifest_path}")
        return None

    ps.environment.update_run_log(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, 'r') as f:
        return json.load(f)


def determine_execution_mode(myLibrary, manifest):
    """
    Determine if running in loop or multiprocessing mode.

    Returns:
        tuple: (mode, tile_id_to_process, all_tile_ids)
            - mode: "loop" or "multiprocessing"
            - tile_id_to_process: Specific tile ID (multiprocessing) or None (loop)
            - all_tile_ids: List of all tile IDs
    """
    lib_name = os.path.basename(myLibrary.name)

    # SyncroSim multiprocessing mode: library name is "Job-X.ssim"
    if lib_name.startswith("Job-") and lib_name.endswith(".ssim"):
        try:
            tile_id = int(lib_name.replace("Job-", "").replace(".ssim", ""))
            all_tile_ids = [t["tile_id"] for t in manifest["tiles"]]
            ps.environment.update_run_log(
                f"Multiprocessing mode detected: Processing tile {tile_id} of {len(all_tile_ids)}"
            )
            return ("multiprocessing", tile_id, all_tile_ids)
        except ValueError:
            ps.environment.update_run_log(f"Warning: Could not extract tile ID from: {lib_name}")
            # Fall through to loop mode

    # Loop mode: Process all tiles sequentially
    all_tile_ids = [t["tile_id"] for t in manifest["tiles"]]
    ps.environment.update_run_log(
        f"Loop mode: Will process {len(all_tile_ids)} tiles sequentially"
    )
    return ("loop", None, all_tile_ids)


# ============================================================================
# BUFFER PROCESSING FUNCTIONS
# ============================================================================

def crop_buffer_from_output(buffered_path, output_path, original_extent, buffer_pixels):
    """
    Remove buffer from output raster.

    Args:
        buffered_path: Path to buffered raster
        output_path: Path for cropped output
        original_extent: Dict with row_start, row_end, col_start, col_end
        buffer_pixels: Buffer size used
    """
    with rasterio.open(buffered_path) as src:
        # Create window to extract original extent
        window = Window(
            buffer_pixels,
            buffer_pixels,
            original_extent["col_end"] - original_extent["col_start"],
            original_extent["row_end"] - original_extent["row_start"]
        )

        data = src.read(1, window=window)

        # Calculate transform for cropped area
        new_transform = rasterio.windows.transform(window, src.transform)

        # Write cropped raster
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=src.dtypes[0],
            crs=src.crs, transform=new_transform,
            nodata=src.nodata, compress='lzw'
        ) as dst:
            dst.write(data, 1)


# ============================================================================
# TILE EXTENT FUNCTIONS
# ============================================================================

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


# ============================================================================
# TILE MERGING FUNCTIONS
# ============================================================================

def merge_tile_outputs(tile_output_paths, final_output_path, full_extent_info):
    """
    Merge tile outputs using GDAL VRT.

    Args:
        tile_output_paths: List of tile output raster paths
        final_output_path: Path for final merged output
        full_extent_info: Dict with width, height, transform, crs from manifest
    """
    from osgeo import gdal

    ps.environment.update_run_log(f"Merging {len(tile_output_paths)} tiles into {os.path.basename(final_output_path)}")

    # Create VRT (virtual raster - instant, no data copying)
    vrt_path = final_output_path.replace('.tif', '.vrt')

    vrt_options = gdal.BuildVRTOptions(
        resolution='highest',
        resampleAlg='nearest',
        srcNodata=-9999,
        VRTNodata=-9999
    )

    vrt = gdal.BuildVRT(vrt_path, tile_output_paths, options=vrt_options)
    vrt = None  # Close VRT

    ps.environment.update_run_log(f"VRT created, translating to GeoTIFF...")

    # Translate VRT to final GeoTIFF (efficient streaming)
    gdal.Translate(
        final_output_path,
        vrt_path,
        format='GTiff',
        creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    )

    # Clean up VRT
    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    ps.environment.update_run_log(f"Merge complete: {os.path.basename(final_output_path)}")
