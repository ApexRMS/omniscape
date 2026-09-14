"""
Helper functions for Omniscape SyncroSim package.
Handles tiling, buffering, and raster processing operations.
"""

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import pysyncrosim as ps


# Value used throughout omniscape to flag "no data"
NODATA_VALUE = -9999


# ============================================================================
# PROGRESS BAR WRAPPER (Linux Compatibility)
# ============================================================================

def safe_progress_bar(message, report_type="message"):
    """
    Wrapper for ps.environment.progress_bar that falls back to print() on Linux.

    On Linux, progress_bar may throw RuntimeError due to SyncroSim environment issues.
    This wrapper catches the error and prints to console instead.

    Args:
        message: Progress message string (supports f-strings)
        report_type: Type of message (default: "message")
    """
    try:
        ps.environment.progress_bar(message=message, report_type=report_type)
    except RuntimeError:
        # Fallback: just print to console (update_run_log also requires SyncroSim environment)
        print(f"[Progress] {message}")


def safe_update_run_log(message):
    """
    Wrapper for ps.environment.update_run_log that falls back to print().

    ps.environment.update_run_log raises RuntimeError whenever the SSIM_*
    environment variables are absent - which happens on Linux under SyncroSim
    and on any run outside SyncroSim at all. Logging must never be the reason a
    transformer fails, so the message is printed to the console instead.

    Args:
        message: Message string (supports f-strings)
    """
    try:
        ps.environment.update_run_log(message)
    except RuntimeError:
        print(f"[Run log] {message}")


# ============================================================================
# NO-DATA HANDLING
# ============================================================================

def nodata_mask(raster_source, raster_data):
    """Return a boolean array that is True wherever a pixel holds no valid data.

    A raster can flag "no data" in more than one way depending on which
    omniscape code path produced it: a sentinel declared in the file header
    (normally -9999), NaN with no declared sentinel (what spatial tiling
    produces when a tile carries no declared nodata), or -9999 present in the
    pixels but undeclared. All three are tested.
    """
    mask = np.zeros(raster_data.shape, dtype=bool)

    if raster_source.nodata is not None:
        if np.isnan(raster_source.nodata):
            mask |= np.isnan(raster_data)
        else:
            mask |= (raster_data == raster_source.nodata)

    if np.issubdtype(raster_data.dtype, np.floating):
        mask |= np.isnan(raster_data)

    mask |= (raster_data == NODATA_VALUE)

    return mask


# ============================================================================
# CLASSIFICATION
# ============================================================================

def resolve_list_option(value, name_by_id, default):
    """Read a SyncroSim List column tolerating either its numeric ID or its display name.

    Which of the two comes back depends on how the datasheet was read, so
    accepting both keeps callers from having to care.
    """
    if value != value or value is None:          # NaN -> default
        return default
    try:
        return name_by_id[int(value)]
    except (ValueError, TypeError, KeyError):
        name = str(value)
        if name in name_by_id.values():
            return name
        sys.exit("Unrecognised option value: " + repr(value) + ".")


def resolve_boolean_option(value, default):
    """Read a SyncroSim Boolean column tolerating every form it comes back in.

    SyncroSim stores Booleans as integers - 0 for false and a non-zero value
    (conventionally -1) for true - but the same column can also arrive as a
    Python bool or as its "Yes"/"No" display string depending on how the
    datasheet was read. Comparing the raw value against "Yes" therefore reads
    -1 as false and 0 as true, which is exactly backwards, so every form is
    handled here instead. An empty value falls back to the given default.
    """
    if value is None or pd.isna(value):           # empty -> default
        return default

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("yes", "true", "y", "t", "-1", "1"):
            return True
        if text in ("no", "false", "n", "f", "0"):
            return False
        if text == "":
            return default
        sys.exit("Unrecognised Yes/No value: " + repr(value) + ".")

    try:
        return int(value) != 0
    except (ValueError, TypeError):
        sys.exit("Unrecognised Yes/No value: " + repr(value) + ".")


def validate_threshold_bands(threshold_table, min_column, max_column, label,
                             lower_limit=None, upper_limit=None, allow_gaps=False):
    """Check that a set of classification bands tile the range without gaps or overlaps.

    Each row must satisfy min < max, and once sorted the rows must run
    end-to-end: every band's maximum is the next band's minimum. Neither
    condition is enforced by the datasheet itself, and both fail silently at
    run time - an overlap lets whichever band is applied last quietly win, and
    a gap drops every pixel that falls in it to no-data. Catching them here
    turns two invisible wrong answers into one clear message.

    lower_limit / upper_limit, when given, additionally bound the whole range
    (quantiles must lie in 0-1; raw values are unbounded).

    allow_gaps relaxes the end-to-end requirement for callers where falling in
    no band is a meaningful outcome rather than a mistake. Resistance modifiers
    are the case: a pixel matching no band keeps a multiplier of 1.0, so a gap
    means "leave this range alone" and is worth supporting. Overlaps stay fatal
    either way, because nothing makes an overlapping pixel's multiplier
    predictable.
    """
    if threshold_table.empty:
        sys.exit(f"The '{label}' datasheet is required and is empty.")

    bands = []
    for row in threshold_table.itertuples():
        low = float(getattr(row, min_column))
        high = float(getattr(row, max_column))

        if low >= high:
            sys.exit(
                f"Invalid range in '{label}': minimum {low} is not less than maximum {high}.")

        if lower_limit is not None and low < lower_limit:
            sys.exit(
                f"Invalid range in '{label}': minimum {low} is below the allowed "
                f"limit of {lower_limit}.")

        if upper_limit is not None and high > upper_limit:
            sys.exit(
                f"Invalid range in '{label}': maximum {high} is above the allowed "
                f"limit of {upper_limit}.")

        bands.append((low, high))

    bands.sort()
    for (_, previous_high), (next_low, _) in zip(bands, bands[1:]):
        if next_low < previous_high:
            sys.exit(
                f"Overlapping ranges in '{label}': one band ends at {previous_high} "
                f"while another begins at {next_low}. Overlapping bands are applied in an "
                "unpredictable order, so each value must fall in exactly one band.")
        if next_low > previous_high and not allow_gaps:
            sys.exit(
                f"Gap in '{label}': one band ends at {previous_high} and the next begins "
                f"at {next_low}. Values in between would be left unclassified. Make each "
                "band's maximum the next band's minimum.")


def classify_by_quantiles(raster_data, mask, quantile_table):
    """Classify a raster into categories using quantile thresholds.

    quantile_table has one row per category with columns classID, minQuantile
    and maxQuantile (each between 0 and 1). The break VALUES are computed from
    the distribution of valid pixels, so the categories adapt to whatever range
    the raster happens to occupy - the quantile analogue of classifying against
    fixed values. This matters most for a combined surface, whose range depends
    on how many Scenarios went into it and how they were weighted, but it works
    on any raster.

    Intervals are half-open [min, max), except that a maxQuantile of 1 is
    closed so the single largest pixel is not left unclassified. Pixels falling
    in no interval are no-data in the output.

    Returns (class_raster, breaks_table) where breaks_table adds the computed
    minBreakValue / maxBreakValue per category.
    """
    valid = raster_data[~mask]

    if valid.size == 0:
        sys.exit("The raster being classified contains no valid pixels.")

    class_raster = np.full(raster_data.shape, NODATA_VALUE, dtype=np.int16)
    break_rows = []

    for row in quantile_table.itertuples():
        low = float(np.quantile(valid, row.minQuantile))
        high = float(np.quantile(valid, row.maxQuantile))

        if row.maxQuantile >= 1.0:
            selected = (raster_data >= low) & (raster_data <= high) & ~mask
        else:
            selected = (raster_data >= low) & (raster_data < high) & ~mask

        class_raster[selected] = int(row.classID)
        break_rows.append({"classID": int(row.classID),
                           "minQuantile": float(row.minQuantile),
                           "maxQuantile": float(row.maxQuantile),
                           "minBreakValue": low, "maxBreakValue": high})

    return class_raster, pd.DataFrame(break_rows)


def classify_by_values(raster_data, mask, threshold_table):
    """Classify a raster into categories using fixed value thresholds.

    threshold_table has one row per category with columns classID, minValue and
    maxValue, interpreted directly against the raster's own values. Intervals
    are half-open [min, max) so adjoining bands do not both claim their shared
    edge - except the topmost band, which is closed so that pixels sitting
    exactly on the overall maximum are not left unclassified. Without that,
    bands of 0-0.25-0.5-0.75-1 silently drop every pixel whose value is exactly
    1. Pixels falling in no interval, and no-data pixels, are no-data in the
    output.

    Returns (class_raster, breaks_table) with the same shape of breaks_table as
    classify_by_quantiles, so callers can treat the two interchangeably; the
    quantile columns are left empty.
    """
    class_raster = np.full(raster_data.shape, NODATA_VALUE, dtype=np.int16)
    break_rows = []

    highest_bound = max(float(row.maxValue) for row in threshold_table.itertuples())

    for row in threshold_table.itertuples():
        low = float(row.minValue)
        high = float(row.maxValue)

        if high >= highest_bound:
            selected = (raster_data >= low) & (raster_data <= high) & ~mask
        else:
            selected = (raster_data >= low) & (raster_data < high) & ~mask
        class_raster[selected] = int(row.classID)
        break_rows.append({"classID": int(row.classID),
                           "minQuantile": None, "maxQuantile": None,
                           "minBreakValue": low, "maxBreakValue": high})

    return class_raster, pd.DataFrame(break_rows)


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

def crop_buffer_from_output(buffered_path, output_path, original_extent, buffer_pixels, buffered_extent=None):
    """
    Remove buffer from output raster.

    Args:
        buffered_path: Path to buffered raster
        output_path: Path for cropped output
        original_extent: Dict with row_start, row_end, col_start, col_end (unbuffered tile extent)
        buffer_pixels: Buffer size used
        buffered_extent: Optional dict with actual buffered extent (after clipping to raster bounds)
    """
    with rasterio.open(buffered_path) as src:
        # Calculate actual offset (handles cases where buffer is missing at raster edges)
        if buffered_extent:
            # Calculate actual buffer offset on each side
            row_offset = original_extent["row_start"] - buffered_extent["row_start"]
            col_offset = original_extent["col_start"] - buffered_extent["col_start"]
        else:
            # Fallback: assume full buffer (backward compatibility)
            row_offset = buffer_pixels
            col_offset = buffer_pixels

        ps.environment.update_run_log(
            f"[crop_buffer] Buffered raster: {src.height}x{src.width}, "
            f"removing buffer (offsets: row={row_offset}, col={col_offset})"
        )

        # Create window to extract original extent
        window = Window(
            col_offset,
            row_offset,
            original_extent["col_end"] - original_extent["col_start"],
            original_extent["row_end"] - original_extent["row_start"]
        )

        ps.environment.update_run_log(
            f"[crop_buffer] Original extent: rows [{original_extent['row_start']}:{original_extent['row_end']}], "
            f"cols [{original_extent['col_start']}:{original_extent['col_end']}]"
        )
        if buffered_extent:
            ps.environment.update_run_log(
                f"[crop_buffer] Buffered extent: rows [{buffered_extent['row_start']}:{buffered_extent['row_end']}], "
                f"cols [{buffered_extent['col_start']}:{buffered_extent['col_end']}]"
            )
        ps.environment.update_run_log(
            f"[crop_buffer] Crop window: col_off={col_offset}, row_off={row_offset}, "
            f"width={window.width}, height={window.height}"
        )
        ps.environment.update_run_log(
            f"[crop_buffer] Extracting pixels: rows [{row_offset}:{row_offset+window.height}], "
            f"cols [{col_offset}:{col_offset+window.width}] from buffered raster"
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

    ps.environment.update_run_log(f"[extend_tile_to_full_extent] Original dtype: {dtype}, numpy dtype: {tile_data.dtype}")

    # Convert Byte/UInt8 to Float32 for SyncroSim merger compatibility
    # Check multiple ways since dtype comparison can be tricky
    dtype_str = str(dtype).lower()
    if 'uint8' in dtype_str or 'byte' in dtype_str or tile_data.dtype == np.uint8:
        ps.environment.update_run_log(f"[extend_tile_to_full_extent] Converting {dtype} to float32 for SyncroSim compatibility")
        tile_data = tile_data.astype(np.float32)
        dtype = rasterio.float32
        if nodata is not None and nodata != -9999:
            nodata = -9999.0  # Use standard nodata for float32
            ps.environment.update_run_log(f"[extend_tile_to_full_extent] Changed nodata to -9999.0")

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

    # Diagnostic logging with explicit pixel ranges
    ps.environment.update_run_log(
        f"[extend_tile_to_full_extent] Tile size: {tile_height}x{tile_width}, "
        f"Full extent: {full_height}x{full_width}"
    )
    ps.environment.update_run_log(
        f"[extend_tile_to_full_extent] Placing tile data:"
    )
    ps.environment.update_run_log(
        f"  Rows: [{row_offset}:{row_end}] (pixels {row_offset} to {row_end-1})"
    )
    ps.environment.update_run_log(
        f"  Cols: [{col_offset}:{col_end}] (pixels {col_offset} to {col_end-1})"
    )

    # Check for out-of-bounds placement
    if row_offset < 0 or col_offset < 0 or row_offset >= full_height or col_offset >= full_width:
        ps.environment.update_run_log(
            f"[extend_tile_to_full_extent] WARNING: Tile offset is out of bounds! "
            f"offset=({row_offset}, {col_offset}), full_extent=({full_height}, {full_width})"
        )

    full_array[row_offset:row_end, col_offset:col_end] = \
        tile_data[0:(row_end - row_offset), 0:(col_end - col_offset)]

    # Log statistics about the placed data
    placed_data = tile_data[0:(row_end - row_offset), 0:(col_end - col_offset)]
    if nodata is not None:
        valid_count = np.sum(placed_data != nodata)
    else:
        valid_count = np.sum(~np.isnan(placed_data))
    ps.environment.update_run_log(
        f"[extend_tile_to_full_extent] Placed {valid_count} valid pixels out of {placed_data.size} total"
    )

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


# ============================================================================
# RESISTANCE RECLASSIFICATION
# ============================================================================

def apply_reclass_table(resistance_path, reclass_table, output_path):
    """Map land cover class IDs to resistance values.

    This stands in for Omniscape.jl's reclassify_resistance! (utils.jl) so the
    step can happen here, BEFORE the resistance modifiers, rather than inside
    Julia after them. Julia reclassifies whatever this package hands it, so a
    modifier used to multiply nominal class IDs - class 41 scaled by 1.5 became
    61.5, which matches no reclass row and survived into the resistance surface
    as a raw class ID. Doing the lookup first makes the modifiers act on real
    resistance values, which is the only scale on which multiplying means
    anything.

    Rules that match Julia:

      * exact-equality lookup against the ORIGINAL pixel values, so a value that
        has just been reclassified is never re-matched by a later row,
      * two rows for one class ID: the LAST row wins,
      * a resistanceValue of NODATA_VALUE (-9999) becomes no-data, matching the
        literal "missing" Omniscape reads from a reclass table file,
      * pixels that were already no-data stay no-data and never match a row.

    One rule that DIVERGES from Julia, deliberately: a class ID present in the
    raster but absent from the table is an error here, where Julia leaves it
    unchanged. Julia only touches the pixels a row names, which means an
    unlisted class survives as a raw class ID and is then read as though it
    were a resistance value - class 95 quietly becomes "resistance 95". A
    nominal class ID is not a resistance, so there is no reading of that which
    is correct, and it is invisible unless the stray value happens to be
    non-positive. Erroring also makes the largest value in the table the true
    maximum of the surface, which is what the modifier rescaling modes rescale
    against.

    reclass_table must carry numeric landCover / resistanceValue columns - not
    the frame whose resistanceValue has been rewritten to the string "missing"
    for Julia's benefit.
    """
    with rasterio.open(resistance_path) as src:
        original = src.read(1)
        meta = src.meta.copy()
        invalid = nodata_mask(src, original)

    # Class IDs arrive as Byte or Int16; resistance values are Double.
    original = original.astype(np.float64)

    # Collapsing the table into a dict gives last-row-wins for free, including
    # across a mix of -9999 and real values, which a per-row mask loop gets
    # wrong unless it also tracks and clears an earlier row's "missing".
    mapping = {}
    for _, row in reclass_table.iterrows():
        mapping[float(row["landCover"])] = float(row["resistanceValue"])

    codes = np.fromiter(mapping.keys(), dtype=np.float64, count=len(mapping))
    values = np.fromiter(mapping.values(), dtype=np.float64, count=len(mapping))
    order = np.argsort(codes)
    codes, values = codes[order], values[order]

    # One O(n log k) pass, rather than a full-array pass per table row.
    idx = np.clip(np.searchsorted(codes, original), 0, len(codes) - 1)
    matched = (codes[idx] == original) & ~invalid

    # Every class in the raster must have a row. The test is a free pass over
    # an array we already have; the np.unique needed to name the offenders only
    # runs when there is an error to explain.
    stranded = ~matched & ~invalid
    if stranded.any():
        missing = sorted(np.unique(original[stranded]).tolist())
        missing = [int(code) if float(code).is_integer() else code
                   for code in missing]
        sys.exit(
            "'Reclass Table' has no entry for land cover class ID(s) "
            f"{missing}, which are present in the resistance raster. Every "
            "class in the raster must have a row. Add a row mapping it to a "
            "resistance value, or to -9999 if it should be NoData.")

    reclassified = np.where(matched, values[idx], original)
    invalid |= matched & (values[idx] == NODATA_VALUE)
    reclassified[invalid] = NODATA_VALUE

    # float64 rather than float32: resistanceValue is a Double, and a float32
    # round trip moves a value like 0.1 by ~1e-8, which is enough to flip an
    # exact comparison against r_cutoff or source_threshold in Omniscape.
    # -9999 rather than NaN for no-data: apply_resistance_modifier restores
    # no-data with an == comparison, which NaN never satisfies, and
    # merge_tile_outputs hardcodes -9999 as its source nodata.
    meta.update(driver="GTiff", count=1, dtype="float64",
                nodata=NODATA_VALUE, compress="lzw")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(reclassified, 1)

    return output_path


# ============================================================================
# RESISTANCE MODIFIER
# ============================================================================

def apply_resistance_modifier(resistance_path, modifier_path, modifier_table,
                               focal_radius=None, focal_function="mean", output_path=None,
                               default_multiplier=1.0):
    """
    Multiply resistance values by a per-pixel multiplier derived from a secondary raster.

    For each pixel:
      1. Read modifier raster value (optionally aggregated via focal window first)
      2. Look up the multiplier: the one range where minValue <= value < maxValue
      3. Multiply resistance by that multiplier (pixels matching no range use 1.0)

    Ranges are half-open and may not overlap - the caller validates that with
    validate_threshold_bands, so each value falls in at most one band and the
    order rows are applied in cannot matter. Gaps between bands are allowed and
    mean "leave this range alone": a pixel in no band keeps default_multiplier,
    normally 1.0.

    default_multiplier exists for Proportional rescaling, which divides the
    whole surface by a constant. That constant folds into the lookup
    multipliers, but pixels matching no band would otherwise keep a literal 1.0
    and escape the division, so the caller passes the constant here too.

    By the time this runs the resistance raster holds resistance values, not
    land cover class IDs - apply_reclass_table has already run if the scenario
    asked for reclassification. Multiplying is only meaningful on that scale.

    The modifier raster must share the CRS of the resistance raster. In tiled mode the
    full modifier raster path is passed and windowed reading clips it to the tile extent.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    with rasterio.open(resistance_path) as res_src:
        resistance_data = res_src.read(1).astype(np.float64)
        res_meta = res_src.meta.copy()
        res_nodata = res_src.nodata
        tile_bounds = res_src.bounds

    with rasterio.open(modifier_path) as mod_src:
        window = mod_src.window(*tile_bounds)
        modifier_data = mod_src.read(1, window=window).astype(np.float64)
        mod_nodata = mod_src.nodata

    mod_mask = (modifier_data == mod_nodata) if mod_nodata is not None else np.zeros_like(modifier_data, dtype=bool)
    focal_input = np.where(mod_mask, np.nan, modifier_data)

    if focal_radius is not None and focal_radius > 0:
        size = 2 * focal_radius + 1
        padded = np.pad(focal_input, focal_radius, mode='constant', constant_values=np.nan)
        windows = sliding_window_view(padded, (size, size))
        fn = focal_function.lower()
        if fn == "mean":
            modifier_data = np.nanmean(windows, axis=(-2, -1))
        elif fn == "sum":
            modifier_data = np.nansum(windows, axis=(-2, -1))
        elif fn == "max":
            modifier_data = np.nanmax(windows, axis=(-2, -1))
        elif fn == "min":
            modifier_data = np.nanmin(windows, axis=(-2, -1))

    multiplier_array = np.full_like(resistance_data, default_multiplier,
                                    dtype=np.float64)
    for _, row in modifier_table.iterrows():
        mask = (~mod_mask) & (modifier_data >= row['minValue']) & (modifier_data < row['maxValue'])
        multiplier_array[mask] = float(row['multiplier'])

    modified = resistance_data * multiplier_array
    if res_nodata is not None:
        modified[resistance_data == res_nodata] = res_nodata

    if output_path is None:
        output_path = resistance_path.replace('.tif', '_modified.tif')

    # float64 for the same reason apply_reclass_table writes it: a float32
    # round trip moves a value enough to flip an exact comparison against
    # r_cutoff in Omniscape, and rescaling multiplies by a non-round constant.
    res_meta.update(dtype='float64', compress='lzw')
    with rasterio.open(output_path, 'w', **res_meta) as dst:
        dst.write(modified.astype(np.float64), 1)

    return output_path


# ============================================================================
# RESISTANCE RESCALING
# ============================================================================

RESCALING_NAMES = {0: "Exact", 1: "Proportional", 2: "Cap", 3: "Min-max"}


def max_effective_multiplier(modifier_table):
    """Largest multiplier a pixel can pick up from one modifier's lookup table.

    Floored at 1.0 because a pixel matching no band keeps 1.0, so the largest
    multiplier actually in play is never below it. That floor is also why
    Proportional rescaling never scales a surface UP: the factor is 1/M with
    M >= 1, so a set of modifiers that only reduce resistance leaves the surface
    where it is rather than inflating it to meet the reference maximum.

    Derived from the table alone, never from pixels. Tiles are processed in
    separate OS processes under multiprocessing, so a factor measured from one
    tile's own values would differ between tiles and leave seams where they meet.
    """
    if modifier_table.empty:
        return 1.0
    return max(1.0, float(modifier_table["multiplier"].max()))


def min_effective_multiplier(modifier_table):
    """Smallest multiplier a pixel can pick up from one modifier's lookup table.

    Capped at 1.0, the mirror of max_effective_multiplier: a pixel matching no
    band keeps 1.0, so the smallest multiplier in play is never above it. Used
    by Min-max rescaling to know where the modified range starts.
    """
    if modifier_table.empty:
        return 1.0
    return min(1.0, float(modifier_table["multiplier"].min()))


def apply_rescaling(resistance_path, mode, output_path,
                    reference_max, reference_min=None,
                    total_max_multiplier=1.0, total_min_multiplier=1.0,
                    is_conductance=False):
    """Bring a modified resistance surface back to its reference maximum.

    Modifiers multiply, so a reclass table topping out at 32 and a x2 modifier
    give 64 - outside the range the table was calibrated on. The modes:

      Cap      out = min(x, reference_max). Pixels no modifier touched come out
               untouched; pixels pushed past the ceiling flatten onto it, which
               is a true statement when the ceiling already means impermeable.
      Min-max  affine map of the modified range onto [reference_min,
               reference_max]. Both ends land exactly. Note that resistance is a
               ratio scale - "twice as resistant" is meaningful - and an affine
               map with an offset does not preserve that.

    Exact and Proportional never reach here: Exact does nothing, and
    Proportional is a global constant the caller folds into the lookup
    multipliers instead, so it costs no pass over the raster at all.

    Every bound is a global constant taken from the datasheets - reference_max
    and reference_min from the Reclass Table, the multiplier totals from the
    lookup tables - so all of them are identical in every tile and every
    parallel job. Nothing here may be derived from pixel statistics: tiles run
    in separate OS processes under multiprocessing, so a bound measured from one
    tile's values would differ between tiles and leave seams where they meet.

    A conductance surface is inverted to resistance, rescaled, and inverted
    back. Working on conductance directly would be wrong for Min-max, because an
    affine map in resistance space is not an affine map in conductance space.
    """
    with rasterio.open(resistance_path) as src:
        data = src.read(1).astype(np.float64)
        meta = src.meta.copy()
        invalid = nodata_mask(src, data)

    valid = ~invalid

    if valid.any():
        # Rescale in resistance terms whatever the surface happens to hold. The
        # placeholder 1.0 keeps no-data out of the arithmetic, including the
        # reciprocal; those pixels are restored below.
        work = np.where(valid, data, 1.0)
        if is_conductance:
            work = 1.0 / work

        if mode == "Cap":
            work = np.minimum(work, reference_max)
        elif mode == "Min-max":
            lo_in = reference_min * total_min_multiplier
            hi_in = reference_max * total_max_multiplier
            if hi_in != lo_in:
                work = (reference_min
                        + (work - lo_in) * (reference_max - reference_min)
                        / (hi_in - lo_in))
            # else: no range to stretch, so leave it rather than divide by
            # zero - the same guard standardize_min_max makes.
        else:
            raise ValueError(f"apply_rescaling does not handle mode {mode!r}")

        if is_conductance:
            work = 1.0 / work

        rescaled = np.where(valid, work, data)
    else:
        rescaled = data

    rescaled[invalid] = NODATA_VALUE

    meta.update(driver="GTiff", count=1, dtype="float64",
                nodata=NODATA_VALUE, compress="lzw")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rescaled, 1)

    return output_path
