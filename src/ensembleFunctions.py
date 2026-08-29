## omniscape

# Helper functions for the Ensemble Connectivity transformer.
#
# Kept separate from helperFunctions.py (which serves the tiling and
# resistance-modifier machinery) so the ensemble feature is self-contained.

import pysyncrosim as ps
import pandas as pd
import numpy as np
import sys

# Value used throughout omniscape to flag "no data"
NODATA_VALUE = -9999

# How closely two rasters must be aligned before they can be combined, as a
# fraction of one pixel. Loose enough to tolerate floating-point differences
# between a raster merged from spatial tiles and one written in a single pass,
# tight enough that a genuine offset of even one pixel is rejected.
GRID_TOLERANCE_FRACTION = 0.01


def safe_update_run_log(*message, sep = "", type = "status"):
    """Write to the SyncroSim run log, falling back to the console.

    ps.environment.update_run_log raises RuntimeError when the SSIM_*
    environment variables are absent (any run outside SyncroSim), so wrapping
    it lets the transformer be run and debugged standalone.
    """
    try:
        ps.environment.update_run_log(*message, sep = sep, type = type)
    except RuntimeError:
        print("[Run log] " + sep.join(str(m) for m in message))


def nodata_mask(raster_source, raster_data):
    """Return a boolean array that is True wherever a pixel holds no valid data.

    A raster can flag "no data" in more than one way depending on which
    omniscape code path produced it: a sentinel declared in the file header
    (normally -9999), NaN with no declared sentinel (what spatial tiling
    produces when a tile carries no declared nodata), or -9999 present in the
    pixels but undeclared. All three are tested. Testing for -9999
    unconditionally is safe here: normalized current is a ratio, so -9999 is
    never a legitimate value.
    """
    mask = np.zeros(raster_data.shape, dtype = bool)

    if raster_source.nodata is not None:
        if np.isnan(raster_source.nodata):
            mask |= np.isnan(raster_data)
        else:
            mask |= (raster_data == raster_source.nodata)

    if np.issubdtype(raster_data.dtype, np.floating):
        mask |= np.isnan(raster_data)

    mask |= (raster_data == NODATA_VALUE)

    return mask


def validate_same_grid(base_source, altr_source, raster_label):
    """Exit unless two rasters describe the same pixel grid.

    Combining rasters pixel-by-pixel is only meaningful if they cover the same
    ground, at the same resolution, in the same coordinate system. The affine
    transform is compared with a sub-pixel tolerance so that floating-point
    noise from tile merging does not fail the check, while a genuine offset of
    even one pixel does.
    """
    if base_source.shape != altr_source.shape:
        sys.exit(
            "The '" + raster_label + "' rasters being combined have different "
            "dimensions (" + repr(base_source.shape) + " and "
            + repr(altr_source.shape) + "). All Scenarios must be run over the "
            "same extent and resolution before they can be combined.")

    if base_source.crs != altr_source.crs:
        sys.exit(
            "The '" + raster_label + "' rasters being combined use different "
            "coordinate reference systems (" + repr(base_source.crs) + " and "
            + repr(altr_source.crs) + "). All Scenarios must use the same "
            "projection before they can be combined.")

    tolerance = GRID_TOLERANCE_FRACTION * min(
        abs(base_source.res[0]), abs(base_source.res[1]))
    base_transform = list(base_source.transform)[:6]
    altr_transform = list(altr_source.transform)[:6]

    if any(abs(b - a) > tolerance for b, a in zip(base_transform, altr_transform)):
        sys.exit(
            "The '" + raster_label + "' rasters being combined are not aligned "
            "to the same grid. Their pixel origins or resolutions differ by "
            "more than " + repr(tolerance) + " map units (" + repr(base_transform)
            + " and " + repr(altr_transform) + "). All Scenarios must be run "
            "over the same extent and resolution before they can be combined.")


def standardize_min_max(raster_data, mask):
    """Rescale a raster to 0-1 using its own min/max over all valid pixels.

    Different Scenarios' normalized current maps can sit on different value
    ranges, so each input is standardized onto a common 0-1 scale before they
    are combined. The range is taken over the full valid extent. A constant
    raster (max == min) standardizes to all zeros rather than dividing by zero.
    """
    data = raster_data.astype(float)
    valid = data[~mask]

    if valid.size == 0:
        return data

    lo, hi = valid.min(), valid.max()

    if hi == lo:
        data = np.where(mask, data, 0.0)
    else:
        data = (data - lo) / (hi - lo)

    return data


def focal_statistic(raster_data, mask, radius, statistic):
    """Apply a square moving-window statistic, ignoring no-data pixels.

    Mirrors the focal-window implementation used by the resistance modifiers
    (sliding_window_view over a NaN-padded array with NaN-aware statistics).
    The window is (2 * radius + 1) pixels on a side. No-data pixels do not
    contribute to any window, and the output keeps the input's no-data
    footprint: smoothing never invents values where there were none.

    statistic is one of "Mean", "Sum", "Max", "Min".
    """
    from numpy.lib.stride_tricks import sliding_window_view

    functions = {"Mean": np.nanmean, "Sum": np.nansum,
                 "Max": np.nanmax, "Min": np.nanmin}

    if statistic not in functions:
        sys.exit("Unknown focal function: " + repr(statistic)
                 + ". Expected one of " + ", ".join(functions) + ".")

    data = raster_data.astype(float)
    data[mask] = np.nan

    padded = np.pad(data, radius, mode = "constant", constant_values = np.nan)
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))

    import warnings
    with np.errstate(invalid = "ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            result = functions[statistic](windows, axis = (-2, -1))

    result[mask] = np.nan
    return result


def combine_layers(layer_stack, mask_stack, weights, method):
    """Combine standardized layers, one per Scenario, into a single ensemble.

    layer_stack and mask_stack are (nScenarios, rows, cols) arrays; weights is
    a sequence of positive floats, one per layer, in the same order. A pixel is
    valid in the ensemble if at least one layer has data there; the statistics
    are taken over whichever layers have data.

    method is one of "Weighted Mean", "Weighted Sum", "Maximum", "Minimum".
    Maximum and Minimum deliberately ignore the weights: they answer "the
    best/worst connectivity for any Scenario", where scaling by a weight would
    change which Scenario wins.

    Returns (ensemble, ensemble_mask).
    """
    data = layer_stack.astype(float).copy()
    data[mask_stack] = np.nan
    w = np.asarray(weights, dtype = float).reshape(-1, 1, 1)

    any_valid = ~np.all(mask_stack, axis = 0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)

        if method == "Weighted Mean":
            weight_totals = np.where(~mask_stack, w, 0.0).sum(axis = 0)
            weighted_sum = np.nansum(data * w, axis = 0)
            with np.errstate(invalid = "ignore", divide = "ignore"):
                ensemble = np.where(weight_totals > 0, weighted_sum / weight_totals, np.nan)
        elif method == "Weighted Sum":
            ensemble = np.nansum(data * w, axis = 0)
        elif method == "Maximum":
            ensemble = np.nanmax(data, axis = 0)
        elif method == "Minimum":
            ensemble = np.nanmin(data, axis = 0)
        else:
            sys.exit("Unknown combination method: " + repr(method) + ".")

    ensemble = np.where(any_valid, ensemble, np.nan)
    return ensemble, ~any_valid


def classify_by_quantiles(ensemble_data, mask, quantile_table):
    """Classify an ensemble raster into categories using quantile thresholds.

    quantile_table has one row per category with columns classID, minQuantile
    and maxQuantile (each between 0 and 1). The break VALUES are computed from
    the distribution of valid ensemble pixels, so the categories adapt to
    whatever range the ensemble happens to occupy - the quantile analogue of
    the fixed-value 'Category Thresholds'.

    Intervals are half-open [min, max), except that a maxQuantile of 1 is
    closed so the single largest pixel is not left unclassified. Pixels falling
    in no interval are no-data in the output.

    Returns (class_raster, breaks_table) where breaks_table adds the computed
    minBreakValue / maxBreakValue per category.
    """
    valid = ensemble_data[~mask]

    if valid.size == 0:
        sys.exit("The ensemble raster contains no valid pixels to classify.")

    class_raster = np.full(ensemble_data.shape, NODATA_VALUE, dtype = np.int16)
    break_rows = []

    for row in quantile_table.itertuples():
        if not (0.0 <= row.minQuantile < row.maxQuantile <= 1.0):
            sys.exit(
                "Invalid quantile range for category ID " + repr(int(row.classID))
                + ": minimum " + repr(row.minQuantile) + ", maximum "
                + repr(row.maxQuantile) + ". Quantiles must satisfy "
                "0 <= minimum < maximum <= 1.")

        lo = float(np.quantile(valid, row.minQuantile))
        hi = float(np.quantile(valid, row.maxQuantile))

        if row.maxQuantile >= 1.0:
            selected = (ensemble_data >= lo) & (ensemble_data <= hi) & ~mask
        else:
            selected = (ensemble_data >= lo) & (ensemble_data < hi) & ~mask

        class_raster[selected] = int(row.classID)
        break_rows.append({"classID": int(row.classID),
                           "minQuantile": float(row.minQuantile),
                           "maxQuantile": float(row.maxQuantile),
                           "minBreakValue": lo, "maxBreakValue": hi})

    return class_raster, pd.DataFrame(break_rows)


def resolve_ensemble_weights(dependency_table, weights_table):
    """Map each dependency Scenario to its ensemble weight.

    Weights come from the 'Ensemble Weights' datasheet, keyed by Scenario ID.
    Any dependency without a row gets weight 1.0, so an empty table is an
    unweighted ensemble. A weight row whose Scenario ID is not a dependency is
    an error - it is probably a typo, and silently ignoring it would let a
    mis-keyed weight do nothing without anyone noticing.

    Returns (weights_list, message) with weights ordered as dependency_table.
    """
    dependency_ids = [int(i) for i in dependency_table.Id]
    weight_by_id = {}

    if weights_table is not None and len(weights_table) != 0:
        for row in weights_table.itertuples():
            scenario_id = int(row.scenarioId)
            if scenario_id not in dependency_ids:
                sys.exit(
                    "The 'Ensemble Weights' datasheet contains a weight for "
                    "Scenario ID " + repr(scenario_id) + ", which is not a "
                    "dependency of this Scenario. Weights can only be assigned "
                    "to dependency Scenarios (found: "
                    + ", ".join(repr(i) for i in dependency_ids) + ").")
            if scenario_id in weight_by_id:
                sys.exit(
                    "The 'Ensemble Weights' datasheet contains more than one "
                    "weight for Scenario ID " + repr(scenario_id) + ".")
            weight_by_id[scenario_id] = float(row.weight)

    weights_list = [weight_by_id.get(i, 1.0) for i in dependency_ids]

    names = list(dependency_table.Name)
    message = ("Ensemble weights: " + ", ".join(
        "'" + str(n) + "' (ID " + repr(i) + ") = " + repr(w)
        for n, i, w in zip(names, dependency_ids, weights_list)) + ".")

    return weights_list, message
