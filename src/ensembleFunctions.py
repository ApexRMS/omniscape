## omniscape

# Helper functions specific to the Ensemble Connectivity transformer: reading
# the dependency Scenarios, putting them on a common scale, and combining them.
#
# Anything the ensemble shares with the rest of the package - the no-data
# convention, the run-log wrapper, classification - lives in helperFunctions.py
# and is re-exported here so importers do not need to know which module a
# given helper came from.

import numpy as np
import pandas as pd
import sys

from helperFunctions import NODATA_VALUE, nodata_mask, safe_update_run_log

# Re-exported for the transformer's benefit; referenced here so linters do not
# read them as unused imports
_REEXPORTED = (NODATA_VALUE, nodata_mask, safe_update_run_log)

# How closely two rasters must be aligned before they can be combined, as a
# fraction of one pixel. Loose enough to tolerate floating-point differences
# between a raster merged from spatial tiles and one written in a single pass,
# tight enough that a genuine offset of even one pixel is rejected.
GRID_TOLERANCE_FRACTION = 0.01

# What a Scenario contributes at when it has not been given a weight of its own
DEFAULT_WEIGHT = 1.0


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

    Both weighted methods normalize by the weight of the layers actually
    present at a pixel, not by the weight of every layer. No-data means "not
    known here", not "zero connectivity", so a pixel covered by 3 of 5
    Scenarios must not be pushed down relative to one covered by all 5 - the
    two would otherwise be classified into different categories on the
    strength of coverage alone. Weighted Sum therefore scales its partial sum
    up to the full weight total, making it exactly Weighted Mean times the sum
    of all weights and keeping the two methods consistent with each other.

    Returns (ensemble, ensemble_mask).
    """
    data = layer_stack.astype(float).copy()
    data[mask_stack] = np.nan
    w = np.asarray(weights, dtype = float).reshape(-1, 1, 1)

    any_valid = ~np.all(mask_stack, axis = 0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)

        if method in ("Weighted Mean", "Weighted Sum"):
            # Weight of the layers holding data at each pixel, so that absent
            # layers neither contribute nor dilute
            presentWeight = np.where(~mask_stack, w, 0.0).sum(axis = 0)
            weightedSum = np.nansum(data * w, axis = 0)

            scale = 1.0 if method == "Weighted Mean" else float(w.sum())

            with np.errstate(invalid = "ignore", divide = "ignore"):
                ensemble = np.where(presentWeight > 0,
                                    weightedSum * scale / presentWeight, np.nan)
        elif method == "Maximum":
            ensemble = np.nanmax(data, axis = 0)
        elif method == "Minimum":
            ensemble = np.nanmin(data, axis = 0)
        else:
            sys.exit("Unknown combination method: " + repr(method) + ".")

    ensemble = np.where(any_valid, ensemble, np.nan)
    return ensemble, ~any_valid


def read_scenario_weight(scenario, scenario_label):
    """Read one Scenario's own weight for the ensemble it is being combined into.

    The weight lives in the single-row 'Ensemble Contribution' datasheet on the
    Scenario being combined, so it is tied to that Scenario by where it is
    stored rather than by a key pointing back at it. Nothing can be mis-keyed
    onto the wrong Scenario, orphaned by a rename, or left silently doing
    nothing - and there is no separate list of weights that has to be kept in
    step with the order the dependencies are read in.

    A Scenario that never set a weight contributes at 1.0, so an ensemble whose
    dependencies were all left alone is an unweighted one.
    """
    try:
        contribution = scenario.datasheets(name = "omniscape_ensembleContribution")
    except Exception as error:
        # A Scenario predating the datasheet still combines, unweighted. Say so
        # rather than swallowing it: a library or connection error reaching the
        # datasheet looks identical from here, and defaulting to 1.0 without a
        # word would turn it into a plausible-looking wrong answer.
        safe_update_run_log(
            "Could not read the ensemble weight for Scenario '" + scenario_label
            + "' (" + repr(error) + "); contributing at " + repr(DEFAULT_WEIGHT) + ".")
        return DEFAULT_WEIGHT

    if contribution.empty or "weight" not in contribution.columns:
        return DEFAULT_WEIGHT

    value = contribution.weight.iloc[0]
    if pd.isna(value):
        return DEFAULT_WEIGHT

    weight = float(value)

    if weight <= 0:
        sys.exit(
            "Scenario '" + scenario_label + "' has an ensemble weight of "
            + repr(weight) + ". Weights must be greater than 0. To leave a "
            "Scenario out of an ensemble, remove it from the ensemble "
            "Scenario's dependencies.")

    return weight


def describe_ensemble_weights(dependency_table, weights):
    """Summarize, for the run log, what each dependency contributed at."""
    return ("Ensemble weights: " + ", ".join(
        "'" + str(n) + "' (ID " + repr(int(i)) + ") = " + repr(w)
        for n, i, w in zip(dependency_table.Name, dependency_table.Id, weights)) + ".")
