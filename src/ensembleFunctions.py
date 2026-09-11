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


def resolve_member_reference(value, member_names, source_label):
    """Read a member Datasheet reference tolerating either its ID or its Name.

    A Datasheet-validated column stores the referenced row's key, but which of
    the key and the display name comes back depends on how the datasheet was
    read - the same tolerance resolve_list_option applies to List columns.
    Coercing blindly would raise a bare ValueError traceback instead of saying
    what was wrong.

    Returns the member ID, or None when nothing is set.
    """
    if value is None or pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    name = str(value).strip()
    for member_id, member_name in member_names.items():
        if member_name == name:
            return member_id

    sys.exit(
        source_label + " refers to ensemble member " + repr(value)
        + ", which is not defined in the Project's 'Ensemble Members' datasheet"
        + ((" (found: " + ", ".join(repr(n) for n in member_names.values()) + ")")
           if member_names else " (that datasheet is empty)") + ".")


def read_scenario_member(scenario, scenario_label, member_names):
    """Read which ensemble member a Scenario represents, or None if it declares none.

    Membership lives in the single-row 'Ensemble Membership' datasheet on the
    Scenario itself, so it is tied to that Scenario by where it is stored rather
    than by a key pointing back at it. The weight then hangs off the member
    rather than off the Scenario, which is what lets one Scenario count for
    different amounts in different ensembles.
    """
    try:
        membership = scenario.datasheets(name = "omniscape_ensembleMembership")
    except Exception as error:
        # A Scenario predating the datasheet still combines, unweighted. Say so
        # rather than swallowing it: a library or connection error reaching the
        # datasheet looks identical from here, and defaulting to no member
        # without a word would turn it into a plausible-looking wrong answer.
        safe_update_run_log(
            "Could not read the ensemble membership of Scenario '" + scenario_label
            + "' (" + repr(error) + "); contributing at weight "
            + repr(DEFAULT_WEIGHT) + ".")
        return None

    if membership.empty or "member" not in membership.columns:
        return None

    return resolve_member_reference(
        membership.member.iloc[0], member_names,
        "The 'Ensemble Membership' of Scenario '" + scenario_label + "'")


def resolve_ensemble_weights(dependency_table, dependency_members, weights_table,
                             member_names):
    """Weight each dependency by the ensemble member it represents.

    dependency_members holds one member ID, or None, per row of
    dependency_table, in that same order - collected as the layers were stacked,
    so the two cannot fall out of step.

    Weights come from the ensemble Scenario's 'Ensemble Weights' datasheet, keyed
    by member rather than by Scenario. A member with no weight row, and a
    dependency declaring no member at all, both contribute at 1.0, so an ensemble
    left alone is an unweighted one.

    Returns (weights_list, message).
    """
    def label(member_id):
        return repr(member_names.get(member_id, "member " + repr(member_id)))

    weight_by_member = {}

    if weights_table is not None and len(weights_table) != 0:
        for row in weights_table.itertuples():
            member_id = resolve_member_reference(
                row.member, member_names, "The 'Ensemble Weights' datasheet")

            if member_id is None:
                sys.exit(
                    "Every row of the 'Ensemble Weights' datasheet needs an "
                    "'Ensemble member'. Remove the blank row, or choose the "
                    "member its weight applies to.")

            if member_id in weight_by_member:
                sys.exit(
                    "The 'Ensemble Weights' datasheet has more than one weight "
                    "for member " + label(member_id) + ".")

            weight_by_member[member_id] = float(row.weight)

    claimed = [m for m in dependency_members if m is not None]

    # Two dependencies representing the same member make that member's weight
    # ambiguous - it is not clear which raster it was meant to scale
    duplicated = sorted({m for m in claimed if claimed.count(m) > 1})
    if duplicated:
        sys.exit(
            "More than one dependency Scenario represents ensemble member "
            + ", ".join(label(m) for m in duplicated) + ". Each member can be "
            "represented by only one Scenario in an ensemble.")

    # A weight matching no dependency is almost certainly the wrong member
    # chosen, and ignoring it would let it do nothing without anyone noticing
    unused = sorted(set(weight_by_member) - set(claimed))
    if unused:
        sys.exit(
            "The 'Ensemble Weights' datasheet has weights for members that no "
            "dependency Scenario represents: " + ", ".join(label(m) for m in unused)
            + ". Set the matching Scenario's 'Ensemble Membership', or remove "
            "the weight.")

    weights_list = [DEFAULT_WEIGHT if m is None else weight_by_member.get(m, DEFAULT_WEIGHT)
                    for m in dependency_members]

    message = ("Ensemble weights: " + ", ".join(
        "'" + str(n) + "' (" + ("no member" if m is None else label(m)) + ") = " + repr(w)
        for n, m, w in zip(dependency_table.Name, dependency_members, weights_list)) + ".")

    return weights_list, message
