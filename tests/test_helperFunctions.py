"""Unit tests for the raster helpers in src/helperFunctions.py.

Deliberately free of pysyncrosim: these exercise pure numpy/rasterio behaviour,
so they run anywhere without a SyncroSim install. helperFunctions imports
pysyncrosim at module scope, so a stub is installed before the import.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

# Stub pysyncrosim so helperFunctions imports without SyncroSim present.
if "pysyncrosim" not in sys.modules:
    stub = types.ModuleType("pysyncrosim")
    stub.environment = types.SimpleNamespace(
        update_run_log=lambda *a, **k: None,
        progress_bar=lambda *a, **k: None,
    )
    sys.modules["pysyncrosim"] = stub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helperFunctions import (  # noqa: E402
    NODATA_VALUE,
    apply_reclass_table,
    apply_rescaling,
    apply_resistance_modifier,
    max_effective_multiplier,
    max_resistance_multiplier,
    min_effective_multiplier,
    validate_threshold_bands,
)


def write_raster(path, array, dtype, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs="EPSG:3857",
        transform=from_origin(0, array.shape[0], 1, 1), nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def reclass(pairs):
    return pd.DataFrame(pairs, columns=["landCover", "resistanceValue"])


def run_reclass(tmpdir, array, pairs, dtype="uint8", nodata=None):
    src = write_raster(os.path.join(tmpdir, "in.tif"), array, dtype, nodata)
    out = os.path.join(tmpdir, "out.tif")
    apply_reclass_table(src, reclass(pairs), out)
    with rasterio.open(out) as ds:
        return ds.read(1), ds


# --------------------------------------------------------------------------
# apply_reclass_table
# --------------------------------------------------------------------------

def test_unlisted_class_is_an_error(tmp_path):
    """Julia leaves an unlisted class alone, so class 90 would survive as
    "resistance 90". A class ID is not a resistance, so we stop instead."""
    with pytest.raises(SystemExit, match=r"no entry for land cover class"):
        run_reclass(str(tmp_path), np.array([[11, 41, 90]]),
                    [(11, 100), (41, 10)])


def test_error_names_every_missing_class_not_just_the_first(tmp_path):
    with pytest.raises(SystemExit, match=r"\[21, 90\]"):
        run_reclass(str(tmp_path), np.array([[11, 21, 90]]), [(11, 100)])


def test_declared_nodata_does_not_need_a_row(tmp_path):
    """255 is nodata, so it is not a class and needs no entry."""
    data, _ = run_reclass(str(tmp_path), np.array([[11, 255]]),
                          [(11, 100)], nodata=255)
    assert data.tolist() == [[100.0, float(NODATA_VALUE)]]


def test_reclassification_does_not_cascade(tmp_path):
    """2->6 must not then be caught by the 6->99 row: the lookup reads the
    ORIGINAL value, so a freshly written 6 is never re-matched."""
    data, _ = run_reclass(str(tmp_path), np.array([[2, 6]]),
                          [(2, 6), (6, 99)])
    assert data.tolist() == [[6.0, 99.0]]


def test_duplicate_class_last_row_wins(tmp_path):
    data, _ = run_reclass(str(tmp_path), np.array([[5]]), [(5, 10), (5, 20)])
    assert data.tolist() == [[20.0]]


def test_last_row_wins_over_an_earlier_missing(tmp_path):
    data, ds = run_reclass(str(tmp_path), np.array([[5]]),
                              [(5, NODATA_VALUE), (5, 20)])
    assert data.tolist() == [[20.0]]
    assert ds.nodata == NODATA_VALUE


def test_missing_wins_over_an_earlier_value(tmp_path):
    data, _ = run_reclass(str(tmp_path), np.array([[5]]),
                             [(5, 20), (5, NODATA_VALUE)])
    assert data.tolist() == [[float(NODATA_VALUE)]]


def test_resistance_value_of_negative_9999_becomes_nodata(tmp_path):
    """-9999 means "this class is NoData" - a mapping, not an absence, so the
    strict unlisted-class check must not fire on it."""
    data, ds = run_reclass(str(tmp_path), np.array([[11, 90]]),
                              [(11, 100), (90, NODATA_VALUE)])
    assert data.tolist() == [[100.0, float(NODATA_VALUE)]]
    assert ds.nodata == NODATA_VALUE


def test_declared_input_nodata_is_preserved_and_never_matched(tmp_path):
    """255 is nodata here, so its reclass row must not fire."""
    data, _ = run_reclass(str(tmp_path), np.array([[11, 255]]),
                             [(11, 100), (255, 7)], nodata=255)
    assert data.tolist() == [[100.0, float(NODATA_VALUE)]]


def test_byte_input_yields_float64_output(tmp_path):
    _, ds = run_reclass(str(tmp_path), np.array([[11]]), [(11, 0.125)])
    assert ds.dtypes[0] == "float64"


def test_fractional_resistance_survives_the_round_trip(tmp_path):
    """float32 would move 0.1; r_cutoff comparisons in Omniscape are exact."""
    data, _ = run_reclass(str(tmp_path), np.array([[11]]), [(11, 0.1)])
    assert data[0][0] == 0.1


# --------------------------------------------------------------------------
# reclass -> modifier, the ordering this change is about
# --------------------------------------------------------------------------

def test_reclass_then_modify_gives_the_hand_computed_surface(tmp_path):
    tmpdir = str(tmp_path)
    lulc = np.array([
        [11, 41, 21, 90],
        [41, 41, 90, 11],
        [21, 255, 11, 41],
        [90, 11, 41, 21],
    ])
    src = write_raster(os.path.join(tmpdir, "lulc.tif"), lulc, "uint8", nodata=255)

    reclassed = apply_reclass_table(
        src,
        reclass([(11, 100), (41, 10), (21, 21), (90, NODATA_VALUE)]),
        os.path.join(tmpdir, "reclass.tif"),
    )

    # Checkerboard: 1 doubles as "in the [0.5, 1.5) band".
    modifier = np.indices(lulc.shape).sum(axis=0) % 2
    mod_src = write_raster(os.path.join(tmpdir, "mod.tif"), modifier, "int16")

    table = pd.DataFrame(
        [("m", 0.0, 0.5, 1.0), ("m", 0.5, 1.5, 3.0)],
        columns=["modifier", "minValue", "maxValue", "multiplier"],
    )
    out = apply_resistance_modifier(reclassed, mod_src, table,
                                    output_path=os.path.join(tmpdir, "final.tif"))

    with rasterio.open(out) as ds:
        got = ds.read(1)

    base = np.array([
        [100.0, 10.0, 21.0, NODATA_VALUE],
        [10.0, 10.0, NODATA_VALUE, 100.0],
        [21.0, NODATA_VALUE, 100.0, 10.0],
        [NODATA_VALUE, 100.0, 10.0, 21.0],
    ])
    expected = np.where(base == NODATA_VALUE, NODATA_VALUE,
                        base * np.where(modifier == 1, 3.0, 1.0))
    assert np.allclose(got, expected)

    # The bug this replaces: modifying first feeds scaled class IDs to the
    # lookup, and 41*3 = 123 matches nothing. Strict reclassification now turns
    # that silent corruption into a stop.
    wrong = apply_resistance_modifier(src, mod_src, table,
                                      output_path=os.path.join(tmpdir, "wrong.tif"))
    with pytest.raises(SystemExit, match=r"no entry for land cover class"):
        apply_reclass_table(
            wrong,
            reclass([(11, 100), (41, 10), (21, 21), (90, NODATA_VALUE)]),
            os.path.join(tmpdir, "wrong-reclass.tif"),
        )


# --------------------------------------------------------------------------
# validate_threshold_bands(allow_gaps=...)
# --------------------------------------------------------------------------

def bands(rows):
    return pd.DataFrame(rows, columns=["minValue", "maxValue"])


def test_overlapping_bands_are_rejected_even_when_gaps_are_allowed():
    with pytest.raises(SystemExit, match="Overlapping ranges"):
        validate_threshold_bands(bands([(0, 10), (5, 20)]), "minValue",
                                 "maxValue", "Modifier Lookup Table (m)",
                                 allow_gaps=True)


def test_gaps_are_accepted_when_allowed():
    validate_threshold_bands(bands([(0, 10), (20, 30)]), "minValue",
                             "maxValue", "Modifier Lookup Table (m)",
                             allow_gaps=True)


def test_gaps_are_still_rejected_by_default():
    with pytest.raises(SystemExit, match="Gap in"):
        validate_threshold_bands(bands([(0, 10), (20, 30)]), "minValue",
                                 "maxValue", "Category Thresholds")


def test_inverted_band_is_rejected():
    with pytest.raises(SystemExit, match="is not less than maximum"):
        validate_threshold_bands(bands([(10, 5)]), "minValue", "maxValue",
                                 "Modifier Lookup Table (m)", allow_gaps=True)


# --------------------------------------------------------------------------
# rescaling
# --------------------------------------------------------------------------

def mod_table(rows):
    return pd.DataFrame(rows, columns=["modifier", "minValue", "maxValue",
                                       "multiplier"])


DOUBLING = mod_table([("m", 0.0, 0.5, 1.0), ("m", 0.5, 1.5, 2.0)])


def rescale_fixture(tmpdir):
    """Row 0 unmodified, row 1 doubled, over reclass values 1 / 8 / 32."""
    lulc = np.array([[11, 21, 31], [11, 21, 31]])
    src = write_raster(os.path.join(tmpdir, "lulc.tif"), lulc, "uint8")
    reclassed = apply_reclass_table(
        src, reclass([(11, 1.0), (21, 8.0), (31, 32.0)]),
        os.path.join(tmpdir, "r.tif"))
    modifier = np.array([[0, 0, 0], [1, 1, 1]])
    mod_src = write_raster(os.path.join(tmpdir, "m.tif"), modifier, "int16")
    return reclassed, mod_src


# A x2 *resistance* multiplier as the transformer stores it under
# 'Resistance is conductance': omniscapeTransformer inverts the lookup table
# (m -> 1/m) so that the datasheet keeps meaning "multiply resistance".
HALVING_CONDUCTANCE = mod_table([("m", 0.0, 0.5, 1.0), ("m", 0.5, 1.5, 0.5)])


def conductance_fixture(tmpdir):
    """The conductance twin of rescale_fixture.

    The reclass table yields conductance 1 / 0.125 / 0.03125, i.e. resistance
    1 / 8 / 32, and row 1 gets the resistance-doubling modifier. The modified
    surface is resistance [[1, 8, 32], [2, 16, 64]], so 64 overshoots the
    table's ceiling of 32 and there is something for rescaling to do.

    Bounds the transformer would derive from this table: reference_max 1.0 and
    reference_min 1/32, both in CONDUCTANCE space.
    """
    lulc = np.array([[11, 21, 31], [11, 21, 31]])
    src = write_raster(os.path.join(tmpdir, "clulc.tif"), lulc, "uint8")
    reclassed = apply_reclass_table(
        src, reclass([(11, 1.0), (21, 0.125), (31, 0.03125)]),
        os.path.join(tmpdir, "cr.tif"))
    modifier = np.array([[0, 0, 0], [1, 1, 1]])
    mod_src = write_raster(os.path.join(tmpdir, "cm.tif"), modifier, "int16")
    return reclassed, mod_src


def conductance_modified(tmpdir, table=None, default_multiplier=1.0):
    """conductance_fixture with a modifier applied. With the default table the
    result is resistance [[1, 8, 32], [2, 16, 64]]."""
    reclassed, mod_src = conductance_fixture(tmpdir)
    return apply_resistance_modifier(
        reclassed, mod_src,
        HALVING_CONDUCTANCE if table is None else table,
        output_path=os.path.join(tmpdir, "cmod.tif"),
        default_multiplier=default_multiplier)


def read_band(path):
    with rasterio.open(path) as ds:
        return ds.read(1)


def test_max_effective_multiplier_floors_at_one():
    """A pixel matching no band keeps 1.0, so the largest multiplier in play is
    never below it - which is why Proportional never scales a surface up."""
    assert max_effective_multiplier(DOUBLING) == 2.0
    assert max_effective_multiplier(
        mod_table([("m", 0.0, 1.0, 0.5), ("m", 1.0, 2.0, 0.25)])) == 1.0


def test_min_effective_multiplier_caps_at_one():
    assert min_effective_multiplier(DOUBLING) == 1.0
    assert min_effective_multiplier(
        mod_table([("m", 0.0, 1.0, 0.5)])) == 0.5


def test_factor_comes_from_the_table_not_the_pixels(tmp_path):
    """The seam regression. Under multiprocessing each tile is a separate
    process, so a factor measured from a tile's own values would differ between
    tiles and show up as seams. Deriving it from the table alone cannot."""
    same_table_different_order = mod_table(
        [("m", 0.5, 1.5, 2.0), ("m", 0.0, 0.5, 1.0)])
    assert (max_effective_multiplier(DOUBLING)
            == max_effective_multiplier(same_table_different_order))


def test_exact_is_the_unrescaled_product(tmp_path):
    reclassed, mod_src = rescale_fixture(str(tmp_path))
    out = apply_resistance_modifier(
        reclassed, mod_src, DOUBLING,
        output_path=os.path.join(str(tmp_path), "e.tif"))
    assert np.allclose(read_band(out), [[1, 8, 32], [2, 16, 64]])


def test_proportional_folds_into_the_multipliers(tmp_path):
    reclassed, mod_src = rescale_fixture(str(tmp_path))
    M = max_effective_multiplier(DOUBLING)
    scaled = DOUBLING.copy()
    scaled["multiplier"] = scaled["multiplier"] / M
    out = apply_resistance_modifier(
        reclassed, mod_src, scaled,
        output_path=os.path.join(str(tmp_path), "p.tif"),
        default_multiplier=1.0 / M)
    assert np.allclose(read_band(out), [[0.5, 4, 16], [1, 8, 32]])


def test_proportional_preserves_every_ratio(tmp_path):
    """It is a single global constant, so it is a change of units - which is
    why normalized current is unaffected by it."""
    tmpdir = str(tmp_path)
    reclassed, mod_src = rescale_fixture(tmpdir)
    M = max_effective_multiplier(DOUBLING)
    exact = read_band(apply_resistance_modifier(
        reclassed, mod_src, DOUBLING,
        output_path=os.path.join(tmpdir, "e.tif")))
    scaled = DOUBLING.copy()
    scaled["multiplier"] = scaled["multiplier"] / M
    prop = read_band(apply_resistance_modifier(
        reclassed, mod_src, scaled, output_path=os.path.join(tmpdir, "p.tif"),
        default_multiplier=1.0 / M))
    assert np.allclose(prop / prop[0, 0], exact / exact[0, 0])


def test_default_multiplier_scales_pixels_matching_no_band(tmp_path):
    """Without it, an unmatched pixel keeps a literal 1.0 and escapes the
    division, leaving the surface only partly rescaled."""
    tmpdir = str(tmp_path)
    reclassed, mod_src = rescale_fixture(tmpdir)
    # Value 0 matches no band, so those pixels fall back on the default. The
    # table is normalized by M=2 the way the transformer does it, and the
    # default carries the same 1/M.
    gapped = mod_table([("m", 0.5, 1.5, 2.0)])
    gapped["multiplier"] = gapped["multiplier"] / 2.0
    out = read_band(apply_resistance_modifier(
        reclassed, mod_src, gapped, output_path=os.path.join(tmpdir, "g.tif"),
        default_multiplier=0.5))
    assert np.allclose(out, [[0.5, 4, 16], [1, 8, 32]])


def test_cap_clamps_and_leaves_unmodified_pixels_alone(tmp_path):
    tmpdir = str(tmp_path)
    reclassed, mod_src = rescale_fixture(tmpdir)
    exact = apply_resistance_modifier(
        reclassed, mod_src, DOUBLING, output_path=os.path.join(tmpdir, "e.tif"))
    out = read_band(apply_rescaling(
        exact, "Cap", os.path.join(tmpdir, "c.tif"), reference_max=32.0))
    assert np.allclose(out, [[1, 8, 32], [2, 16, 32]])
    assert np.array_equal(out[0], read_band(exact)[0])


def test_min_max_lands_on_both_ends(tmp_path):
    tmpdir = str(tmp_path)
    reclassed, mod_src = rescale_fixture(tmpdir)
    exact = apply_resistance_modifier(
        reclassed, mod_src, DOUBLING, output_path=os.path.join(tmpdir, "e.tif"))
    out = read_band(apply_rescaling(
        exact, "Min-max", os.path.join(tmpdir, "mm.tif"),
        reference_max=32.0, reference_min=1.0,
        total_max_multiplier=2.0, total_min_multiplier=1.0))
    assert np.isclose(out.min(), 1.0)
    assert np.isclose(out.max(), 32.0)


def test_cap_on_conductance_becomes_a_floor(tmp_path):
    """Capping resistance at 32 is flooring conductance at 1/32.

    The regression. This used to pass reference_max=32.0 - a RESISTANCE value -
    which no caller ever supplies: under 'Resistance is conductance' the reclass
    table yields conductance, so the transformer derives reference_max 1.0 and
    reference_min 1/32 from it. Fed those, the old code clamped inverted pixels
    against an uninverted bound, running min(resistance, 1.0) and flattening the
    whole surface onto resistance 1 - the minimum - instead of capping at 32.
    """
    tmpdir = str(tmp_path)
    modified = conductance_modified(tmpdir)
    out = read_band(apply_rescaling(
        modified, "Cap", os.path.join(tmpdir, "cc.tif"),
        reference_max=1.0, reference_min=1.0 / 32, is_conductance=True))
    assert np.allclose(1.0 / out, [[1, 8, 32], [2, 16, 32]])


def test_min_max_on_conductance_lands_on_both_ends(tmp_path):
    """Both ends of the ORIGINAL conductance range, which is also what catches
    the multiplier endpoints failing to swap: the totals are combined from the
    inverted table, so the resistance-space lower endpoint is
    1 / total_max_multiplier and the upper is 1 / total_min_multiplier."""
    tmpdir = str(tmp_path)
    modified = conductance_modified(tmpdir)
    out = read_band(apply_rescaling(
        modified, "Min-max", os.path.join(tmpdir, "cmm.tif"),
        reference_max=1.0, reference_min=1.0 / 32,
        total_max_multiplier=max_effective_multiplier(HALVING_CONDUCTANCE),
        total_min_multiplier=min_effective_multiplier(HALVING_CONDUCTANCE),
        is_conductance=True))
    assert np.isclose(out.max(), 1.0)
    assert np.isclose(out.min(), 1.0 / 32)


def test_conductance_rescaling_requires_reference_min(tmp_path):
    """Cap never needed reference_min before, so the optional default would
    reach 1.0 / None and raise an opaque TypeError."""
    tmpdir = str(tmp_path)
    src = write_raster(os.path.join(tmpdir, "c.tif"), np.array([[0.5]]),
                       "float64")
    with pytest.raises(ValueError, match="needs reference_min"):
        apply_rescaling(src, "Cap", os.path.join(tmpdir, "o.tif"),
                        reference_max=1.0, is_conductance=True)


def test_max_resistance_multiplier_is_max_effective_on_resistance():
    assert (max_resistance_multiplier(DOUBLING)
            == max_effective_multiplier(DOUBLING) == 2.0)


def test_proportional_under_conductance_normalizes_a_raised_surface(tmp_path):
    """The Proportional regression, replicating the transformer's fold.

    HALVING_CONDUCTANCE is a x2 resistance modifier as the transformer stores
    it, so resistance reaches 64 against the table's ceiling of 32 and
    Proportional must divide resistance by 2. max_effective_multiplier reads
    1.0 off the inverted table, folds nothing, and leaves the overshoot.
    """
    tmpdir = str(tmp_path)
    assert max_effective_multiplier(HALVING_CONDUCTANCE) == 1.0   # the bug
    M = max_resistance_multiplier(HALVING_CONDUCTANCE, is_conductance=True)
    assert M == 2.0

    folded = HALVING_CONDUCTANCE.copy()
    folded["multiplier"] = folded["multiplier"] * M   # conductance: multiply
    out = read_band(conductance_modified(tmpdir, folded, default_multiplier=M))

    assert np.allclose(1.0 / out, [[0.5, 4, 16], [1, 8, 32]])
    assert np.isclose((1.0 / out).max(), 32.0)        # back under the ceiling


def test_proportional_under_conductance_leaves_a_lowered_surface_alone():
    """A chain that only lowers resistance never overshoots, so Proportional
    must do nothing. A x0.5 resistance modifier is stored as 2.0, which
    max_effective_multiplier reads as a factor of 2 and would use to scale
    resistance UP - creating the overshoot Proportional exists to prevent."""
    doubling_conductance = mod_table(
        [("m", 0.0, 0.5, 1.0), ("m", 0.5, 1.5, 2.0)])
    assert max_effective_multiplier(doubling_conductance) == 2.0   # the bug
    assert max_resistance_multiplier(
        doubling_conductance, is_conductance=True) == 1.0


def test_proportional_under_conductance_preserves_every_ratio(tmp_path):
    """Still a single global constant, so it stays a change of units."""
    tmpdir = str(tmp_path)
    exact = read_band(conductance_modified(tmpdir))
    M = max_resistance_multiplier(HALVING_CONDUCTANCE, is_conductance=True)
    folded = HALVING_CONDUCTANCE.copy()
    folded["multiplier"] = folded["multiplier"] * M
    prop = read_band(conductance_modified(
        tmpdir, folded, default_multiplier=M))
    assert np.allclose(prop / prop[0, 0], exact / exact[0, 0])


def test_rescaling_preserves_nodata(tmp_path):
    tmpdir = str(tmp_path)
    data = np.array([[1.0, 64.0], [float(NODATA_VALUE), 8.0]])
    src = write_raster(os.path.join(tmpdir, "n.tif"), data, "float64",
                       nodata=NODATA_VALUE)
    out = read_band(apply_rescaling(
        src, "Cap", os.path.join(tmpdir, "o.tif"), reference_max=32.0))
    assert out[1, 0] == NODATA_VALUE
    assert np.allclose([out[0, 0], out[0, 1], out[1, 1]], [1.0, 32.0, 8.0])


def test_rescaling_an_all_nodata_tile_is_a_no_op(tmp_path):
    tmpdir = str(tmp_path)
    data = np.full((2, 2), float(NODATA_VALUE))
    src = write_raster(os.path.join(tmpdir, "n.tif"), data, "float64",
                       nodata=NODATA_VALUE)
    out = read_band(apply_rescaling(
        src, "Cap", os.path.join(tmpdir, "o.tif"), reference_max=32.0))
    assert np.all(out == NODATA_VALUE)


def test_min_max_on_a_constant_surface_does_not_divide_by_zero(tmp_path):
    tmpdir = str(tmp_path)
    data = np.full((2, 2), 5.0)
    src = write_raster(os.path.join(tmpdir, "c.tif"), data, "float64")
    out = read_band(apply_rescaling(
        src, "Min-max", os.path.join(tmpdir, "o.tif"),
        reference_max=5.0, reference_min=5.0,
        total_max_multiplier=1.0, total_min_multiplier=1.0))
    assert np.all(np.isfinite(out))


def test_rescaling_rejects_an_unknown_mode(tmp_path):
    tmpdir = str(tmp_path)
    src = write_raster(os.path.join(tmpdir, "c.tif"), np.array([[1.0]]),
                       "float64")
    with pytest.raises(ValueError, match="does not handle mode"):
        apply_rescaling(src, "Nonsense", os.path.join(tmpdir, "o.tif"),
                        reference_max=32.0)
