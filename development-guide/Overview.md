# Overview.md

## Project Overview

**omniscape** is a SyncroSim package that wraps Omniscape.jl (Julia library) for omnidirectional habitat connectivity analysis based on circuit theory. The package provides a Windows GUI interface through SyncroSim and manages the execution of Julia-based connectivity models.

## Architecture

### SyncroSim Package Structure

This is a SyncroSim package with three main transformers (pipeline stages):

1. **prepMultiprocessingTransformer** (`src/prepMultiprocessingTransformer.py`) - Tiling preparation transformer that:
   - Analyzes raster dimensions and calculates optimal tile grid
   - Creates spatial tiles with hybrid buffering (real + optional fake padding)
   - Generates tile manifest for parallel or sequential processing
   - Outputs spatial multiprocessing grid for SyncroSim

2. **omniscapeTransformer** (`src/omniscapeTransformer.py`) - Main transformer that:
   - Loads configuration from SyncroSim datasheets
   - Validates inputs (resistance files, source files, Julia configuration)
   - Generates INI configuration files for Omniscape.jl
   - Implements hierarchical parallelization (tile-level + Julia-level)
   - Creates and executes Julia scripts via system calls
   - Post-processes tile outputs (buffer removal, merging)
   - Saves output rasters back to SyncroSim

3. **movementCategoriesTransformer** (`src/movementCategoriesTransformer.py`) - Post-processing transformer that:
   - Categorizes connectivity output into user-defined classes
   - Generates tabular summaries (area, proportion of area)
   - Creates categorical rasters from normalized current flow

### Configuration Files

- `src/package.xml` - SyncroSim package definition containing:
  - DataSheet schemas (input/output data structures)
  - Transformer definitions and dependencies
  - UI layout configurations
  - Map and chart visualization settings
- `src/omniscapeEnvironmentv2.yml` - Conda environment specification with Python dependencies (GDAL, rasterio, pysyncrosim, numpy, pandas)

### Key Data Flow

#### Standard Workflow (No Tiling)
1. User configures scenarios through SyncroSim GUI
2. SyncroSim loads Python transformers in Conda environment
3. omniscapeTransformer reads datasheets using pysyncrosim
4. Transformer validates inputs and generates Julia configuration
5. Transformer executes Julia script via `os.system()` call with optional threading
6. Julia installs Omniscape.jl (if needed) and runs connectivity analysis
7. Transformer saves output rasters to scenario-specific directories
8. Results displayed in SyncroSim UI via configured layouts

#### Tiling Workflow (With Spatial Multiprocessing)
1. User configures scenario with tiling options (tile count, buffer pixels)
2. **prepMultiprocessingTransformer runs first**:
   - Analyzes raster dimensions
   - Calculates optimal tile grid (or uses user-specified count)
   - Crops tiles from full raster with hybrid buffering
   - Creates tile manifest (JSON) with extent and buffer information
   - Generates spatial multiprocessing grid raster
3. **SyncroSim spawns parallel jobs** (or sequential if multiprocessing disabled):
   - Each job is a copy of the library (Job-1.ssim, Job-2.ssim, etc.)
   - Each job processes one tile
4. **omniscapeTransformer runs in each job**:
   - Loads tile manifest to determine execution mode
   - Processes assigned tile with hierarchical parallelization
   - Removes buffer from outputs
   - Extends tile to full extent (multiprocessing) or collects for merging (loop)
5. **SyncroSim merges outputs** (multiprocessing mode) or transformer merges (loop mode)
6. Results displayed in SyncroSim UI

## Requirements

- **SyncroSim**: Version 3.1.0 or greater
- **Julia**: Version 1.9 or greater with Omniscape.jl
- **Python**: Version 3.12 (managed via Conda)
- **Conda**: Required for managing Python environment

## Important Implementation Notes

### Julia Integration

The package calls Julia via system commands. Julia configuration is managed through the `core_JlConfig` system datasheet (Library > Options > Julia Tools).

Critical requirements:

- Julia executable path must be configured in Library > Options > Julia Tools
- Julia executable path must NOT contain spaces
- SyncroSim Library path must NOT contain spaces
- Julia path must point to `julia.exe` file

### File Path Handling

- All transformer scripts use absolute paths via `e.data_directory.item()`
- Scenario data stored in: `{data_directory}/Scenario-{scenario_id}/`
- External files (rasters) validated with `show_full_paths=True` parameter
- Output paths constructed using `os.path.join()` for cross-platform compatibility

### Input Validation

Both transformers perform extensive validation:

- Check required fields are present and valid
- Verify CRS and extent matching between resistance and source files
- Validate raster values (e.g., resistance cannot be 0 or negative). Under
  reclassification the input holds land cover class IDs, where 0 and negative
  fill codes are ordinary, so that check moves to the reclass table and to the
  final surface - see Building the Resistance Surface
- Exit with clear error messages using `sys.exit()`

### Boolean Handling

SyncroSim stores booleans as "Yes"/"No" strings. Transformers convert these to "true"/"false" for Julia configuration files:

```python
generalOptions = generalOptions.replace({'Yes': 'true', 'No': 'false'})
```

### Datasheet Patterns

Common pysyncrosim patterns used throughout:

- `myScenario.datasheets(name="...")` - Load scenario data
- `show_full_paths=True` - Get absolute paths for external files
- `include_key=True` - Include primary key columns
- `myParentScenario.save_datasheet(...)` - Save outputs to parent scenario (not current)

### Building the Resistance Surface

The surface handed to Omniscape is built in `omniscapeTransformer.py`, inside the
per-tile loop, in this order:

```
Required.resistanceFile          raw input (land cover class IDs, or resistance)
  -> apply_reclass_table()       omniscape_ResistanceReclass/[tile-N-]reclass-resistance.tif
  -> apply_resistance_modifier() omniscape_ResistanceModifier/[tile-N-]mod{i}-resistance.tif
  -> config.ini resistance_file  -> Omniscape.jl
```

**The order is load-bearing.** Reclassification is an exact-equality lookup on
nominal land cover class IDs; multiplication is only defined on a ratio scale.
Reclassifying first turns class IDs into resistance values, which is the only
scale on which a modifier's multiplier means anything. Applying a modifier first
scales the class IDs themselves - class 41 with a 1.5x multiplier becomes 61.5,
which matches no reclass row, and an integral multiplier is worse still because
it can land on another class's ID and silently pick up its resistance.

#### Reclassification

`apply_reclass_table` (`helperFunctions.py`) reproduces Omniscape.jl's
`reclassify_resistance!` so the step can run here rather than inside Julia:

- exact-equality lookup against the original pixel values, so reclassifications
  never cascade
- class IDs absent from the table are left unchanged
- two rows for one class ID: the last row wins
- the user enters -9999 for missing values in the GUI, and the helper turns those
  pixels into no-data, matching the literal `missing` Omniscape reads from a
  reclass table file
- output is always float64 with a declared no-data of -9999; float32 would move a
  value like 0.1 enough to flip an exact comparison against `r_cutoff`, and a NaN
  no-data would not survive the modifier stage, which restores no-data with `==`

`config.ini` therefore always carries `reclassify_resistance = false` and an empty
`reclass_table`. The value is left empty rather than pointed at a file because
Omniscape warns whenever a table is supplied with reclassification switched off.
`reclass_table.txt` is no longer written.

#### Resistance Modifiers

Each row of `ResistanceModifiers` names a secondary raster and is applied in turn,
so multipliers compound. `ResistanceModifierTable` maps ranges of that raster's
values to multipliers, half-open as `minValue <= value < maxValue`. Ranges may not
overlap - `validate_threshold_bands(..., allow_gaps=True)` rejects that - but gaps
are allowed and mean "leave this range alone": a pixel matching no band keeps a
multiplier of 1.0.

Omniscape inverts the surface to conductance *after* reclassification, so when
`Resistance is conductance` is Yes the modifiers would be scaling conductance. The
transformer inverts the multipliers (`m -> 1/m`) in that case and says so in the
run log, so "multiply resistance by 2" means that in both modes.

#### Validation

Three checks, each catching something different:

- the raw input must be positive, but only when it is not going to be reclassified
- no reclass table row may map a class to a resistance of 0 or less (-9999, meaning
  no-data, is exempt) - the cheapest check, and it fires before any raster is read
- the final surface, after reclass and modifiers, must be positive. This one
  matters because Omniscape's own guard calls `@error` and returns rather than
  raising: Julia exits 0 and writes nothing, so without the check the run looks
  successful until the outputs turn up missing. When an unlisted class ID is the
  cause, the error names it.

#### Tiling

Reclassification is a per-pixel lookup with no neighbourhood, so splitting into
tiles cannot change a value. Both the reclassified and modified rasters get the
same buffer-crop, full-extent-extend and merge treatment as Omniscape's own
outputs. Note that a modifier's focal window is computed per tile from a windowed
read of the full modifier raster, so near a tile edge it sees NaN padding rather
than the neighbouring tile - the tile buffer only covers this when the buffer is
at least as wide as the focal radius.

## Advanced Features

### Hierarchical Parallelization

Combines two levels of parallelization for optimal performance:

**Tile-Level Parallelization**:
- Multiple tiles processed simultaneously (via SyncroSim multiprocessing)
- Each tile is an independent job with its own resources

**Julia-Level Parallelization**:
- Julia threading within each tile job (via `-t` flag)
- Workers distributed intelligently: `julia_workers_per_tile = MaximumJobs / num_tiles`
- Loop mode: Each tile gets all workers (sequential processing)
- Multiprocessing mode: Workers divided across parallel tiles

**Example**: 8 workers, 4 tiles → 4 parallel jobs × 2 Julia threads each = 8 total workers

**Configuration**:
- Controlled via `core_Multiprocessing` datasheet (MaximumJobs)
- Automatic calculation prevents over-subscription
- Detailed logging shows worker distribution strategy

**Documentation**: `development-guide/hierarchical-parallelization-spec.md`

### Real Data Buffering

Tiles use intelligent buffering with real overlapping data to eliminate edge effects:

**Internal Tile Boundaries**:
- Use real overlapping data from full raster
- Tiles overlap by `2 × buffer_pixels`
- Omniscape computes with actual neighbor values
- Result: No edge effects at tile boundaries ✓

**Raster Boundaries**:
- Use real data only - no padding beyond raster edge
- Tiles at edges are clipped to raster bounds
- Preserves authentic edge conditions of the study area

**How It Works**:
1. Calculate buffered extent (tile extent ± buffer_pixels)
2. Clip buffered extent to raster bounds (crop only what exists)
3. Crop real data from full raster
4. No artificial padding - real data only

**Example**:
```
Internal Tile: [REAL:50-150] - All real overlapping data
Edge Tile: [REAL:0-150] - Real data only, clipped at boundary
```

**Documentation**:
- `development-guide/TILING_EDGE_EFFECTS_ANALYSIS.md`

### Tiling Options

**Tile Count**:
- Auto-calculation: Targets ~100K pixels per tile
- Manual override: Specify exact tile count
- Set to 1 to disable tiling

**Buffer Pixels**:
- Recommended: Set equal to Omniscape radius
- Creates overlap between tiles using real overlapping data
- Prevents edge effects at internal tile boundaries
- No padding beyond raster boundary - real data only
- Default: 0 (no buffer)

## Testing Approach

### Automated Tests

Unit tests for hierarchical parallelization logic:
- Location: `tests/test_hierarchical_parallelization.py`
- Coverage: Worker distribution, efficiency, over-subscription, edge cases
- Run: `python test_hierarchical_parallelization.py`
- Status: 23 tests, all passing ✓

Log validation tool:
- Location: `tests/validate_logs.py`
- Validates correct parallelization behavior from run logs
- Usage: `python validate_logs.py <log_file_or_directory>`

### Manual Testing

Through SyncroSim GUI:
1. Load example library templates (Omniscape Example)
2. Configure tiling and multiprocessing options
3. Run scenarios and validate outputs
4. Check maps and charts in Results section
5. Verify logs show correct parallelization strategy

Detailed testing guide: `tests/MANUAL_TESTING_GUIDE.md`

## Version Compatibility

Current version: 2.6.0

- **SyncroSim**: v3.1.0+ required
- **Julia**: v1.9+ with Omniscape.jl
- **Python**: 3.12 (Conda environment version 3.5)
- **Conda environment**: omniscapeEnvironmentv2.yml

### Recent Updates

**v2.6.0**:
- Simplified buffering to use real overlapping data only
- No artificial padding beyond raster boundaries
- Preserves authentic edge conditions of study area

**v2.5.0**:
- Migrated Julia configuration to core system datasheet (core_JlConfig)
- Hierarchical parallelization (tile + Julia threading)
- Improved tiling with buffer support
- Helper functions for tile processing

**v2.4.0**:
- Added prepMultiprocessingTransformer for tiling
- Spatial multiprocessing support
- Tile manifest system
