# Overview.md

## Project Overview

**omniscape** is a SyncroSim package that wraps Omniscape.jl (Julia library) for omnidirectional habitat connectivity analysis based on circuit theory. The package provides a Windows GUI interface through SyncroSim and manages the execution of Julia-based connectivity models.

## Architecture

### SyncroSim Package Structure

This is a SyncroSim package with two main transformers (pipeline stages):

1. **omniscapeTransformer** (`src/omniscapeTransformer.py`) - Main transformer that:

   - Loads configuration from SyncroSim datasheets
   - Validates inputs (resistance files, source files, Julia configuration)
   - Generates INI configuration files for Omniscape.jl
   - Creates and executes Julia scripts via system calls
   - Saves output rasters back to SyncroSim

2. **movementCategoriesTransformer** (`src/movementCategoriesTransformer.py`) - Post-processing transformer that:
   - Categorizes connectivity output into user-defined classes
   - Generates tabular summaries (area, percent cover)
   - Creates categorical rasters from normalized current flow

### Configuration Files

- `src/package.xml` - SyncroSim package definition containing:
  - DataSheet schemas (input/output data structures)
  - Transformer definitions and dependencies
  - UI layout configurations
  - Map and chart visualization settings
- `src/omniscapeEnvironmentv2.yml` - Conda environment specification with Python dependencies (GDAL, rasterio, pysyncrosim, numpy, pandas)

### Key Data Flow

1. User configures scenarios through SyncroSim GUI
2. SyncroSim loads Python transformers in Conda environment
3. Transformer reads datasheets using pysyncrosim
4. Transformer validates inputs and generates Julia configuration
5. Transformer executes Julia script via `os.system()` call
6. Julia installs Omniscape.jl (if needed) and runs connectivity analysis
7. Transformer saves output rasters to scenario-specific directories
8. Results displayed in SyncroSim UI via configured layouts

## Requirements

- **SyncroSim**: Version 3.1.0 or greater
- **Julia**: Version 1.9 or greater with Omniscape.jl
- **Python**: Version 3.12 (managed via Conda)
- **Conda**: Required for managing Python environment

## Important Implementation Notes

### Julia Integration

The package calls Julia via system commands. Critical requirements:

- Julia executable path must NOT contain spaces (`omniscapeTransformer.py:142-143`)
- SyncroSim Library path must NOT contain spaces (`omniscapeTransformer.py:308-309`)
- Julia path must point to `julia.exe` file (`omniscapeTransformer.py:145-146`)

### File Path Handling

- All transformer scripts use absolute paths via `e.data_directory.item()`
- Scenario data stored in: `{data_directory}/Scenario-{scenario_id}/`
- External files (rasters) validated with `show_full_paths=True` parameter
- Output paths constructed using `os.path.join()` for cross-platform compatibility

### Input Validation

Both transformers perform extensive validation:

- Check required fields are present and valid
- Verify CRS and extent matching between resistance and source files
- Validate raster values (e.g., resistance cannot be 0 or negative)
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

### Reclass Table Handling

Special handling for missing values in resistance reclassification:

- User enters -9999 for missing values in GUI
- Transformer converts to "missing" string for Julia (`omniscapeTransformer.py:220`)

## Testing Approach

No automated tests present. Manual testing through SyncroSim GUI:

1. Load example library templates (Omniscape Example)
2. Run scenarios and validate outputs
3. Check maps and charts in Results section

## Version Compatibility

Current version: 2.2.0

- Updated for SyncroSim v3.1.0+ (recent migration from v2)
- Conda environment version: 3 (specified in package.xml)
- Documentation is being updated for SyncroSim v3 compatibility
