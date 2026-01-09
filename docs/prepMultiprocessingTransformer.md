# PrepMultiprocessing Transformer

## Overview

The PrepMultiprocessing transformer is a preprocessing step that enables spatial parallelization of large Omniscape analyses. It divides the landscape into manageable tiles, pre-processes them with appropriate buffering, and creates a tile manifest that guides the main Omniscape transformer execution.

## Purpose

**Problem:** Large landscape connectivity analyses can take hours or days to complete on a single processor, even with Julia's multi-threading.

**Solution:** Divide the landscape into spatial tiles that can be processed independently across multiple CPU cores, then merge the results. This provides dramatic speedups (2-8× typical) for large rasters.

## When to Use

**Use PrepMultiprocessing when:**

- Raster dimensions are large (typically >3000×3000 pixels)
- Multiple CPU cores are available (4-12 cores recommended)
- Analysis parameters create significant computational load
- Runtime on full extent would be >30 minutes

**Skip PrepMultiprocessing when:**

- Small rasters (<2000×2000 pixels) - overhead exceeds benefits
- Very large radius relative to raster size - tiles can't be independent
- Limited system RAM - tiling won't help if memory-constrained

## How It Works

### 1. Parameter Analysis

The transformer first gathers key analysis parameters:

- **Radius**: Omniscape search radius (pixels) - determines tile overlap requirements
- **Block size**: Omniscape internal block size - affects minimum tile dimensions
- **Available cores**: From `MaximumJobs` setting
- **Parallelization intensity**: User preference (Conservative/Balanced/Aggressive/Auto)
- **Buffer multiplier**: How much overlap between tiles (default: 1.0× radius)

### 2. Tile Configuration Calculation

The transformer calculates optimal tile count using multiple constraints:

**Minimum tile size constraint:**

```
min_tile_dimension = 7 × block_size
min_tile_dimension_with_buffer = min_tile_dimension + (2 × buffer_pixels)
```

Tiles must be large enough for efficient Omniscape processing after buffering. The 7× multiplier ensures sufficient analysis area within each tile.

**Parallelization intensity constraint:**

```
Conservative: tile_count = available_cores × 1.0
Balanced:     tile_count = available_cores × 2.0  (default)
Aggressive:   tile_count = available_cores × 4.0
```

More tiles enable better load balancing but increase merging overhead.

**The final tile count is the minimum of all constraints** to ensure tiles are large enough and don't over-parallelize.

### 3. Grid Generation

Creates a spatial grid matching the raster's aspect ratio:

- Calculates grid dimensions (rows × cols) that best match raster aspect ratio
- Ensures even tile distribution across the landscape
- Assigns unique tile IDs (1 to N)
- Masks NoData areas (excluded from processing)

### 4. Tile Pre-processing

For each tile, the transformer:

**Crops the base tile** from resistance and source rasters

**Applies buffering** (if buffer_multiplier > 0):

- Extends tile extent by `buffer_pixels` in all directions
- Uses real overlapping data from adjacent tiles
- Does NOT pad beyond raster boundaries (no artificial values)
- Edge tiles naturally have less buffer at boundaries

**Saves pre-processed tiles** as GeoTIFF files:

- `tile-{N}-resistance.tif`
- `tile-{N}-source.tif` (if source file provided)

### 5. Manifest Creation

Creates `tile_manifest.json` containing:

- Tile count and configuration
- Buffer settings
- File paths for each tile's input rasters
- Original and buffered extents for each tile
- Full raster metadata (CRS, transform, dimensions)

This manifest guides the main transformer's execution and result merging.

## Key Settings

### Parallelization Intensity (TilingOptions datasheet)

- **Auto** (default): Balanced approach, 2× available cores
- **Conservative**: Match core count, larger tiles, less overhead
- **Balanced**: 2× core count, good balance of parallelization and efficiency
- **Aggressive**: 4× core count, maximum parallelization, more overhead

### Buffer Multiplier (TilingOptions datasheet)

Controls tile overlap to prevent edge effects:

- **1.0** (default): Buffer = radius (recommended minimum)
- **>1.0**: More overlap, safer but more redundant computation
- **<1.0**: Less overlap, faster but may have edge artifacts (not recommended)

### RAM Per Thread (TilingOptions datasheet)

Estimated memory usage per Julia thread (default: 16 GB):

- Used to calculate safe thread counts based on available RAM
- Adjust based on actual memory usage observations
- Conservative estimate prevents out-of-memory errors

## Output Files

**Scenario directory:**

- `smpGrid-{N}-{size}K.tif` - Spatial tile grid raster (also in core_SpatialMultiprocessing datasheet)

**OmniscapeTiles subdirectory:**

- `tile_manifest.json` - Tile configuration and metadata
- `tile-{N}-resistance.tif` - Pre-cropped resistance for each tile
- `tile-{N}-source.tif` - Pre-cropped source for each tile (if applicable)
- Copy of grid raster

## Performance Considerations

### Speedup Expectations

- **2-4 cores**: 1.5-3× speedup typical
- **6-8 cores**: 3-6× speedup typical
- **12+ cores**: 4-8× speedup typical (diminishing returns due to overhead)

Actual speedup depends on:

- Tile count vs core count
- Raster size relative to tile size
- I/O speed (SSD vs HDD)
- Merging overhead

### Memory Requirements

Each tile uses: `julia_workers_per_tile × RAM_per_thread`

Peak memory: `simultaneous_tiles × RAM_per_tile`

The transformer estimates memory usage and warns if it may exceed available RAM.

### Radius-to-Raster Ratio

Critical factor for parallelization effectiveness:

- **radius < 10% of raster dimension**: Excellent parallelization
- **radius = 10-25% of raster dimension**: Good parallelization
- **radius = 25-50% of raster dimension**: Limited parallelization
- **radius > 50% of raster dimension**: Minimal or no parallelization

## Execution Workflow

1. **User runs PrepMultiprocessing transformer** - analyzes raster, creates tiles
2. **User runs Omniscape transformer** - automatically detects tiles and enters multiprocessing mode:
   - **Multiprocessing mode**: SyncroSim creates N parallel jobs (one per tile)
   - Each job processes one tile independently with Julia multi-threading
   - Results are automatically merged after all tiles complete
3. **Final outputs** - seamless merged results in output datasheet

The tile manifest path (`Scenario-{ID}/OmniscapeTiles/tile_manifest.json`) is the signal that enables multiprocessing mode.

## Troubleshooting

**"Raster too small for multiprocessing"**

- Raster doesn't meet minimum size requirements for tiling
- Use single-process mode (skip PrepMultiprocessing)

**"Estimated memory usage exceeds available RAM"**

- Reduce MaximumJobs (fewer simultaneous tiles)
- Use Conservative intensity
- Reduce RamPerThreadGB if estimate is too high

**Single-core usage despite tiling**

- Radius too large relative to raster size
- Tiles have too much overlap to parallelize effectively
- Reduce radius or use larger raster

**Edge artifacts in results**

- Buffer multiplier too low
- Increase to 1.0 or higher
- Check tile boundaries in output for discontinuities
