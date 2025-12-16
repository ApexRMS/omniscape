# Hierarchical Parallelization Specification

## Overview

This document specifies the implementation of hierarchical parallelization in the omniscape package, which combines:
1. **Tile-level parallelization**: Multiple tile jobs running simultaneously (via SyncroSim multiprocessing)
2. **Julia-level parallelization**: Julia threading within each tile job (via Omniscape.jl)

## Current Behavior (Before Implementation)

### Non-Tiling Mode
- Single job processes the full extent
- Julia parallelization enabled via:
  - Command line: `-t N` flag
  - Config.ini: `parallelize = true`, `parallel_batch_size = N`
- Worker count: `N = MaximumJobs` from `core_Multiprocessing` datasheet

### Tiling Mode
- Multiple tile jobs run (either sequentially or via SyncroSim multiprocessing)
- Julia parallelization **DISABLED** (lines 414-426, 452-453 in omniscapeTransformer.py)
- Rationale: Avoid over-subscription when multiple tile jobs run simultaneously

## Problem Statement

When tiling is enabled, we completely disable Julia parallelization to avoid over-subscription. However, this leaves performance on the table in two scenarios:

1. **Uneven division**: If user specifies `MaximumJobs = 8` and there are 3 tiles, we only use 3 cores
2. **Small tiles**: Tiles might be small enough that Julia parallelism within each tile would improve performance

## Proposed Solution: Intelligent Worker Distribution

### Core Formula

```
julia_workers_per_tile = floor(MaximumJobs / num_tiles)
actual_total_workers = num_tiles × julia_workers_per_tile
```

### Example Scenarios

| MaximumJobs | Tiles | Julia Workers/Tile | Total Workers Used | Efficiency |
|-------------|-------|--------------------|--------------------|------------|
| 8           | 4     | 2                  | 8                  | 100%       |
| 8           | 3     | 2                  | 6                  | 75%        |
| 8           | 2     | 4                  | 8                  | 100%       |
| 8           | 1     | 8                  | 8                  | 100%       |
| 4           | 8     | 1 (min)            | 8                  | 200%*      |

\* Over-subscription case - see Safety Checks below

## Julia Parallelization: Avoiding Double-Counting

### Problem: Two Parallelization Settings

Omniscape.jl accepts parallelization configuration in two places:

1. **Julia Runtime** (`-t N` flag):
   - Sets global Julia thread count
   - Controls `Threads.@threads` macros
   - Example: `julia -t 4 script.jl` → 4 threads available

2. **Omniscape.jl Config** (`config.ini`):
   - `parallelize = true/false`
   - `parallel_batch_size = N`
   - Controls Omniscape.jl's internal parallel processing

### Risk: Over-Subscription

If both are set, we could get multiplicative parallelization:
- `-t 4` (4 Julia threads)
- `parallel_batch_size = 2` (Omniscape spawns 2 parallel tasks)
- **Actual workers**: 4 × 2 = 8 threads competing for resources

### Solution: Single Point of Control

**OPTION A: Use Julia `-t` flag only** (RECOMMENDED)
```python
# Command line
runOmniscape = f"{jlExe} -t {julia_workers_per_tile} {runFile}"

# config.ini - disable Omniscape's internal parallelization
parallelize = false
parallel_batch_size = 1
```

**OPTION B: Use config.ini only**
```python
# Command line - no -t flag
runOmniscape = f"{jlExe} {runFile}"

# config.ini - control parallelization here
parallelize = true
parallel_batch_size = {julia_workers_per_tile}
```

**Recommendation**: Use Option A (`-t` flag only) because:
- More explicit control
- Julia's thread pool is better integrated with the runtime
- Easier to verify worker count (can log `Threads.nthreads()` in Julia)

## Implementation Details

### Code Changes Required

**Location**: `src/omniscapeTransformer.py`

#### 1. Calculate Worker Distribution (before line 414)

```python
# ========================================================================
# CALCULATE HIERARCHICAL PARALLELIZATION
# ========================================================================

julia_workers_per_tile = 1  # Default: no Julia parallelization

if is_tiling and multiprocessing.EnableMultiprocessing.item() == "true":
    total_workers = int(multiprocessing.MaximumJobs.item())
    num_tiles = manifest['tile_count']

    # Distribute workers across tiles
    julia_workers_per_tile = max(1, total_workers // num_tiles)

    # Calculate actual utilization
    actual_workers_used = num_tiles * julia_workers_per_tile
    efficiency = (actual_workers_used / total_workers) * 100

    # Log parallelization strategy
    ps.environment.update_run_log(
        f"Hierarchical parallelization: {num_tiles} tiles × {julia_workers_per_tile} Julia workers/tile "
        f"= {actual_workers_used} total workers ({efficiency:.0f}% of requested {total_workers})"
    )

    # Warning if low efficiency
    if efficiency < 75 and total_workers > 2:
        ps.environment.update_run_log(
            f"WARNING: Low worker efficiency ({efficiency:.0f}%). Consider adjusting tile count "
            f"to better divide {total_workers} workers.",
            report_type="message"
        )
```

#### 2. Update config.ini Generation (lines 414-426)

```python
# ALWAYS disable Omniscape.jl's internal parallelization
# (we control parallelization via Julia's -t flag instead)
config_file.write(
    "[Multiprocessing]" + "\n"
    "parallelize = false" + "\n"
    "parallel_batch_size = 1" + "\n"
)
```

**Note**: We now use the same config for both tiling and non-tiling modes, avoiding inconsistency.

#### 3. Update Julia Execution (lines 451-458)

```python
# ========================================================================
# CONFIGURE JULIA THREADING
# ========================================================================

runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")

# Determine Julia thread count based on execution mode
if is_tiling:
    # Tiling mode: use calculated workers per tile
    num_threads = julia_workers_per_tile
else:
    # Non-tiling mode: use all available workers
    if multiprocessing.EnableMultiprocessing.item() == "true":
        num_threads = int(multiprocessing.MaximumJobs.item())
    else:
        num_threads = 1

# Build command with threading
if num_threads > 1:
    runOmniscape = f"{jlExe} -t {num_threads} {runFile}"
    ps.environment.update_run_log(f"Julia threading: {num_threads} threads")
else:
    runOmniscape = f"{jlExe} {runFile}"
    ps.environment.update_run_log(f"Julia threading: disabled (single-threaded)")
```

### Variable Tracking

To maintain clarity, we introduce:
- `julia_workers_per_tile`: Number of Julia threads per tile job (scoped to transformer)
- Replaces the previous `numThreads` variable with clearer intent

## Safety Checks and Warnings

### 1. Over-Subscription Detection

```python
import os

# Get physical CPU count
physical_cores = os.cpu_count() or 1

if is_tiling and mode == "multiprocessing":
    # In multiprocessing mode, all tiles may run simultaneously
    potential_workers = num_tiles * julia_workers_per_tile

    if potential_workers > physical_cores * 1.5:
        ps.environment.update_run_log(
            f"WARNING: Potential over-subscription detected! "
            f"{potential_workers} workers requested but only {physical_cores} physical cores available. "
            f"Performance may degrade due to context switching.",
            report_type="message"
        )
```

### 2. Minimum Worker Guarantee

```python
# Ensure at least 1 Julia worker per tile
julia_workers_per_tile = max(1, total_workers // num_tiles)
```

### 3. Efficiency Threshold

If efficiency < 75%, warn user that tile count doesn't divide evenly:
```
WARNING: Low worker efficiency (66%). Consider adjusting tile count to better divide 8 workers.
```

## User-Facing Behavior

### Example 1: Perfect Division
- User sets `MaximumJobs = 8`
- Tiling creates 4 tiles
- **Result**: 4 tile jobs × 2 Julia threads = 8 workers
- Log message: "Hierarchical parallelization: 4 tiles × 2 Julia workers/tile = 8 total workers (100% of requested 8)"

### Example 2: Uneven Division
- User sets `MaximumJobs = 8`
- Tiling creates 3 tiles
- **Result**: 3 tile jobs × 2 Julia threads = 6 workers (2 unused)
- Log message: "Hierarchical parallelization: 3 tiles × 2 Julia workers/tile = 6 total workers (75% of requested 8)"
- Warning: "Low worker efficiency (75%). Consider adjusting tile count to better divide 8 workers."

### Example 3: More Tiles Than Workers
- User sets `MaximumJobs = 4`
- Tiling creates 8 tiles
- **Result**: 8 tile jobs × 1 Julia thread = 8 workers (sequential batches of 4)
- Log message: "Hierarchical parallelization: 8 tiles × 1 Julia workers/tile = 8 total workers (200% of requested 4)"
- Note: SyncroSim's job scheduler will run 4 tiles at a time (2 batches)

## Testing Strategy

### Manual Testing Checklist

1. **Non-tiling mode with multiprocessing**
   - Set `MaximumJobs = 4`, disable tiling
   - Verify Julia runs with `-t 4`
   - Check log for "Julia threading: 4 threads"

2. **Tiling mode: perfect division**
   - Set `MaximumJobs = 8`, create 4 tiles
   - Verify each tile runs with `-t 2`
   - Check log for "4 tiles × 2 Julia workers/tile = 8 total workers (100%)"

3. **Tiling mode: uneven division**
   - Set `MaximumJobs = 8`, create 3 tiles
   - Verify each tile runs with `-t 2`
   - Check log for efficiency warning

4. **Tiling mode: more tiles than workers**
   - Set `MaximumJobs = 2`, create 4 tiles
   - Verify each tile runs with `-t 1` (single-threaded)
   - Check log for "4 tiles × 1 Julia workers/tile"

5. **Non-tiling mode: single-threaded**
   - Disable multiprocessing, disable tiling
   - Verify Julia runs without `-t` flag
   - Check log for "Julia threading: disabled"

### Performance Verification

Compare runtime for a test dataset:
- Baseline: Tiling only (8 tiles, 1 Julia worker each)
- Hierarchical: 4 tiles × 2 Julia workers
- Expected: Hierarchical should be 10-30% faster (depends on tile size and Julia's parallel efficiency)

## Edge Cases

### 1. Single Tile
- `num_tiles = 1`, `MaximumJobs = 8`
- Result: 1 tile × 8 Julia workers = 8 workers
- This is equivalent to non-tiling mode (optimal)

### 2. Tile Count > MaximumJobs
- `num_tiles = 16`, `MaximumJobs = 4`
- Result: 16 tiles × 1 Julia worker = 16 total workers
- SyncroSim schedules 4 at a time, runs 4 batches sequentially

### 3. MaximumJobs = 1
- Any tile count
- Result: Each tile runs single-threaded
- No parallelization (as expected)

### 4. Loop Mode (Non-Multiprocessing)
- Tiles processed sequentially in single job
- Each tile can use all `MaximumJobs` workers
- Result: `julia_workers_per_tile = MaximumJobs` (NOT divided)

**Important**: In loop mode, tiles don't compete for resources!

```python
if mode == "loop":
    # Sequential processing: each tile gets full worker allocation
    julia_workers_per_tile = int(multiprocessing.MaximumJobs.item()) if multiprocessing.EnableMultiprocessing.item() == "true" else 1
else:
    # Multiprocessing mode: divide workers across tiles
    julia_workers_per_tile = max(1, total_workers // num_tiles)
```

## Configuration Compatibility

### SyncroSim Datasheets
- `core_Multiprocessing.EnableMultiprocessing`: Controls whether tiling jobs run in parallel
- `core_Multiprocessing.MaximumJobs`: Total worker budget to distribute
- `omniscape_TilingOptions.TileCount`: Number of spatial tiles (affects division)

### Backward Compatibility
- Existing scenarios without tiling: **No change** (Julia threading works as before)
- Existing scenarios with tiling: **Performance improvement** (Julia threading now enabled per tile)
- No breaking changes to datasheets or configuration

## Open Questions

### 1. Omniscape.jl Parallelization Model
- **Question**: Does Omniscape.jl benefit from threading for small tiles (<100K pixels)?
- **Investigation needed**: Benchmark small vs large tiles with/without Julia threading
- **Hypothesis**: Threading overhead may exceed benefits for tiles <50K pixels

### 2. Memory Scaling
- **Question**: How does Julia thread count affect memory usage?
- **Concern**: Each Julia thread might allocate working memory
- **Mitigation**: Monitor memory usage in testing; reduce `julia_workers_per_tile` if needed

### 3. I/O Bottlenecks
- **Question**: For large tiles, is I/O the bottleneck rather than compute?
- **Investigation**: Profile tile processing to identify bottleneck (disk I/O vs CPU)
- **Implication**: If I/O-bound, Julia threading won't help

## Future Enhancements

### 1. Auto-Tuning
Automatically adjust tile count and Julia workers based on:
- Total raster size
- Available memory
- CPU core count
- Target: Maximize `num_tiles × julia_workers_per_tile` while avoiding over-subscription

### 2. Adaptive Tile Sizing
- Smaller tiles on edges (less work) → more Julia workers
- Larger tiles in center (more work) → fewer Julia workers
- Requires tile workload estimation

### 3. User Control
Add to `omniscape_TilingOptions`:
- `JuliaWorkersPerTile`: Manual override (advanced users)
- `MaxTotalWorkers`: Cap on `num_tiles × julia_workers`

## References

- Julia threading documentation: https://docs.julialang.org/en/v1/manual/multi-threading/
- Omniscape.jl: https://github.com/Circuitscape/Omniscape.jl
- Current implementation: `src/omniscapeTransformer.py` lines 414-458

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2025-12-16 | Initial | Created specification document |
