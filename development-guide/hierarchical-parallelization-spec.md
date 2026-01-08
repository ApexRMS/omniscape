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

The original implementation completely disabled Julia parallelization when tiling was enabled to avoid over-subscription. However, this was based on a misunderstanding of how SyncroSim's multiprocessing works.

**Key Insight**: SyncroSim's MaximumJobs controls **job-level** (OS process-level) parallelism, NOT thread-level parallelism within each job. When tiling with multiprocessing:
- SyncroSim spawns separate Job-X.ssim processes (one per tile)
- SyncroSim manages how many Job-X.ssim processes run concurrently
- Each Job-X.ssim should use ALL available Julia threads (MaximumJobs)
- SyncroSim prevents over-subscription by limiting concurrent Job-X.ssim processes

## Proposed Solution: Enable Julia Parallelization in All Modes

### Core Formula

**IMPORTANT**: In SyncroSim multiprocessing mode, each Job-X.ssim process handles ONE tile. SyncroSim's MaximumJobs setting controls how many Job-X.ssim processes run concurrently at the OS level. Therefore:

- **Loop mode** (sequential tiles): `julia_workers_per_tile = MaximumJobs` (each tile runs alone, uses all workers)
- **Multiprocessing mode** (parallel tiles via SyncroSim): `julia_workers_per_tile = MaximumJobs` (each Job-X.ssim uses all workers, SyncroSim handles job-level parallelism)

### Example Scenarios

**Loop Mode** (tiles run sequentially in single transformer):
| MaximumJobs | Tiles | Julia Workers/Tile | Execution |
| ----------- | ----- | ------------------ | --------- |
| 8           | 4     | 8                  | Tile 1 (8 threads) → Tile 2 (8 threads) → ... |
| 4           | 8     | 4                  | Tile 1 (4 threads) → Tile 2 (4 threads) → ... |

**Multiprocessing Mode** (SyncroSim spawns separate Job-X.ssim per tile):
| MaximumJobs | Tiles | Julia Workers/Job | SyncroSim Behavior |
| ----------- | ----- | ----------------- | ------------------ |
| 8           | 4     | 8                 | Runs up to 8 jobs concurrently (all 4 tiles run simultaneously if resources allow) |
| 4           | 8     | 4                 | Runs 4 jobs concurrently, 2 batches of 4 tiles each |
| 2           | 4     | 2                 | Runs 2 jobs concurrently, 2 batches of 2 tiles each |

**Key Insight**: SyncroSim's multiprocessing controls job-level (process-level) parallelism. Each job should use MaximumJobs Julia threads, and SyncroSim will limit how many jobs run simultaneously based on available resources and MaximumJobs setting.

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

**OPTION A: Use Julia `-t` flag only** (NOT RECOMMENDED - DOES NOT WORK)

```python
# Command line
runOmniscape = f"{jlExe} -t {julia_workers_per_tile} {runFile}"

# config.ini - disable Omniscape's internal parallelization
parallelize = false
parallel_batch_size = 1
```

**Issue**: Setting `parallelize = false` prevents Omniscape.jl from using threads, even if they're available via `-t` flag. Omniscape.jl requires `parallelize = true` in config.ini to actually distribute work across threads.

**OPTION B: Use config.ini** (IMPLEMENTED ✓)

```python
# Command line - provide threads via -t flag
runOmniscape = f"{jlExe} -t {julia_workers_per_tile} {runFile}"

# config.ini - enable Omniscape to use those threads
parallelize = true
parallel_batch_size = {julia_workers_per_tile}
```

**Implementation**: Using Option B with both `-t` flag (ensures threads are available to Julia) and `parallelize = true` (tells Omniscape to use them). This belt-and-suspenders approach is most robust and aligns with [Omniscape.jl documentation](https://docs.circuitscape.org/Omniscape.jl/latest/usage/).

## Implementation Details

### Code Changes Required

**Location**: `src/omniscapeTransformer.py`

#### 1. Calculate Worker Distribution (before line 414)

```python
# ========================================================================
# CALCULATE HIERARCHICAL PARALLELIZATION
# ========================================================================

julia_workers_per_tile = 1  # Default: no Julia parallelization

if multiprocessing.EnableMultiprocessing.item() == "Yes":
    total_workers = int(multiprocessing.MaximumJobs.item())

    if is_tiling:
        num_tiles = manifest['tile_count']

        if mode == "loop":
            # Loop mode: tiles run sequentially, each tile gets ALL workers
            julia_workers_per_tile = total_workers
            ps.environment.update_run_log(
                f"Loop mode parallelization: {num_tiles} tiles (sequential) × {julia_workers_per_tile} Julia workers/tile"
            )
        else:
            # Multiprocessing mode: Each Job-X.ssim handles ONE tile
            # SyncroSim controls job-level parallelism, so each job should use all available workers
            julia_workers_per_tile = total_workers
            ps.environment.update_run_log(
                f"Multiprocessing mode: This job will use {julia_workers_per_tile} Julia workers"
            )
            ps.environment.update_run_log(
                f"Note: SyncroSim will spawn up to {total_workers} concurrent jobs (tiles)"
            )
    else:
        # Non-tiling mode: use all available workers
        julia_workers_per_tile = total_workers
        ps.environment.update_run_log(f"Non-tiling parallelization: {julia_workers_per_tile} Julia workers")
else:
    # Multiprocessing disabled: single-threaded
    julia_workers_per_tile = 1
    ps.environment.update_run_log("Multiprocessing disabled: single-threaded execution")
```

#### 2. Update config.ini Generation (lines 521-539)

```python
# Option B: Control parallelization via Omniscape.jl's config.ini
# This ensures Omniscape actually uses the available threads
if julia_workers_per_tile > 1:
    config_file.write(
        "[Multiprocessing]" + "\n"
        "parallelize = true" + "\n"
        f"parallel_batch_size = {julia_workers_per_tile}" + "\n"
        "\n"
    )
    ps.environment.update_run_log(f"Omniscape parallelization: enabled with {julia_workers_per_tile} workers")
else:
    config_file.write(
        "[Multiprocessing]" + "\n"
        "parallelize = false" + "\n"
        "parallel_batch_size = 1" + "\n"
        "\n"
    )
    ps.environment.update_run_log("Omniscape parallelization: disabled (single-threaded)")
```

**Note**: We set `parallelize = true` when using multiple workers, which tells Omniscape.jl to actually distribute work across threads.

#### 3. Update Julia Execution (lines 561-573)

```python
# ========================================================================
# CONFIGURE JULIA COMMAND (Option B: config.ini controls parallelization)
# ========================================================================

# Build command - parallelization is controlled via config.ini
# We still set -t flag to ensure Julia has threads available for Omniscape to use
if julia_workers_per_tile > 1:
    runOmniscape = f"{jlExe} -t {julia_workers_per_tile} {runFile}"
else:
    runOmniscape = f"{jlExe} {runFile}"

ps.environment.update_run_log(f"Executing: {runOmniscape}")
ps.environment.update_run_log(">>> Starting Omniscape.jl <<<")
```

**Note**: We use both `-t` flag (to make threads available to Julia) and `parallelize = true` in config.ini (to tell Omniscape to use them). The config.ini setting is now the primary control mechanism.

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

| Date       | Author  | Change                         |
| ---------- | ------- | ------------------------------ |
| 2025-12-16 | Initial | Created specification document |
