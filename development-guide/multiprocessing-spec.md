# Multiprocessing Specification for Omniscape SyncroSim Package

## Executive Summary

The omniscape SyncroSim package currently reads multiprocessing configuration from SyncroSim's `core_Multiprocessing` datasheet and passes it to Omniscape.jl via INI configuration files. However, **Julia is not being launched with threading enabled**, which means the multiprocessing configuration is not actually being utilized by Julia's runtime.

This document provides a complete analysis of the current state, the gap, and implementation requirements to enable true multiprocessing support.

---

## Current State Analysis

### What's Already Implemented

1. **Configuration Reading** (`omniscapeTransformer.py:43`)
   ```python
   multiprocessing = myScenario.datasheets(name = "core_Multiprocessing")
   ```
   The transformer successfully reads the SyncroSim multiprocessing datasheet containing:
   - `EnableMultiprocessing`: Boolean (Yes/No) to enable/disable parallel processing
   - `MaximumJobs`: Integer (1-9999) specifying number of parallel jobs

2. **INI Configuration Generation** (`omniscapeTransformer.py:277-279`)
   ```python
   "[Multiprocessing]" + "\n"
   "parallelize = " + multiprocessing.EnableMultiprocessing.item() + "\n"
   "parallel_batch_size = " + repr(multiprocessing.MaximumJobs.item()) + "\n"
   ```
   These values are written to the `config.ini` file that Omniscape.jl reads.

3. **Boolean Conversion** (`omniscapeTransformer.py:207`)
   ```python
   multiprocessing = multiprocessing.replace({'Yes': 'true', 'No': 'false'})
   ```
   Properly converts SyncroSim's "Yes"/"No" to Julia's "true"/"false".

### The Critical Gap

**Julia is invoked without thread specification** (`omniscapeTransformer.py:305-313`):
```python
jlExe = juliaConfig.juliaPath.item()
runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")
runOmniscape = jlExe + " " + runFile
os.system(runOmniscape)
```

This launches Julia with its **default single-threaded configuration**, meaning:
- The `parallelize = true` setting in the INI file is read by Omniscape.jl
- Omniscape.jl attempts to use parallel processing
- But Julia only has 1 thread available, so parallelization has no effect

---

## How Omniscape.jl Multiprocessing Works

### Threading Architecture

Omniscape.jl uses **Julia's multi-threading** (not multi-processing) to parallelize computations:

1. **Default Behavior**: Omniscape.jl enables parallel processing by default (`parallelize = true`)
2. **Parallel Strategy**: Individual moving windows are solved in parallel across available threads
3. **Batch Processing**: The `parallel_batch_size` parameter controls how many jobs are sent to each worker thread

### Configuration Parameters

From Omniscape.jl documentation:

- **`parallelize`** (boolean, default: true)
  - Controls whether to use parallel processing
  - Currently being set correctly via INI file

- **`parallel_batch_size`** (integer, default: 10)
  - Number of jobs sent to each parallel worker
  - Larger values reduce I/O overhead but may reduce worker utilization
  - Performance tuning parameter for fast individual solves
  - Currently being set to `MaximumJobs` value from SyncroSim

### Thread Specification Requirements

Julia requires thread count to be specified **before runtime** via one of:

1. **Command-line flag** (Julia 1.5+): `julia -t N script.jl` or `julia --threads N script.jl`
   - Can use specific integer: `-t 4`
   - Can use "auto": `-t auto` (Julia 1.7+)
   - Can specify thread pools: `-t 3,1` (3 default + 1 interactive)

2. **Environment variable**: `JULIA_NUM_THREADS=N`
   - Must be set before launching Julia
   - Windows: `set JULIA_NUM_THREADS=4`
   - Bash: `export JULIA_NUM_THREADS=4`
   - PowerShell: `$env:JULIA_NUM_THREADS=4`

---

## Implementation Requirements

### Primary Recommendation: Use `-t` Flag

**Modify the Julia invocation** to include thread specification:

```python
# Current implementation (line 311)
runOmniscape = jlExe + " " + runFile

# Proposed implementation
if multiprocessing.EnableMultiprocessing.item() == "true":
    numThreads = multiprocessing.MaximumJobs.item()
    runOmniscape = jlExe + " -t " + str(numThreads) + " " + runFile
else:
    runOmniscape = jlExe + " " + runFile
```

### Implementation Location

**File**: `src/omniscapeTransformer.py`
**Lines**: 305-313 (Julia execution section)

### Step-by-Step Implementation

1. **Read multiprocessing configuration** (already done at line 43)

2. **Convert boolean to string** (already done at line 207)

3. **Construct Julia command with threading** (needs modification):
   ```python
   # Line ~305-313
   jlExe = juliaConfig.juliaPath.item()
   runFile = os.path.join(dataPath, "omniscape_Required", "runOmniscape.jl")

   if ' ' in dataPath:
       sys.exit("Due to julia requirements, the path to the SyncroSim Library may not contain any spaces.")

   # NEW: Add thread specification if multiprocessing is enabled
   if multiprocessing.EnableMultiprocessing.item() == "true":
       numThreads = int(multiprocessing.MaximumJobs.item())
       runOmniscape = f"{jlExe} -t {numThreads} {runFile}"
   else:
       runOmniscape = f"{jlExe} {runFile}"

   os.system(runOmniscape)
   ```

### Validation Requirements

Add validation to ensure:

1. **Julia version compatibility** (requires Julia 1.5+ for `-t` flag)
2. **Thread count validity** (MaximumJobs should be ≥ 1)
3. **Logical thread count warning** (optional: warn if exceeds system capabilities)

Example validation code:
```python
# After line 146 (Julia path validation)
if multiprocessing.EnableMultiprocessing.item() == "Yes":
    if multiprocessing.MaximumJobs.empty or multiprocessing.MaximumJobs.item() < 1:
        sys.exit("'Maximum Jobs' must be at least 1 when multiprocessing is enabled.")
```

---

## Alternative Approaches

### Option 2: Environment Variable Method

Set `JULIA_NUM_THREADS` before calling Julia:

```python
import os

if multiprocessing.EnableMultiprocessing.item() == "true":
    os.environ["JULIA_NUM_THREADS"] = str(multiprocessing.MaximumJobs.item())

os.system(runOmniscape)
```

**Pros**:
- Works with older Julia versions (pre-1.5)
- No command string modification needed

**Cons**:
- Environment variable persists for entire Python process
- Less explicit than command-line flag
- Could affect other Julia calls (if any)

### Option 3: Hybrid Approach

Use environment variable as fallback:

```python
# Check Julia version, use -t if >= 1.5, else use env var
if multiprocessing.EnableMultiprocessing.item() == "true":
    numThreads = str(multiprocessing.MaximumJobs.item())
    # Try -t flag first
    runOmniscape = f"{jlExe} -t {numThreads} {runFile}"
```

---

## Semantic Consideration: `parallel_batch_size`

### Current Mapping Issue

Currently, `MaximumJobs` is mapped to **both**:
1. Julia thread count (via `-t` flag, once implemented)
2. `parallel_batch_size` parameter in INI file (line 279)

### Semantic Mismatch

These have different meanings:
- **Thread count**: Total number of parallel workers available
- **Batch size**: Number of jobs sent to **each** worker at once

### Recommendation

**Keep current implementation** because:

1. SyncroSim's `core_Multiprocessing` only provides one value (`MaximumJobs`)
2. Users conceptually think of "number of parallel jobs" as one setting
3. Using the same value for both is reasonable:
   - More threads = more parallelism
   - Larger batches = better efficiency for those threads
4. Adding separate UI controls would complicate user experience

### Advanced Option (Future Enhancement)

If fine-grained control is needed, create a custom datasheet:
```xml
<dataSheet name="omniscapeMultiprocessingOptions" displayName="Advanced Multiprocessing" isSingleRow="True">
    <column name="parallelBatchSize" displayName="Parallel Batch Size" dataType="Integer" isOptional="True" defaultValue="10" />
</dataSheet>
```

Then use:
- `MaximumJobs` → Julia thread count (`-t` flag)
- `parallelBatchSize` → INI `parallel_batch_size` parameter

---

## Testing Strategy

### Functional Testing

1. **Baseline Test**: Run scenario with multiprocessing disabled
   - Verify Julia launches with single thread
   - Verify results are correct

2. **Multiprocessing Test**: Run same scenario with multiprocessing enabled (e.g., 4 jobs)
   - Verify Julia launches with 4 threads
   - Verify results match baseline
   - Verify execution time is reduced

3. **Edge Cases**:
   - Test with MaximumJobs = 1 (should work, minimal parallelism)
   - Test with MaximumJobs = large number (e.g., 32)
   - Test toggling between enabled/disabled

### Verification Methods

1. **Thread Count Verification**: Modify Julia script temporarily to check thread count:
   ```julia
   using Omniscape
   println("Julia is using ", Threads.nthreads(), " threads")
   run_omniscape("config.ini")
   ```

2. **Performance Verification**:
   - Time execution with 1 thread vs N threads
   - Expect near-linear speedup for compute-bound scenarios

3. **Output Verification**:
   - Compare raster outputs (should be identical regardless of thread count)
   - Verify no race conditions or threading artifacts

### System Task Manager Verification

On Windows, monitor Task Manager during execution:
- Single thread: ~12.5% CPU usage (1/8 cores on 8-core system)
- 4 threads: ~50% CPU usage (4/8 cores)
- Confirms threads are actually being utilized

---

## Performance Considerations

### Expected Performance Gains

Per Omniscape.jl documentation and SyncroSim best practices:

1. **Speedup is NOT linear**: Due to overhead from:
   - Thread synchronization
   - Memory bandwidth limitations
   - I/O contention

2. **Optimal thread count**: Generally `N - 1` where `N` is logical processor count
   - Leaves one core for system operations
   - Reduces context switching overhead

3. **Model-specific tuning**: Optimal settings depend on:
   - Raster resolution
   - Radius size (window size)
   - Block size
   - Available RAM

### User Guidance (for Documentation)

Recommended starting points:
- **Small models** (<10 minute runtime): Use 2-4 threads
- **Medium models** (10-60 minutes): Use `cores - 1`
- **Large models** (>1 hour): Experiment with different values

Monitor first run to ensure system doesn't thrash (excessive memory paging).

---

## Documentation Updates Required

### User-Facing Documentation

Update `docs/getting_started.md` and tutorial files:

1. Add section on "Enabling Multiprocessing"
2. Explain the Multiprocessing toolbar button
3. Provide thread count recommendations
4. Note Julia 1.5+ requirement for multiprocessing

### Developer Documentation

Update `CLAUDE.md` and `development-guide/`:

1. Document the `-t` flag implementation
2. Explain thread count vs batch size semantics
3. Add testing procedures for multiprocessing
4. Note Julia version requirements

---

## Migration Path for Existing Users

### Backward Compatibility

Implementation is **fully backward compatible**:

1. Users with multiprocessing disabled: No change in behavior
2. Users with multiprocessing enabled: Will now actually get parallel execution
3. Existing INI files: Still valid and respected

### User Communication

Since this fixes a functionality gap rather than breaking changes:
- Release notes should clarify "multiprocessing is now functional"
- Acknowledge that previous "multiprocessing" setting had no effect
- Encourage users to experiment with thread counts

---

## Implementation Checklist

- [ ] Modify Julia invocation to include `-t` flag when multiprocessing enabled
- [ ] Add validation for MaximumJobs value (>= 1)
- [ ] Test with multiprocessing disabled (verify single-threaded)
- [ ] Test with multiprocessing enabled (verify multi-threaded)
- [ ] Verify output consistency across thread counts
- [ ] Measure performance improvement
- [ ] Update CLAUDE.md with implementation notes
- [ ] Update user documentation with multiprocessing guidance
- [ ] Add version requirement note (Julia 1.5+)
- [ ] Test on clean Julia installation (no environment variables set)
- [ ] Create example scenarios demonstrating performance gains

---

## References

### Omniscape.jl Documentation
- User Guide: https://docs.circuitscape.org/Omniscape.jl/latest/usage/
- GitHub: https://github.com/Circuitscape/Omniscape.jl

### Julia Threading Documentation
- Multi-Threading Manual: https://docs.julialang.org/en/v1/manual/multi-threading/
- Parallel Computing: https://docs.julialang.org/en/v1/manual/parallel-computing/

### SyncroSim Multiprocessing
- How-To Guide: https://docs.syncrosim.com/how_to_guides/modelrun_multiproc.html
- Datasheet Reference: https://docs.syncrosim.com/reference/ds_scenario_multiprocessing.html

---

## Conclusion

Enabling multiprocessing in the omniscape package requires a **single-line modification** to add the `-t` flag to the Julia invocation command. The infrastructure for reading configuration and passing parameters to Omniscape.jl is already in place; the only missing piece is launching Julia with threading enabled.

This is a low-risk, high-reward change that will provide immediate performance benefits to users processing large connectivity analyses.
