# Buffer Implementation Specification for Omniscape SyncroSim Package

**Version:** 1.0  
**Date:** December 11, 2025  
**Author:** Alex (ApexRMS)  
**Purpose:** Reduce edge effects in connectivity analysis through automated buffer creation

---

## Executive Summary

This specification describes the implementation of automated buffer creation and removal for the Omniscape SyncroSim package. The buffer system will expand input rasters before processing, run Omniscape.jl on the buffered extent, and crop outputs back to the original extent. This reduces edge effects in connectivity analysis while maintaining output compatibility with existing workflows.

---

## 1. Overview

### 1.1 Problem Statement

Edge effects in connectivity analysis occur when movement corridors or current flows reach the boundary of the study area. Without buffers, these analyses artificially constrain movement at edges, leading to:
- Underestimation of connectivity near boundaries
- Artificial "sinks" at raster edges
- Biased flow patterns in peripheral areas

### 1.2 Solution Approach

Implement a three-stage buffer workflow:
1. **Pre-processing**: Expand resistance and source rasters by N pixels
2. **Processing**: Run Omniscape.jl on buffered extent
3. **Post-processing**: Crop outputs back to original extent

### 1.3 Design Principles

- **Backward Compatible**: Default behavior (no buffer) unchanged
- **User-Configurable**: Buffer size controlled via datasheet
- **Transparent**: Buffered intermediates stored for debugging
- **Efficient**: Minimal performance overhead for buffer=0 case

---

## 2. Architecture Changes

### 2.1 New Functions

#### 2.1.1 `buffer_raster()`

**Purpose**: Create buffered version of input raster by extending edges

**Signature**:
```python
def buffer_raster(input_path, output_path, buffer_pixels):
    """
    Create a buffered version of a raster by extending edges.
    
    Args:
        input_path (str): Path to input raster file
        output_path (str): Path for buffered output raster
        buffer_pixels (int): Number of pixels to buffer on each side
    
    Returns:
        dict: Original bounds metadata for later cropping
            - 'height': Original raster height
            - 'width': Original raster width
            - 'transform': Original affine transform
            - 'bounds': Original geographic bounds
            - 'crs': Coordinate reference system
    
    Raises:
        IOError: If input file cannot be read
        ValueError: If buffer_pixels is negative
    """
```

**Implementation Details**:
- Use `rasterio` for raster I/O
- Use `numpy.pad()` with `mode='edge'` for replicating edge values
- Recalculate affine transform for expanded extent
- Preserve all metadata (CRS, nodata, datatype)
- Handle both integer and float rasters
- Support single-band rasters (Omniscape requirement)

**Padding Strategy**:
- **Mode**: `'edge'` - Replicates edge pixel values outward
- **Rationale**: Avoids creating artificial barriers or habitat discontinuities
- **Alternatives Considered**:
  - `'reflect'`: Could create unrealistic mirrored patterns
  - `'constant'`: Would require choosing arbitrary resistance value
  - `'wrap'`: Only appropriate for toroidal study areas

#### 2.1.2 `crop_to_original()`

**Purpose**: Crop buffered output back to original extent

**Signature**:
```python
def crop_to_original(buffered_path, output_path, original_bounds, buffer_pixels):
    """
    Crop buffered output back to original extent.
    
    Args:
        buffered_path (str): Path to buffered raster
        output_path (str): Path for cropped output
        original_bounds (dict): Bounds metadata from buffer_raster()
        buffer_pixels (int): Buffer size used (for window calculation)
    
    Returns:
        None
    
    Raises:
        IOError: If buffered file cannot be read
        ValueError: If window extends beyond buffered raster
    """
```

**Implementation Details**:
- Use `rasterio.windows.Window` for efficient cropping
- Calculate window offset based on buffer_pixels
- Restore original affine transform
- Preserve metadata from buffered raster
- Handle nodata values appropriately

### 2.2 Modified Functions

#### 2.2.1 `run_scenario()`

**Changes Required**:

1. **Add buffer_pixels parameter**:
```python
def run_scenario(scenarioId, myLibrary, e, buffer_pixels=0):
```

2. **Buffer input rasters** (after loading file paths, before INI generation):
```python
if buffer_pixels > 0:
    # Create scenario directory if needed
    os.makedirs(scenario_data_directory, exist_ok=True)
    
    # Define buffered file paths
    buffered_resistance = os.path.join(
        scenario_data_directory, 
        'buffered_resistance.tif'
    )
    buffered_source = os.path.join(
        scenario_data_directory, 
        'buffered_source.tif'
    )
    
    # Buffer both rasters
    original_bounds = buffer_raster(
        resistance_path, 
        buffered_resistance, 
        buffer_pixels
    )
    buffer_raster(
        source_path, 
        buffered_source, 
        buffer_pixels
    )
    
    # Update paths for INI generation
    resistance_path = buffered_resistance
    source_path = buffered_source
```

3. **Crop outputs** (after Julia execution, before saving to SyncroSim):
```python
if buffer_pixels > 0:
    output_files = [
        'normalized_cum_currmap.tif',
        'cum_currmap.tif',
        'flow_potential.tif'
    ]
    
    for output_file in output_files:
        buffered_output = os.path.join(scenario_data_directory, output_file)
        
        if os.path.exists(buffered_output):
            # Rename to temp
            temp_path = buffered_output.replace('.tif', '_buffered.tif')
            os.rename(buffered_output, temp_path)
            
            # Crop to original
            crop_to_original(
                temp_path, 
                buffered_output, 
                original_bounds, 
                buffer_pixels
            )
            
            # Clean up buffered version
            os.remove(temp_path)
```

4. **Error handling**: Wrap buffer operations in try-except with informative messages

#### 2.2.2 `main()`

**Changes Required**:

1. **Read buffer configuration**:
```python
run_control_sheet = myLibrary.datasheets(name="omniscape_RunControl")
max_jobs = 1
buffer_pixels = 0

if not run_control_sheet.empty:
    if 'MaximumJobs' in run_control_sheet.columns:
        max_jobs = int(run_control_sheet['MaximumJobs'].iloc[0])
    if 'BufferPixels' in run_control_sheet.columns:
        buffer_pixels = int(run_control_sheet['BufferPixels'].iloc[0])
```

2. **Pass buffer_pixels to scenario processing**:
```python
# Sequential
results = [run_scenario(sid, myLibrary, e, buffer_pixels) 
           for sid in scenarioIds]

# Parallel
process_func = partial(
    run_scenario, 
    myLibrary=myLibrary, 
    e=e, 
    buffer_pixels=buffer_pixels
)
results = pool.map(process_func, scenarioIds)
```

---

## 3. Configuration Schema Changes

### 3.1 Package.xml Modifications

Add `BufferPixels` column to `RunControl` datasheet:

```xml
<datasheet name="RunControl" displayName="Run Control">
  <column name="RunControlID" dataType="Integer" isPrimary="True"/>
  <column name="MaximumJobs" dataType="Integer" defaultValue="1" 
          validationType="WholeNumber" validationCondition="GreaterEqual" 
          validationValue="1" displayName="Maximum Jobs"
          description="Number of scenarios to process in parallel"/>
  <column name="BufferPixels" dataType="Integer" defaultValue="0"
          validationType="WholeNumber" validationCondition="GreaterEqual"
          validationValue="0" displayName="Buffer Pixels"
          description="Number of pixels to buffer around extent to reduce edge effects (0 = no buffer)"/>
</datasheet>
```

### 3.2 UI Layout Additions

Add buffer configuration to run control layout:

```xml
<item name="BufferPixels" displayName="Buffer Pixels" 
      tooltip="Number of pixels to add around the study area to reduce edge effects. Higher values reduce edge bias but increase computation time. Recommended: 50-200 pixels for most analyses."/>
```

---

## 4. Implementation Details

### 4.1 Dependencies

**New Python Dependencies**:
- `rasterio` (already in environment for raster I/O)
- `numpy` (already in environment for array operations)

**No additional packages required** - all functionality uses existing dependencies.

### 4.2 File Naming Conventions

| File Type | Naming Pattern | Location | Persistence |
|-----------|----------------|----------|-------------|
| Buffered Resistance | `buffered_resistance.tif` | `Scenario-{id}/` | Kept for debugging |
| Buffered Source | `buffered_source.tif` | `Scenario-{id}/` | Kept for debugging |
| Buffered Outputs (temp) | `{output}_buffered.tif` | `Scenario-{id}/` | Deleted after cropping |
| Final Outputs | Original names | `Scenario-{id}/` | Saved to SyncroSim |

### 4.3 Memory Considerations

**Memory Usage Formula**:
```
BufferedSize = (Width + 2*Buffer) * (Height + 2*Buffer) * BytesPerPixel
```

**Example** (1000x1000 pixel raster, Float32, 200-pixel buffer):
- Original: 1000 × 1000 × 4 bytes = 4 MB
- Buffered: 1400 × 1400 × 4 bytes = 7.84 MB
- Additional memory: ~4 MB per raster

**Recommendations**:
- For large rasters (>10,000 × 10,000), limit buffer to 100-200 pixels
- Monitor memory when using parallel processing with buffers
- Consider tiling very large analyses rather than using extreme buffers

### 4.4 Performance Impact

| Buffer Size | Processing Overhead | Output Size Increase |
|-------------|---------------------|----------------------|
| 0 pixels | 0% (baseline) | 0% |
| 50 pixels | +2-5% | +10-20% intermediate |
| 100 pixels | +5-10% | +20-40% intermediate |
| 200 pixels | +10-20% | +40-80% intermediate |

**Notes**:
- Overhead includes buffering, Julia processing, and cropping
- Julia processing time increases with total pixel count
- Final outputs are original size (no increase)

### 4.5 Edge Cases and Error Handling

#### 4.5.1 Invalid Buffer Sizes

```python
if buffer_pixels < 0:
    raise ValueError(f"Buffer pixels must be non-negative, got {buffer_pixels}")
```

#### 4.5.2 Buffer Larger Than Raster

**Issue**: Buffer pixels > raster dimensions could cause issues

**Solution**: Add validation check:
```python
with rasterio.open(input_path) as src:
    if buffer_pixels > min(src.height, src.width):
        warnings.warn(
            f"Buffer ({buffer_pixels} pixels) is larger than minimum "
            f"raster dimension ({min(src.height, src.width)} pixels). "
            f"This may cause artifacts."
        )
```

#### 4.5.3 Missing Input Files

Existing validation in omniscapeTransformer.py already handles this before buffering occurs.

#### 4.5.4 Disk Space

**Issue**: Buffered intermediates require additional disk space

**Solution**: Document space requirements and clean up temp files aggressively

#### 4.5.5 CRS Mismatches

Existing validation ensures resistance and source have matching CRS. Buffer operations preserve CRS from input.

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Test Suite**: `test_buffer_functions.py`

```python
def test_buffer_raster_basic():
    """Test basic buffering with simple test raster"""
    # Create 10x10 test raster
    # Buffer by 5 pixels
    # Assert output is 20x20
    # Assert edge values replicated correctly

def test_buffer_raster_nodata():
    """Test buffering handles nodata correctly"""
    # Create raster with nodata values at edges
    # Verify nodata preserved in buffered output

def test_buffer_raster_transform():
    """Test affine transform updated correctly"""
    # Buffer raster
    # Verify geographic bounds expanded correctly
    # Verify pixel size unchanged

def test_crop_to_original():
    """Test cropping returns exact original extent"""
    # Buffer a raster
    # Crop back
    # Assert output identical to input

def test_buffer_zero():
    """Test buffer_pixels=0 is no-op"""
    # Should not create buffered files
    # Should use original paths

def test_buffer_preserves_metadata():
    """Test CRS, nodata, datatype preserved"""
    # Buffer various raster types
    # Verify metadata unchanged
```

### 5.2 Integration Tests

**Test Library**: Create `omniscape_buffer_test.ssim` library

**Scenarios**:

1. **No Buffer (Baseline)**:
   - BufferPixels = 0
   - Verify outputs unchanged from current behavior
   
2. **Small Buffer**:
   - BufferPixels = 50
   - Compare outputs to baseline
   - Verify edge effects reduced
   
3. **Large Buffer**:
   - BufferPixels = 200
   - Verify no crashes with large buffer
   - Check performance acceptable
   
4. **Multi-scenario with Buffer**:
   - Run 3 scenarios with BufferPixels = 100
   - MaximumJobs = 2
   - Verify parallel execution + buffering works

### 5.3 Validation Criteria

**Correctness**:
- [ ] Buffered rasters have correct dimensions: `(H+2B) × (W+2B)`
- [ ] Affine transforms updated correctly
- [ ] Cropped outputs match original extent exactly
- [ ] Edge pixel values replicated appropriately
- [ ] CRS and metadata preserved

**Performance**:
- [ ] Buffer=0 has no measurable overhead
- [ ] Buffer=100 completes within 20% of baseline time
- [ ] Memory usage within expected bounds

**Robustness**:
- [ ] Handles various raster sizes (100×100 to 10,000×10,000)
- [ ] Works with integer and float rasters
- [ ] Handles nodata values correctly
- [ ] Appropriate error messages for invalid inputs

---

## 6. User Documentation

### 6.1 Help Text

**Buffer Pixels Setting**:

```
Buffer Pixels: Number of pixels to add around the study area to reduce edge effects.

Edge effects occur when connectivity flows reach the boundary of your study area, 
artificially constraining movement patterns. Adding a buffer extends the analysis 
area temporarily, allowing flows to develop naturally before reaching the true edge.

Recommended values:
- 0: No buffer (default) - fastest, but may have edge artifacts
- 50-100: Light buffering - good for most analyses, minimal performance impact
- 100-200: Moderate buffering - recommended for analyses where edges are important
- 200-500: Heavy buffering - for large-scale analyses or when edge accuracy is critical

Note: Buffer increases processing time and memory usage. The buffered area is 
automatically removed from outputs, so final results match your original study extent.

Performance impact: Each 100 pixels of buffer adds approximately 5-10% to processing time.
```

### 6.2 Tutorial Example

**Title**: "Reducing Edge Effects with Buffers"

**Steps**:
1. Open Omniscape library
2. Navigate to Run Control datasheet
3. Set Buffer Pixels to 100
4. Run scenario
5. Compare results:
   - View `buffered_resistance.tif` to see expanded extent
   - View output to verify matches original extent
   - Compare edge pixel values with/without buffer

### 6.3 Best Practices Guide

**Choosing Buffer Size**:

| Study Area Size | Recommended Buffer |
|----------------|-------------------|
| < 500 × 500 pixels | 50-100 pixels |
| 500-2000 × 2000 | 100-200 pixels |
| 2000-5000 × 5000 | 200-300 pixels |
| > 5000 × 5000 | 200-500 pixels |

**When to Use Buffers**:
- ✅ When study area edges cross known corridors
- ✅ When analyzing connectivity near boundaries
- ✅ When comparing connectivity across different extents
- ✅ For publication-quality analyses

**When Buffers May Not Be Needed**:
- ❌ Study area bounded by natural barriers (ocean, desert)
- ❌ Exploratory analyses where speed is priority
- ❌ When computational resources are limited

---

## 7. Implementation Checklist

### Phase 1: Core Functionality
- [ ] Implement `buffer_raster()` function
- [ ] Implement `crop_to_original()` function
- [ ] Add unit tests for both functions
- [ ] Test with various raster types and sizes

### Phase 2: Integration
- [ ] Modify `run_scenario()` to handle buffers
- [ ] Modify `main()` to read buffer configuration
- [ ] Update package.xml with BufferPixels column
- [ ] Add validation and error handling

### Phase 3: Testing
- [ ] Create integration test library
- [ ] Run performance benchmarks
- [ ] Test parallel processing + buffers
- [ ] Verify backward compatibility

### Phase 4: Documentation
- [ ] Update user guide with buffer documentation
- [ ] Create tutorial example
- [ ] Add help tooltips to UI
- [ ] Document performance characteristics

### Phase 5: Release
- [ ] Update version number
- [ ] Update CHANGELOG
- [ ] Package and test installer
- [ ] Deploy to users

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Memory overflow with large buffers | Medium | High | Document limits, add validation |
| Performance degradation | Low | Medium | Benchmark, provide guidance |
| Raster corruption | Low | High | Extensive testing, validate outputs |
| CRS transformation errors | Low | High | Use rasterio's robust transforms |

### 8.2 User Adoption Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Users set buffer too large | Medium | Medium | Add warnings, document recommendations |
| Confusion about buffer purpose | Low | Low | Clear documentation, examples |
| Unexpected results near edges | Low | Medium | Explain edge replication behavior |

---

## 9. Future Enhancements

### 9.1 Potential Improvements

1. **Multiple Padding Modes**:
   - Add `BufferMode` setting (edge, reflect, constant)
   - Allow user to choose padding strategy
   - Useful for different types of analyses

2. **Adaptive Buffering**:
   - Automatically calculate buffer based on dispersal distance
   - Use `radius` parameter from Omniscape to determine buffer
   - Formula: `buffer = max(radius * 2, 50)`

3. **Asymmetric Buffers**:
   - Allow different buffer sizes per edge
   - Useful when only certain edges need buffering
   - Parameters: `BufferTop`, `BufferBottom`, `BufferLeft`, `BufferRight`

4. **Buffer Visualization**:
   - Display buffer extent in map viewer
   - Show buffered vs. original extent overlay
   - Help users understand impact

5. **Smart Buffer Recommendations**:
   - Analyze raster and suggest optimal buffer
   - Consider study area size, pixel resolution, dispersal distance
   - Display recommendation in UI

### 9.2 Integration with Other Features

- **Tiling**: Combine buffering with tile-based processing for very large rasters
- **Multiprocessing**: Optimize buffer operations for parallel execution
- **Caching**: Cache buffered rasters across runs if inputs unchanged

---

## 10. Appendix

### 10.1 Code Example: Complete Buffer Workflow

```python
# Pseudo-code showing complete buffer workflow

# 1. User configuration
buffer_pixels = 100  # From RunControl datasheet

# 2. Buffer inputs
if buffer_pixels > 0:
    original_bounds = buffer_raster(
        'resistance.tif',
        'buffered_resistance.tif',
        buffer_pixels
    )
    buffer_raster(
        'source.tif',
        'buffered_source.tif', 
        buffer_pixels
    )

# 3. Run Omniscape on buffered inputs
run_omniscape(
    resistance='buffered_resistance.tif',
    source='buffered_source.tif',
    output_dir='outputs/'
)

# 4. Crop outputs back to original
if buffer_pixels > 0:
    for output in outputs:
        crop_to_original(
            f'outputs/{output}',
            f'cropped/{output}',
            original_bounds,
            buffer_pixels
        )

# 5. Save cropped outputs to SyncroSim
save_to_syncrosim('cropped/normalized_cum_currmap.tif')
```

### 10.2 Mathematical Formulation

**Affine Transform for Buffered Raster**:

Given original transform `T_orig = [a, b, c, d, e, f]` where:
- `a` = pixel width
- `e` = pixel height (negative)
- `c` = left coordinate
- `f` = top coordinate

Buffered transform:
```
T_buffered = [
    a,  # pixel width unchanged
    b,  # rotation unchanged (usually 0)
    c - (buffer_pixels * a),  # shift left
    d,  # rotation unchanged (usually 0)
    e,  # pixel height unchanged (negative)
    f - (buffer_pixels * e)   # shift up (note: e is negative)
]
```

### 10.3 References

- **Omniscape.jl Documentation**: https://github.com/Circuitscape/Omniscape.jl
- **SyncroSim Developer Guide**: https://docs.syncrosim.com/
- **Rasterio Documentation**: https://rasterio.readthedocs.io/
- **Circuit Theory in Landscape Ecology**: McRae et al. (2008)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-11 | Alex | Initial specification |

---

## Approval

**Technical Review**: ___________________ Date: ___________

**Product Owner**: ___________________ Date: ___________

**QA Sign-off**: ___________________ Date: ___________
