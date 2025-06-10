# HRRR Parameter Fix Context Guide

**Purpose**: This document provides complete context for fixing individual map product issues in the HRRR processing system.

## Quick Start for Parameter Fixes

When you receive a criticism like "*FAIL ship_f01_REFACTORED.png – Formula does not fully match canonical SHIP recipe*", follow this workflow:

### 1. Locate the Parameter Implementation

```bash
# Find the parameter configuration
grep -r "ship" parameters/        # Check config files
grep -r "significant_hail" parameters/  # Check by full name

# Find the function implementation  
find derived_params/ -name "*ship*" -o -name "*hail*"
```

**Key files to check:**
- `parameters/derived.json` - Parameter configuration and metadata
- `derived_params/significant_hail_parameter.py` - Function implementation
- `derived_parameters.py` - Function registration

### 2. Understand Current Implementation

```python
# Check current config in parameters/derived.json
{
  "ship": {
    "title": "Significant Hail Parameter (SHIP)",
    "inputs": ["mucape", "mucin", "lapse_rate_700_500", "wind_shear_06km", "freezing_level"],
    "function": "significant_hail_parameter",
    "description": "Current formula description here"
  }
}
```

```python
# Check current function signature in derived_params/
def significant_hail_parameter(mucape, mucin, lapse_rate, shear, freezing_level):
    # Current implementation
```

### 3. Reference SPC Specifications

**Official SPC SHIP Formula (v1.1):**
```
SHIP = (MUCAPE/1500) × (MU mixing ratio/13.6) × (lapse rate/7°C/km) × 
       (wind shear/20 m/s) × ((freezing level - 500mb temp)/8°C)
```

**Five required components:**
1. **muCAPE**: Most Unstable CAPE (J/kg) ÷ 1500
2. **MU mixing ratio**: Most Unstable mixing ratio (g/kg) ÷ 13.6  
3. **Lapse rate**: 700-500mb lapse rate (°C/km) ÷ 7
4. **Wind shear**: 0-6km bulk shear (m/s) ÷ 20
5. **Temperature term**: (Freezing level height - 500mb temp in height coords) ÷ 8

## System Architecture Overview

### Parameter Definition Flow
```
parameters/derived.json → derived_parameters.py → derived_params/function.py → GRIB processing → PNG output
```

### File Structure
```
/home/ubuntu2/hrrr_com/complete_src/
├── parameters/
│   ├── derived.json          # Derived parameter configs
│   ├── severe.json           # Base severe weather fields  
│   ├── instability.json      # CAPE/CIN/etc configs
│   └── atmospheric.json      # Pressure level fields
├── derived_params/
│   ├── significant_hail_parameter.py  # SHIP implementation
│   ├── supercell_composite_parameter.py  # SCP implementation
│   └── common.py             # Shared utilities
├── derived_parameters.py     # Function registry
├── field_templates.py        # Base field templates
├── hrrr_processor_refactored.py  # Main processing engine
└── field_registry.py         # Field lookup system
```

### Parameter Configuration Schema

```json
{
  "parameter_name": {
    "title": "Display Title",
    "units": "parameter units",
    "cmap": "colormap_name",
    "levels": [0, 1, 2, 4, 6, 8, 10],
    "extend": "max",
    "category": "severe",
    "derived": true,
    "inputs": ["input1", "input2", "input3"],
    "function": "function_name",
    "description": "Formula and explanation"
  }
}
```

### Function Implementation Pattern

```python
def parameter_function(input1: np.ndarray, input2: np.ndarray, 
                      input3: np.ndarray = None) -> np.ndarray:
    """
    Compute [Parameter Name] - Official SPC Recipe
    
    Formula: PARAM = (input1/norm1) × (input2/norm2) × (input3/norm3)
    
    Args:
        input1: Description with units and source
        input2: Description with units and source  
        input3: Optional description [optional]
        
    Returns:
        np.ndarray: Parameter values (≥ 0, typical range 0-10)
        
    References:
        SPC specification document/URL
    """
    
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [input1, input2]):
        raise ValueError("All inputs must be numpy arrays")
    
    # Quality control - log extreme values
    extreme_input1 = np.any(input1 > threshold1)
    if extreme_input1:
        print(f"🔍 {param} outliers detected: input1 > {threshold1}")
    
    # Term calculations with proper normalization
    term1 = input1 / normalization1
    term2 = input2 / normalization2
    term3 = input3 / normalization3 if input3 is not None else 1.0
    
    # Apply caps/floors per SPC spec
    term1 = np.clip(term1, 0, max_value1)
    term2 = np.clip(term2, 0, max_value2)
    
    # Final calculation
    result = term1 * term2 * term3
    
    # Mask invalid/negative values
    valid_mask = (
        np.isfinite(input1) & (input1 >= 0) &
        np.isfinite(input2) & (input2 >= 0)
    )
    
    if input3 is not None:
        valid_mask = valid_mask & np.isfinite(input3)
    
    result = np.where(valid_mask & (result >= 0), result, 0.0)
    
    return result
```

## Common Fix Types

### 1. Formula Corrections
**Issue**: Missing terms, wrong normalization, incorrect caps
**Files to modify**: `derived_params/function_name.py`
**Fix approach**: 
- Look up SPC specification
- Implement each term with correct normalization
- Add proper caps/floors
- Update function docstring with complete formula

### 2. Input Field Mapping
**Issue**: Wrong input fields or missing required fields
**Files to modify**: `parameters/derived.json`
**Fix approach**:
- Check `parameters/` configs for available fields
- Map SPC formula requirements to HRRR field names
- Add missing fields to inputs array
- Update function signature to match

### 3. Metadata Updates  
**Issue**: Description doesn't show complete formula
**Files to modify**: `parameters/derived.json`
**Fix approach**:
- Update "description" field with complete SPC formula
- Show each term with normalization values
- Include caps/floors if applicable
- Reference SPC specification version

### 4. Missing Input Fields
**Issue**: Required field not available in HRRR
**Files to check**: 
- `parameters/instability.json` - CAPE/CIN fields
- `parameters/atmospheric.json` - Pressure level fields  
- `parameters/surface.json` - Surface fields
- `field_templates.py` - Template definitions
**Fix approach**:
- Add field definition if missing
- Use appropriate GRIB access pattern
- Add to field templates if needed

## Testing Your Fixes

### 1. Function-Level Testing
```python
# Create test_param_fix.py
import numpy as np
from derived_params.your_function import your_function

# Test with realistic values
input1 = np.full((10, 10), typical_value1)
input2 = np.full((10, 10), typical_value2)

result = your_function(input1, input2)
print(f"Result range: {np.nanmin(result):.2f} to {np.nanmax(result):.2f}")
print(f"Expected range: 0-10 typical for most SPC parameters")
```

### 2. Integration Testing
```bash
# Test single parameter processing
python3 process_single_hour.py 2025031421 0 --fields your_param --use-local-grib

# Check for errors in output
tail -20 logs/processing_*.log
```

### 3. Formula Verification
```python
# Manual calculation check
expected = (input1_val/norm1) * (input2_val/norm2) * (input3_val/norm3)
actual = your_function(test_arrays)[0,0]
print(f"Expected: {expected:.2f}, Actual: {actual:.2f}")
```

## Available Input Fields

### Instability Fields (parameters/instability.json)
- `sbcape`, `sbcin` - Surface-based CAPE/CIN
- `mlcape`, `mlcin` - Mixed-layer CAPE/CIN  
- `mucape`, `mucin` - Most-unstable CAPE/CIN
- `lcl_height` - Lifted condensation level

### Wind/Shear Fields (parameters/severe.json)
- `srh_01km`, `srh_03km` - Storm-relative helicity
- `wind_shear_u_06km`, `wind_shear_v_06km` - Wind shear components
- Derived: `wind_shear_06km` - Wind shear magnitude

### Atmospheric Fields (parameters/atmospheric.json)  
- `temp_500`, `temp_700`, `temp_850` - Temperature at pressure levels
- `height_500`, `height_700` - Geopotential heights
- `freezing_level` - Height of 0°C isotherm

### Surface Fields (parameters/surface.json)
- `t2m`, `d2m` - 2m temperature/dewpoint
- `surface_pressure` - Surface pressure
- `u10`, `v10` - 10m wind components

## Debugging Tools

### 1. Enable Verbose Logging
```python
# Add to function for debugging
print(f"🔍 {param_name} inputs: {[np.nanmax(x) for x in inputs]}")
print(f"🔍 {param_name} terms: term1={np.nanmax(term1):.2f}, term2={np.nanmax(term2):.2f}")
print(f"🔍 {param_name} result: max={np.nanmax(result):.2f}, mean={np.nanmean(result):.2f}")
```

### 2. Field Availability Check
```python
# In derived_parameters.py, add diagnostic
if param_name == 'your_param':
    print(f"🔎 Available inputs: {list(input_data.keys())}")
    for field, data in input_data.items():
        print(f"   {field}: {data.shape}, range {np.nanmin(data):.1f} to {np.nanmax(data):.1f}")
```

### 3. Configuration Validation
```bash
# Check parameter is properly registered
python3 -c "from field_registry import FieldRegistry; r=FieldRegistry(); print(r.get_field('your_param'))"
```

## Example Fix: SHIP Implementation

### Current Issue Analysis
```bash
# 1. Find current implementation
grep -A 10 '"ship"' parameters/derived.json
cat derived_params/significant_hail_parameter.py
```

### SPC SHIP v1.1 Requirements
1. **muCAPE term**: MUCAPE (J/kg) ÷ 1500, cap at 1.0
2. **MU mixing ratio term**: MU mixing ratio (g/kg) ÷ 13.6, cap at 1.0  
3. **Lapse rate term**: 700-500mb lapse rate (°C/km) ÷ 7, cap at 1.0
4. **Wind shear term**: 0-6km bulk shear (m/s) ÷ 20, cap at 1.0
5. **Temperature term**: (Freezing level - 500mb temp height) ÷ 8°C, cap at 1.0

### Implementation Fix
```python
def significant_hail_parameter(mucape: np.ndarray, mu_mixing_ratio: np.ndarray,
                              lapse_rate_700_500: np.ndarray, wind_shear_06km: np.ndarray, 
                              freezing_level: np.ndarray, temp_500: np.ndarray) -> np.ndarray:
    """
    Official SPC SHIP v1.1 - All five required terms
    
    SHIP = (muCAPE/1500) × (MU_mr/13.6) × (lapse/7) × (shear/20) × ((frz_lvl-T500_hgt)/8)
    """
    
    # Term 1: muCAPE
    cape_term = np.clip(mucape / 1500.0, 0, 1.0)
    
    # Term 2: MU mixing ratio  
    mr_term = np.clip(mu_mixing_ratio / 13.6, 0, 1.0)
    
    # Term 3: Lapse rate
    lapse_term = np.clip(lapse_rate_700_500 / 7.0, 0, 1.0)
    
    # Term 4: Wind shear
    shear_term = np.clip(wind_shear_06km / 20.0, 0, 1.0) 
    
    # Term 5: Temperature (convert temp to height coords)
    temp_height = pressure_to_height(500, temp_500)  # Implementation needed
    temp_term = np.clip((freezing_level - temp_height) / 8.0, 0, 1.0)
    
    # Final SHIP calculation
    ship = cape_term * mr_term * lapse_term * shear_term * temp_term
    
    return ship
```

## Critical Success Factors

1. **Follow SPC specifications exactly** - Don't improvise formulas
2. **Implement ALL required terms** - Missing terms invalidate the parameter  
3. **Use correct normalizations** - These are specified in SPC documentation
4. **Apply proper caps/floors** - Usually 0-1 for each term, 0+ for final result
5. **Update metadata completely** - Show full formula in description
6. **Test with realistic data** - Verify output ranges make sense
7. **Check input field availability** - Ensure all required fields exist in HRRR

## When You Need Help

**Missing HRRR field**: Check if alternative field exists or needs to be derived  
**Unclear SPC specification**: Reference original SPC documentation or ask for clarification  
**Complex coordinate transforms**: May need geometric height conversions or interpolation  
**Performance issues**: Use vectorized numpy operations, avoid loops

This context should give you everything needed to systematically fix parameter implementations. Focus on one parameter at a time and verify each component matches the official SPC specification.

---

## Recent Fixes & Search Patterns for Future LLMs

### Common Parameter Issues & Quick Fixes

#### 1. Formula Compliance Issues
**Pattern**: "*FAIL param_f01_REFACTORED.png – Formula does not fully match canonical recipe*"

**Quick search commands:**
```bash
# Find parameter by short name
grep -n "param_name" parameters/derived.json

# Find implementation file
find derived_params/ -name "*param*" -o -name "*keyword*"

# Check function signature
grep -A 5 "def.*param" derived_params/*.py
```

**Recent examples:**
- **SHIP**: Fixed to complete SPC v1.1 formula (all 5 terms) - see `derived_params/significant_hail_parameter.py`
- **SCP**: Fixed over-scaling and made SPC-compliant - see `derived_params/supercell_composite_parameter.py`

#### 2. Saturation/Scaling Issues
**Pattern**: "*WARN param exhibits broad area of saturated max values ('red sea')*"

**Solution approaches:**
- **Anti-saturation damping**: Exponential compression for extreme values (see EHI fix)
- **Extended color scales**: Increase upper range to accommodate peaks
- **Threshold adjustments**: Modify normalization factors

**Recent example:**
- **EHI**: Added damping for |EHI| > 5 to prevent red sea effect - see `derived_params/energy_helicity_index.py`

#### 3. Metadata Display Issues
**Pattern**: "*metadata panel showing long code snippet instead of clean equation*"

**Quick fix:**
```bash
# Find and update description in parameters/derived.json
grep -A 10 "param_name" parameters/derived.json
# Replace description with clean mathematical formula
```

**Recent example:**
- **Lapse Rate 0-3km**: Changed from verbose description to clean formula `LR 0–3 km = (T_surface – T_3 km AGL) / 3 km`

### Advanced Search Techniques

#### Finding Parameters by Symptom
```bash
# Find all EHI-related parameters
grep -r "energy.*helicity\|ehi" parameters/ derived_params/

# Find parameters with specific inputs
grep -r "srh_03km\|effective_srh" parameters/

# Find damping implementations
grep -r "damp\|clip\|saturate" derived_params/

# Find color scale definitions
grep -A 3 "levels.*\[" parameters/*.json
```

#### Finding Related Parameters
```bash
# Parameters using similar inputs
grep -r "mucape.*srh" parameters/

# Parameters in same category
jq '.[] | select(.category=="severe") | keys' parameters/derived.json

# Parameters with similar formulas
grep -r "cape.*shear" derived_params/
```

### Parameter Implementation Patterns

#### Standard Severe Weather Parameter Structure
```python
def parameter_name(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """
    Brief description with formula
    
    Formula: PARAM = (input1/norm1) × (input2/norm2) [with optional damping]
    """
    # Quality control flags
    extreme_input1 = np.any(input1 > threshold)
    
    # Term calculations with normalization
    term1 = np.clip(input1 / norm1, 0, cap1)
    term2 = np.clip(input2 / norm2, 0, cap2)
    
    # Optional: Anti-saturation damping
    if damping_needed:
        result_raw = term1 * term2
        result = apply_damping(result_raw, threshold)
    else:
        result = term1 * term2
    
    # Validation and masking
    valid_mask = np.isfinite(input1) & np.isfinite(input2)
    result = np.where(valid_mask, result, 0.0)
    
    return result
```

#### Anti-Saturation Damping Template
```python
# For parameters that hit extreme values causing "red sea" effect
def apply_exponential_damping(values, threshold=5.0):
    """Apply exponential damping above threshold to prevent saturation"""
    abs_vals = np.abs(values)
    damped = np.where(
        abs_vals > threshold,
        threshold + np.log(abs_vals / threshold),
        abs_vals
    )
    return np.sign(values) * damped
```

### Color Scale Guidelines

#### Common Ranges by Parameter Type
- **EHI**: `[-2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]`
- **STP/SCP**: `[0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]`  
- **SHIP**: `[0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]`
- **Lapse Rates**: `[3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0]`

#### When to Extend Scales
- **Peak values exceed max level** → Add higher levels
- **Poor discrimination in key range** → Add intermediate levels  
- **Red sea saturation** → Consider damping + extended scale

### Field Dependencies Quick Reference

#### Instability Fields (`parameters/instability.json`)
- `sbcape`, `mlcape`, `mucape` - Various CAPE types
- `sbcin`, `mlcin`, `mucin` - Various CIN types
- `lcl_height` - Lifted condensation level

#### Severe Weather Fields (`parameters/severe.json`)  
- `srh_01km`, `srh_03km` - Storm-relative helicity
- `wind_shear_u_06km`, `wind_shear_v_06km` - Shear components

#### Derived Fields (`parameters/derived.json`)
- `wind_shear_06km` - Derived shear magnitude
- `effective_srh`, `effective_shear` - Effective layer parameters
- `lapse_rate_700_500`, `lapse_rate_03km` - Lapse rate variants

#### Atmospheric Fields (`parameters/atmospheric.json`)
- `temp_500`, `temp_700` - Pressure level temperatures
- `height_500`, `height_700` - Geopotential heights  
- `freezing_level` - 0°C isotherm height

### Testing Your Fixes

Always create a simple test script to validate changes:
```python
#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append('/home/ubuntu2/hrrr_com/complete_src')
from derived_params.your_parameter import your_function

# Test with realistic values
def test_parameter():
    input1 = np.array([moderate_value, extreme_value])
    input2 = np.array([moderate_value, extreme_value])
    
    result = your_function(input1, input2)
    
    print(f"Moderate case: {result[0]:.2f}")
    print(f"Extreme case: {result[1]:.2f}")
    print(f"Within expected range: {result.min():.1f} to {result.max():.1f}")

if __name__ == "__main__": test_parameter()
```

### Common Git Workflow
```bash
# Check current status
git status && git diff

# Stage and commit fixes
git add file1 file2
git commit -m "Fix parameter: brief description

- Key change 1
- Key change 2  
- Resolves specific issue

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

This extended context provides patterns and examples from recent successful fixes to help future parameter debugging sessions.