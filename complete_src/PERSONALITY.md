# HRRR Personality Composites - Complete Guide

Custom weather indices designed for specific personalities and their unique weather interests. Each composite combines multiple meteorological parameters to highlight conditions that matter most to different types of weather enthusiasts.

---

## 🌟 Overview

Personality composites are meteorologically sound but personality-driven composite indices that highlight specific weather phenomena in a fun, memorable way. These are **real calculations** based on solid meteorological science - the personality names just add character to serious weather analysis!

### Philosophy
- **Science first** - All composites use established meteorological relationships
- **Personality driven** - Each composite reflects its namesake's weather interests and "vibe"  
- **Practical application** - Designed for real-world weather analysis and forecasting
- **Fun factor** - Memorable names and interpretations make weather analysis engaging

---

## 🌵 Available Personality Composites

### 🌵 Seqouigrove Weird-West Composite (SW²C)
**Field name:** `seqouigrove_weird_west`

Highlights weird moisture-charged events in normally dry western US regions. Perfect for spotting those rare moments when the desert gets juicy.

**Formula:**
```
SW²C = (PWAT_anomaly) + (CAPE_sfc / 250) + (0.1 × T_anom_500) + (0.05 × |VORT_500|) - (LCL_height / 1000)
```

**What it shows:**
- Unusual moisture surges in dry regions
- Cold-core systems bringing unexpected precipitation
- Atmospheric vorticity patterns in arid areas
- Events that make desert dwellers look up and say "that's not normal"

**Components:**
- **Moisture pop** (PWAT anomaly): Departure from typical 12mm western US precipitable water
- **Desert juice** (CAPE/250): Surface CAPE normalized by 250 J/kg (rewards rare southwestern CAPE spikes)  
- **Cold-core kick** (T500 anomaly × 0.1): 500mb temperature departure from 265K (closed lows over Reno/Vegas)
- **Vorticity sauce** (|ζ500| × 0.05): 500mb absolute vorticity (captures spinning features)
- **Dry-air tax** (LCL/1000): Penalty for high lifted condensation levels (bone-dry setups)

**How to read it:**
- **High values (6-10+):** Peak weirdness - something genuinely unusual is happening
- **Moderate values (2-5):** Interesting but not headline-worthy
- **Low values (<2):** Typical dry western conditions
- **Negative values:** Drier than usual, even for the desert

**Best used for:** Identifying rare moisture events, unusual storm systems, and atmospheric oddities in typically arid regions.

---

### 🌡️ Seqouiagrove Thermal Range Index (STRI)
**Field name:** `seqouiagrove_thermal_range`

Estimates potential for dramatic daily temperature swings based on current atmospheric conditions. Highlights areas where clear, dry conditions create stunning thermal contrasts.

**What it shows:**
- Regions with potential for large day/night temperature differences
- Areas where dry air and light winds favor big thermal swings
- Conditions that create those spectacular desert-style temperature ranges

**How to read it:**
- **Epic ranges (40+):** Desert-level diurnal potential - expect 30°C+ daily swings
- **Big swings (25-39):** Significant thermal contrast likely
- **Moderate (12-24):** Noticeable daily temperature variation
- **Small (<12):** Stable conditions, minimal daily range

**Best used for:** Identifying regions with potential for dramatic temperature contrasts, planning for thermal stress, understanding local climate variations.

---

### 🎸 Kazoo's Everything-to-Eleven Composite (K-MAX)
**Field name:** `kazoo_maxx`

A pure hype index that rewards big numbers. If the atmosphere is flexing anywhere in your domain, K-MAX will glow. No regional penalties, no subtlety - just raw atmospheric energy.

**What it shows:**
- Areas where multiple severe weather ingredients are simultaneously elevated
- Raw atmospheric firepower without location bias
- Places where the atmosphere is genuinely "turned up to eleven"

**How to read it:**
- **≥12:** All-time banger - widespread significant severe ingredients
- **8-11:** Classic outbreak conditions or high-end severe weather
- **4-7:** Respectable severe thunderstorm day, photogenic storms possible
- **1-3:** Garden-variety storms or isolated activity
- **<1:** Atmospheric snooze-fest

**Best used for:** Quick assessment of overall severe weather potential, identifying days with genuine multi-parameter support for significant weather.

---

### 💀 Destroyer Reality-Check Composite (DRC)
**Field name:** `destroyer_reality_check`

A "no-hype-allowed" score that only lights up when every critical severe weather ingredient is simultaneously strong. Any weak link drags the score toward zero, exposing hollow setups.

**What it shows:**
- Areas where ALL severe weather ingredients are present and strong
- Reality check for overhyped weather situations
- Locations with genuine, well-rounded severe weather support

**How to read it:**
- **≥0.60:** Destroyer-approved chase day - robust everything, legitimate severe setup
- **0.30-0.59:** Respectable severe weather potential but probably not historic
- **<0.30:** Something critical is missing - call out the hype

**Key principle:** Uses multiplicative (not additive) math, so one weak ingredient tanks the whole score. Perfect for reality-checking social media storm hype.

**Best used for:** Validating severe weather forecasts, identifying truly robust setups, countering weather hype with data-driven reality checks.

---

### 💨 Samuel Outflow Propensity Index (S-OPI)
**Field name:** `samuel_outflow_propensity`

A science-first composite that quantifies how likely storms are to go "cold-pool dominant" in the Norman, OK corridor. Based on three key physical drivers of strong outflow.

**What it shows:**
- Potential for downdraft-dominated storm evolution
- Areas where evaporation and cooling will create strong cold pools
- Conditions favoring gust fronts and outflow-dominant storm modes
- Science-based assessment of when storms will "undercut themselves"

**Physical components:**
1. **Downdraft CAPE (DCAPE):** Energy available for downdrafts - larger DCAPE = denser, faster cold pools
2. **Evaporative cooling potential:** Dry surface layers maximize raindrop evaporation and cooling
3. **Mid-level moisture:** Moist troposphere supplies condensate that can fall, evaporate, and accelerate downdrafts

**How to read it:**
- **≥0.40:** Violent outflow - expect 50-70 kt gust fronts, bowing segments likely
- **0.20-0.39:** Healthy cold pool - storms trend outflow-dominant after 1-2 hours
- **0.05-0.19:** Marginal mixed modes - some cells may still undercut themselves  
- **<0.05:** Outflow weakness - sustained supercells possible

**Key principle:** Uses multiplicative math based on Norman spring-summer climatology. One weak ingredient (like very moist surface) crushes the score, matching real-world physics.

**Best used for:** Identifying when "gust fronts will wreck everything," understanding storm mode evolution, predicting transition from supercells to QLCS, operational forecasting in the southern Plains.

---

### 🌊 Mason-Flappity Bayou Buzz Composite (MF-Buzz)
**Field name:** `mason_flappity_bayou_buzz`

A mesoscale diagnostic for synoptic-to-convective "interest" in SW Louisiana, specifically tuned to the Lake Charles–Beaumont corridor. Weighs five independent processes that routinely drive high-impact weather along the NW Gulf Coast.

**What it shows:**
- Column moisture anomaly relative to warm-season climatology
- Convective potential through mixed-layer CAPE
- Storm-scale organization via 0-6km bulk wind shear
- Mesoscale forcing through moisture flux convergence
- Low-level vorticity for enhanced rotation potential

**How to read it:**
- **≥4.0:** Hurricane landfall, >200mm rain event, or classic tornadic supercell day
- **2.5-3.9:** SLGT–ENH calibre severe or scattered flash-flood watch day
- **1.0-2.4:** Ordinary diurnal thunderstorms; spot warnings possible
- **<1.0:** Synoptically benign; seabreeze showers at best

**Key principle:** Additive formula (MF-Buzz = M + J + S + C + V) where each term contributes 0-1 for total range 0-5. Designed to flag environments capable of producing tropical-cyclone impacts, prolific warm-rain flooding, or organized severe convection in the humid subtropical Gulf Coast environment.

**Best used for:** High-impact weather assessment in SW Louisiana, tropical/subtropical convective environments, identifying multi-hazard potential (wind/flood/tornado), operational forecasting for petrochemical corridor emergency management.

---

## 🎯 Usage Examples

### Processing Personality Composites
```bash
# Process all personality composites
python smart_hrrr_processor.py --latest --categories personality

# Process specific personality composite
python smart_hrrr_processor.py 20250603 17 --max-hours 1 --categories personality

# Monitor personality composites in real-time
python smart_hrrr_processor.py --latest --categories personality --check-interval 30
```

### Storm Chasing Applications
```bash
# Reality-check hyped weather setups
python smart_hrrr_processor.py 20250602 17 --categories personality
# Check DRC values - low values expose hollow setups

# Quick severe weather assessment
python smart_hrrr_processor.py --latest --categories personality,severe
# Use K-MAX for initial broad-brush potential assessment
```

### Regional Weather Analysis
```bash
# Check for unusual western US weather patterns
python smart_hrrr_processor.py --latest --categories personality
# Monitor SW²C for rare moisture events in desert regions

# Assess Gulf Coast severe weather potential
python smart_hrrr_processor.py --latest --categories personality,severe
# Use MF-Buzz for multi-hazard assessment
```

---

## 🔧 Technical Implementation

### Data Requirements
All personality composites use standard HRRR fields:
- Temperature and moisture parameters (t2m, d2m, rh2m, pwat)
- Wind and shear components (u10, v10, wind_shear_06km)
- Instability indices (CAPE, CIN, LCL height)
- Upper-air fields (500mb temperature, vorticity)

### Calculation Approach
- **Real-time estimation:** Most composites use single-time HRRR snapshots with intelligent environmental factor analysis
- **Multiplicative vs. Additive:** K-MAX uses additive math (rewards big numbers), DRC uses multiplicative math (exposes weak links)
- **Regional adaptation:** Some indices include location-specific climatological adjustments

### Adding New Personality Composites

**1. Add calculation function to `derived_parameters.py`:**
```python
@staticmethod
def your_composite_name(param1: np.ndarray, param2: np.ndarray, ...) -> np.ndarray:
    """
    Compute Your Composite Name (YCN)
    
    Brief description of what it measures and why it matters.
    
    Args:
        param1: Description (units)
        param2: Description (units)
        
    Returns:
        YCN values (scale and units)
    """
    # Your calculation logic here
    # Always include:
    # 1. Input validation/unit conversion
    # 2. Main calculation
    # 3. NaN masking for invalid data
    # 4. Bounds checking/clipping
    
    result = your_formula_here
    
    # Mask invalid data
    result = np.where(
        (np.isnan(param1)) | (np.isnan(param2)),
        np.nan, result
    )
    
    return result
```

**2. Add configuration to `parameters/personality.json`:**
```json
"your_composite_name": {
  "title": "Your Composite Display Name (YCN)",
  "units": "appropriate units (0-1, index, etc)",
  "cmap": "YourCustomColormap",
  "levels": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
  "extend": "max",
  "category": "personality",
  "derived": true,
  "inputs": ["input_field1", "input_field2", "input_field3"],
  "function": "your_composite_name",
  "description": "Brief description of what it shows and how to interpret values."
}
```

**3. Add custom colormap to `hrrr_processor_refactored.py`:**
```python
# In create_spc_colormaps method around line 116-140
# Your Composite Name - Theme description
your_colors = ['#color1', '#color2', '#color3', '#color4', '#color5', 
               '#color6', '#color7', '#color8', '#color9', '#color10']
colormaps['YourCustomColormap'] = LinearSegmentedColormap.from_list('YourName', your_colors)
```

### Implementation Patterns

**Physics-Based Composites** (like S-OPI):
- Use real meteorological relationships
- Include units and scaling based on climatology
- Reference scientific literature or operational experience
- Multiplicative for critical thresholds

**Hype Indices** (like K-MAX):
- Additive formulas that reward big numbers
- No regional penalties or complex adjustments
- Simple scaling, easy interpretation
- Emphasize extreme values

**Reality-Check Indices** (like DRC):
- Multiplicative to expose weak links
- 0-1 scaling with clear thresholds
- Anti-hype, conservative interpretation
- Emphasize robustness over peak values

**Environmental Indices** (like STRI):
- Include multiple environmental factors
- Consider diurnal, seasonal, or regional effects
- Proxy calculations when direct measurements unavailable
- Clear physical meaning

---

## 🧪 Testing Your Implementation

### Create Test Script
```python
# test_your_composite.py
import numpy as np
from derived_parameters import DerivedParameters

# Test with realistic synthetic data
size = (10, 10)
param1 = np.random.uniform(low, high, size)
param2 = np.random.uniform(low, high, size)

result = DerivedParameters.your_composite_name(param1, param2)
print(f"Range: {result.min():.3f} - {result.max():.3f}")

# Test edge cases (NaN, extremes, etc.)
```

### Integration Test
```bash
cd complete_src
python smart_hrrr_processor.py YYYYMMDD HH --max-hours 0 --categories personality
```

---

## 📊 Interpretation Guidelines

### General Principles
- **Compare relatively:** These indices work best for comparing different areas or times
- **Know the personality:** Each composite reflects its namesake's weather interests and biases
- **Combine with other data:** Use alongside traditional meteorological parameters for complete picture
- **Understand limitations:** Snapshot-based calculations may not capture evolving conditions

### Usage Examples by Application

**Storm Chasing:**
- Use K-MAX for initial broad-brush potential assessment
- Apply DRC for reality-checking hyped setups
- Check STRI for post-storm temperature recovery potential

**Weather Analysis:**
- SW²C for identifying unusual regional weather patterns
- DRC for validating forecast confidence
- STRI for understanding local climate variations

**Operational Forecasting:**
- S-OPI for storm mode evolution predictions
- MF-Buzz for multi-hazard assessment in Gulf Coast
- DRC for confidence assessment in severe weather outlooks

**Climate Research:**
- Track personality composite trends over time
- Identify changing patterns in regional weather characteristics
- Validate model performance against observed unusual events

---

## 🚨 Common Gotchas & Troubleshooting

### Implementation Issues
1. **Function naming**: Must match exactly between `derived_parameters.py` and `personality.json`
2. **Input field names**: Must exist in field registry (check `parameters/` directory)
3. **Units consistency**: HRRR uses Kelvin for temps, Pa for pressure - convert as needed
4. **NaN handling**: Always mask invalid data to prevent crashes
5. **Colormap naming**: Must match exactly between `personality.json` and `hrrr_processor_refactored.py`
6. **Array operations**: Use numpy functions that handle arrays properly
7. **Memory**: Large HRRR grids (1059x1799) - be memory efficient

### Validation Checklist
- [ ] Function added to `derived_parameters.py`
- [ ] Configuration added to `personality.json`  
- [ ] Colormap added to `hrrr_processor_refactored.py`
- [ ] Documentation added to this file
- [ ] Test script runs without errors
- [ ] Integration test produces map output
- [ ] Values are in expected range
- [ ] NaN handling works correctly
- [ ] Edge cases behave appropriately

---

## 🌟 Future Composite Ideas

### Potential Additions
- **Sierra Downslope Rotor Index** - Captures mountain wave/rotor activity
- **Monsoon Moisture Transport** - IVT-based southwestern moisture tracking
- **Cold Pool Composite** - Outflow boundary strength for convective complexes
- **Lake Effect Snow Potential** - Great Lakes snowband setup indicator
- **Tornado Alley Composite** - Plains-specific supercell environment
- **Atmospheric River Strength** - West Coast moisture plume intensity

### Guidelines for New Composites

**Scientific Standards:**
- All calculations must be meteorologically sound
- Use established formulas and physical principles
- Include proper unit conversions and bounds checking
- Add references to scientific literature where applicable

**Personality Standards:**
- Named after weather Twitter personalities (with permission)
- Should capture their "vibe" or specialty area
- Humorous but respectful descriptions
- Include interpretation guide with personality-specific language

**Technical Standards:**
- Follow existing code patterns in the system
- Include comprehensive docstrings
- Add test functions for validation
- Use appropriate visualization (colormap, levels, etc.)

---

*Remember: These are real meteorological calculations that happen to be named in a fun way. The science is solid, the vibe is just bonus!* 🌪️

**Once implementation is complete:** Your personality composite will be available via `--categories personality` in the HRRR processing system and will integrate seamlessly with all processing modes (single hour, model run, live monitoring).