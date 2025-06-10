# HRRR Derived Parameters Reference

This document provides a comprehensive reference for all derived meteorological parameters calculated from HRRR model output. Each parameter includes its formula, inputs, physical meaning, and operational thresholds.

## Table of Contents

- [Severe Weather Composite Parameters](#severe-weather-composite-parameters)
- [CAPE/CIN Backup Calculations](#capecin-backup-calculations)
- [Advanced Severe Weather Parameters](#advanced-severe-weather-parameters)
- [Supercell Parameters](#supercell-parameters)
- [Basic Meteorological Parameters](#basic-meteorological-parameters)
- [Heat Stress Indices](#heat-stress-indices)
- [Wind Shear Parameters](#wind-shear-parameters)
- [Stability Indices](#stability-indices)
- [Boundary Layer Parameters](#boundary-layer-parameters)
- [Fire Weather Parameters](#fire-weather-parameters)

---

## Severe Weather Composite Parameters

### Supercell Composite Parameter (SCP)

**Formula (Traditional):** `SCP = (MUCAPE/1000) × (SRH/50) × (Shear/20)`

**Enhanced Formula:** `SCP_eff = (MUCAPE/1000) × (Effective_SRH/50) × (Effective_Shear/20)`

**Inputs:**
- `mucape` - Most-Unstable CAPE (J/kg)
- `srh_03km` - 0-3 km Storm Relative Helicity (m²/s²) OR `effective_srh` - Effective SRH (m²/s²)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s) OR `effective_shear` - Effective bulk shear (m/s)

**Physical Meaning:** Combines instability, rotation, and shear to diagnose supercell thunderstorm potential. Enhanced version uses effective layers that account for storm depth and capping inversions.

**Interpretation:**
- SCP > 1: Supercell potential
- SCP > 4: Significant supercell environment  
- SCP > 8: Extreme supercell environment
- Positive values: Right-moving (cyclonic) supercells
- Negative values: Left-moving (anticyclonic) supercells

**Special Logic:**
- CAPE term: Only positive CAPE contributes
- SRH term: Sign preserved (negative for left-movers)
- Shear term: Set to 0 for shear < 10 m/s, capped at 1.0 for shear ≥ 20 m/s
- **Enhanced version uses effective layers** which are more representative of actual storm inflow

**Validation:** Formula consistent with Thompson et al. (2004) and SPC operational usage. Uses virtual temperature for CAPE when available.

---

### Significant Tornado Parameter (STP)

**Formula (Traditional):** `STP = (MLCAPE/1500) × (0-1km SRH/150) × (0-6km Shear/20) × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)`

**Enhanced Formula:** `STP_eff = (MLCAPE/1500) × (Effective_SRH/150) × (Effective_Shear/20) × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)`

**Inputs:**
- `mlcape` - Mixed Layer CAPE (J/kg)
- `mlcin` - Mixed Layer CIN (J/kg, negative values)
- `srh_01km` - 0-1 km Storm Relative Helicity (m²/s²) OR `effective_srh` - Effective SRH (m²/s²)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s) OR `effective_shear` - Effective bulk shear (m/s)
- `mllcl_height` - Mixed Layer LCL height (m AGL)

**Physical Meaning:** Diagnoses potential for significant (EF2+) tornadoes in supercell environments. Based on Thompson et al. (2004) with effective layer enhancements for improved accuracy.

**Interpretation:**
- STP > 1: Heightened significant tornado risk
- STP > 4: Extreme tornado potential  
- STP > 8: Historic outbreak-level environment

**Component Details:**
- **CAPE term:** MLCAPE/1500 (no arbitrary cap - let physics decide)
- **LCL term:** (2000-MLLCL)/1000, clipped: LCL < 1000m → 1.0 (optimal), LCL > 2000m → 0.0 (too high)
- **SRH term:** Effective_SRH/150 or 0-1km SRH/150 (only positive values contribute)
- **Shear term:** Effective_Shear/20 or 0-6km Shear/20, clipped: < 10 m/s → 0, ≥ 20 m/s → 1.0
- **CIN term:** (MLCIN + 200)/150, MLCIN > -50 → 1.0 (uncapped), MLCIN < -200 → 0.0 (strong cap)

**Validation:** Formula follows SPC's operational definition. CIN normalization does not use virtual temperature per SPC practice.

---

### Energy-Helicity Index (EHI)

**Formula (Updated):** `EHI = (CAPE/1600) × (SRH/50)`

**Variants:**
- **0-3 km EHI:** Uses 0-3 km SRH (general supercell potential)
- **0-1 km EHI:** Uses 0-1 km SRH (tornado-specific potential)

**Inputs:**
- `cape` - CAPE (J/kg) - Surface-Based or Mixed-Layer
- `srh_03km` - 0-3 km Storm Relative Helicity (m²/s²) OR
- `srh_01km` - 0-1 km Storm Relative Helicity (m²/s²)

**Physical Meaning:** Represents co-location of instability and low-level rotational potential. Updated normalization based on operational standards.

**Interpretation:**
- EHI > 1: Notable for supercells
- EHI > 2: Significant tornado potential
- EHI_01km > 1: Notable tornado potential (0-1 km version)
- EHI_01km > 2: High tornado potential
- Positive EHI: Cyclonic (right-moving) supercell potential
- Negative EHI: Anticyclonic (left-moving) supercell potential

**Validation:** Updated formula uses proper normalization factors per meteorological literature. 0-1 km EHI more directly linked to tornado potential near surface.

---

## Stability Indices

### Lifted Index (LI)

**Formula:** `LI = T_env(500mb) - T_parcel(500mb)`

**Inputs:**
- `temp_surface` - Surface temperature (K)
- `dewpoint_surface` - Surface dewpoint (K)
- `temp_500mb` - Environmental temperature at 500mb (K)

**Physical Meaning:** Temperature difference between environment and surface parcel lifted to 500mb. Incorporates both low-level moisture and mid-level lapse rates.

**Interpretation:**
- LI > 0: Stable atmosphere
- 0 to -3: Marginal instability
- -4 to -6: Very unstable
- < -6: Extremely unstable

**Validation:** Standard meteorological index, widely used operationally. Negative LI indicates instability.

---

### Showalter Index (SI)

**Formula:** `SI = T_env(500mb) - T_parcel_850→500(500mb)`

**Inputs:**
- `temp_850mb` - Temperature at 850mb (K)
- `dewpoint_850mb` - Dewpoint at 850mb (K)
- `temp_500mb` - Environmental temperature at 500mb (K)

**Physical Meaning:** Similar to LI but uses 850mb parcel instead of surface. Useful for elevated instability when surface is cool.

**Interpretation:**
- SI > 0: Stable
- 0 to -3: Moderately unstable
- < -6: Extremely unstable

**Validation:** Less sensitive to surface conditions than LI. Can highlight elevated instability above cool surface layers.

---

## Advanced Severe Weather Parameters

### Significant Hail Parameter (SHIP)

**Formula:** `SHIP = f(MUCAPE, MUCIN, LapseRate_700-500, BulkShear, FreezingLevel)`

**Inputs:**
- `mucape` - Most-Unstable CAPE (J/kg)
- `mucin` - Most-Unstable CIN (J/kg, negative values)
- `lapse_rate_700_500` - 700-500mb lapse rate (°C/km)
- `bulk_shear_06km` - 0-6km bulk shear (m/s)
- `freezing_level_height` - Freezing level height (m AGL)

**Physical Meaning:** Composite index designed to identify environments favorable for significant hail (≥2" diameter). Combines instability, mid-level lapse rates, shear, and hail growth zone characteristics.

**Interpretation:**
- SHIP > 1: Favorable for significant hail
- SHIP > 4: Extremely high hail potential

**Validation:** Based on research into significant hail environments. SHIP = 1.0 corresponds to median environment for 2"+ hail.

---

### Craven Significant Severe Parameter

**Formula:** `Craven = MLCAPE × 0-6km Bulk Shear`

**Inputs:**
- `mlcape` - Mixed-Layer CAPE (J/kg)
- `bulk_shear_06km` - 0-6km bulk shear magnitude (m/s)

**Physical Meaning:** Simple product of instability and shear, representing CAPE-shear synergy for severe weather.

**Interpretation:**
- > 20,000 m³/s³: Significant severe weather potential
- > 50,000 m³/s³: Very high severe potential

**Validation:** Threshold of 20,000 m³/s³ discriminates significant severe events per Craven et al. (2004).

---

### Vorticity Generation Parameter (VGP)

**Formula:** `VGP ∝ (CAPE × Low-level Shear)`

**Inputs:**
- `cape` - CAPE (J/kg)
- `wind_shear_01km` - 0-1km wind shear magnitude (m/s)

**Physical Meaning:** Estimates rate at which updrafts can generate vertical vorticity from horizontal vorticity through tilting and stretching.

**Interpretation:**
- > 0.2 m/s²: Increased tornado potential
- > 0.5 m/s²: High tornado potential

**Validation:** Based on Rasmussen (2003) vorticity generation research. Flags co-located low-level shear and instability.

---

## Effective Layer Parameters

### Effective Storm-Relative Helicity

**Formula:** SRH computed only within effective inflow layer

**Criteria for Effective Layer:**
- CAPE ≥ 100 J/kg (minimum buoyancy)
- CIN ≥ -250 J/kg (not too much inhibition)
- LCL ≤ 2500 m (reasonable cloud base)

**Physical Meaning:** SRH limited to areas where convection is effectively possible, filtering out capped or unfavorable layers.

**Interpretation:**
- Effective SRH > 150-200 m²/s²: Large for significant tornadoes
- Better discriminates tornadic vs non-tornadic supercells than fixed-layer SRH

**Validation:** Used in SPC mesoanalysis. More representative than fixed 0-3km SRH in capped environments.

---

## Wind Shear Parameters

### 0–6 km Bulk Wind Shear

**Formula:** `Shear = √[(u_6km - u_0)² + (v_6km - v_0)²]`

**Physical Meaning:** Vector difference between winds at 6km AGL and surface. Deep-layer shear relevant for storm organization.

**Interpretation:**
- > ~40 kt (20 m/s): Minimum for supercells
- > 50-60 kt: Very organized storms

**Validation:** Standard operational parameter. ~40 kt threshold widely used for supercell forecasting.

---

### 0–1 km Bulk Wind Shear

**Formula:** `Shear = √[(u_1km - u_0)² + (v_1km - v_0)²]`

**Physical Meaning:** Low-level shear in storm's inflow layer, critical for tornado potential.

**Interpretation:**
- > 15-20 kt (8-10 m/s): Minimum for significant tornadoes
- > 30 kt (15 m/s): High tornado potential

**Validation:** Strong correlation with tornado occurrence when combined with CAPE and SRH.

---

## CAPE/CIN Backup Calculations

*Note: These are backup calculations used when direct HRRR CAPE/CIN fields are unavailable. Primary operations use direct model-calculated CAPE/CIN.*

### Surface-Based CAPE (Backup)

**Formula:** Simplified Bolton (1980) approximation:
```
θe_surface = θ × exp(Lv × r / (Cp × T))
CAPE ≈ Cp × T × ln(θe_parcel / θe_environment)
```

**Inputs:**
- `t2m` - 2m temperature (K)
- `d2m` - 2m dewpoint (K)
- `surface_pressure` - Surface pressure (Pa)

**Physical Meaning:** Estimates buoyant energy available to surface parcels when full profile unavailable.

---

### Surface-Based CIN (Backup)

**Formula:** `CIN = -10.0 × (T - Td) × temp_factor`

**Inputs:**
- `t2m` - 2m temperature (K)
- `d2m` - 2m dewpoint (K)
- `surface_pressure` - Surface pressure (Pa)

**Physical Meaning:** Estimates capping inversion strength from surface dewpoint depression.

**Logic:**
- Strong dewpoint depression → stronger cap
- Hot surface temps (>30°C) → 1.5× multiplier
- Dewpoint depression < 5°C → CIN = 0

---

## Advanced Severe Weather Parameters

### Bulk Richardson Number (BRN)

**Formula:** `BRN = CAPE / (0.5 × Shear²)`

**Inputs:**
- `sbcape` - Surface-Based CAPE (J/kg)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s)

**Physical Meaning:** Compares instability to wind shear, indicating storm organization mode.

**Interpretation:**
- BRN < 10: Extreme shear (storms may struggle)
- BRN 10-45: Optimal balance for supercells
- BRN > 50: Weak shear (pulse/multicell storms)

**Special Logic:**
- Minimum shear of 1 m/s to avoid division by zero
- Capped at 999 for display purposes

---

### Effective Storm Relative Helicity

**Formula:** Effective SRH applied only where criteria met:
```
Effective SRH = SRH × (CAPE ≥ 100) × (CIN ≥ -250) × (LCL ≤ 2500)
```

**Inputs:**
- `srh_03km` - 0-3 km Storm Relative Helicity (m²/s²)
- `mlcape` - Mixed Layer CAPE (J/kg)
- `mlcin` - Mixed Layer CIN (J/kg)
- `lcl_height` - LCL height (m)

**Physical Meaning:** SRH limited to areas where convection is effectively possible.

---

### Craven-Brooks Composite Parameter

**Formula:** `CBC = 0.4 × (CAPE/1000) + 0.4 × (Shear/20) + 0.2 × (SRH/200)`

**Inputs:**
- `sbcape` - Surface-Based CAPE (J/kg)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s)
- `srh_03km` - 0-3 km Storm Relative Helicity (m²/s²)

**Physical Meaning:** Weighted combination of severe weather ingredients.

---

## Supercell Parameters

### Right-Mover Supercell Composite

**Formula:** Enhanced SCP with storm motion considerations (implementation varies)

**Inputs:**
- `mucape` - Most-Unstable CAPE (J/kg)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s)
- `srh_03km` - 0-3 km Storm Relative Helicity (m²/s²)

---

### Supercell Strength Index

**Formula:** Combines CAPE, shear, updraft helicity, and LCL for supercell intensity

**Inputs:**
- `mucape` - Most-Unstable CAPE (J/kg)
- `wind_shear_06km` - 0-6 km bulk wind shear magnitude (m/s)
- `updraft_helicity` - Updraft helicity (m²/s²)
- `lcl_height` - LCL height (m)

---

## Basic Meteorological Parameters

### 10m Wind Speed

**Formula:** `Speed = √(u² + v²)`

**Inputs:**
- `u10` - 10m U wind component (m/s)
- `v10` - 10m V wind component (m/s)

---

### 10m Wind Direction

**Formula:** `Direction = atan2(v, u) × 180/π`

**Inputs:**
- `u10` - 10m U wind component (m/s)
- `v10` - 10m V wind component (m/s)

---

### 2m Mixing Ratio

**Formula:** `MR = 0.622 × es / (P - es)` where `es = 6.112 × exp(17.67×Td/(Td+243.5))`

**Inputs:**
- `d2m` - 2m dewpoint temperature (K)
- `surface_pressure` - Surface pressure (Pa)

---

### Wet Bulb Temperature

**Formula:** Uses MetPy calculation or Stull (2011) approximation fallback

**Inputs:**
- `t2m` - 2m temperature (K)
- `d2m` - 2m dewpoint temperature (K)
- `surface_pressure` - Surface pressure (Pa)

---

## Heat Stress Indices

### WBGT Shade

**Formula:** `WBGT = 0.7 × WB + 0.3 × DB`

**Inputs:**
- `wet_bulb_temp` - Wet bulb temperature (°C)
- `t2m` - 2m temperature (K, converted to °C)

**Physical Meaning:** Heat stress index for shaded conditions (no solar load).

---

### WBGT Estimated Outdoor

**Formula:** `WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB`
where `BG = DB + solar_load - wind_cooling`

**Inputs:**
- `wet_bulb_temp` - Wet bulb temperature (°C)
- `t2m` - 2m temperature (K)
- `wind_speed_10m` - 10m wind speed (m/s)

**Physical Meaning:** Heat stress with estimated solar radiation and wind cooling effects.

---

### WBGT Simplified Outdoor

**Formula:** `WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB`
where `BG = DB + 2°C - wind_cooling`

**Inputs:**
- `wet_bulb_temp` - Wet bulb temperature (°C)
- `t2m` - 2m temperature (K)
- `wind_speed_10m` - 10m wind speed (m/s)

**Physical Meaning:** Simplified outdoor WBGT with constant moderate solar load.

---

## Wind Shear Parameters

### 0-6 km Bulk Wind Shear Magnitude

**Formula:** `Shear = √(u_shear² + v_shear²)`

**Inputs:**
- `wind_shear_u_06km` - 0-6 km U-component bulk shear (m/s)
- `wind_shear_v_06km` - 0-6 km V-component bulk shear (m/s)

---

### 0-1 km Bulk Wind Shear Magnitude

**Formula:** `Shear = √(u_shear² + v_shear²)`

**Inputs:**
- `wind_shear_u_01km` - 0-1 km U-component bulk shear (m/s)
- `wind_shear_v_01km` - 0-1 km V-component bulk shear (m/s)

---

### Shear Vector Magnitude Ratio

**Formula:** `Ratio = |Shear_0-1km| / |Shear_0-6km|`

**Inputs:**
- `wind_shear_u_01km`, `wind_shear_v_01km` - 0-1 km shear components
- `wind_shear_u_06km`, `wind_shear_v_06km` - 0-6 km shear components

**Physical Meaning:** Indicates how much shear is concentrated in low levels vs. deep layer.

---

## Stability Indices

### Cross Totals Index (Simplified)

**Formula:** `CT = Dewpoint_surface - Temperature_surface` (simplified from 850-500mb)

**Inputs:**
- `t2m` - 2m temperature (K)
- `d2m` - 2m dewpoint temperature (K)

**Physical Meaning:** Simplified instability index using surface data only.

---

## Boundary Layer Parameters

### Surface Richardson Number (Simplified)

**Formula:** `Ri = (g/T) × (dT/dz) / (du/dz)²` (estimated from available fields)

**Inputs:**
- `t2m` - 2m temperature (K)
- `wind_shear_01km` - 0-1 km wind shear (m/s)

**Physical Meaning:** Stability parameter indicating turbulent vs. laminar flow potential.

---

### Ventilation Rate

**Formula:** `VR = WindSpeed × PBL_Height` where `WindSpeed = √(u² + v²)`

**Inputs:**
- `u10`, `v10` - 10m wind components (m/s)
- `pbl_height` - Planetary boundary layer height (m)

**Physical Meaning:** Atmosphere's capacity to dilute pollutants through horizontal transport.

---

## Fire Weather Parameters

### Fire Weather Index

**Formula:** Complex function combining temperature, humidity, and wind speed

**Inputs:**
- `t2m` - 2m temperature (K)
- `rh2m` - 2m relative humidity (%)
- `u10`, `v10` - 10m wind components (m/s)

**Physical Meaning:** Composite index for fire weather conditions.

---

### Enhanced Smoke Dispersion Index

**Formula:** Combines vertical wind shear, temperature, boundary layer height, and surface winds

**Inputs:**
- `wind_shear_u_01km`, `wind_shear_v_01km` - Low-level shear components
- `t2m` - 2m temperature (K)
- `pbl_height` - Boundary layer height (m)
- `u10`, `v10` - Surface wind components (m/s)

**Physical Meaning:** Atmospheric capacity for smoke mixing and dispersion.

---

## Special Parameters

### Updraft Helicity > 75 (Tornado Risk)

**Formula:** `Binary = 1 if UH ≥ 75, else 0`

**Inputs:**
- `updraft_helicity` - Updraft helicity values (m²/s²)

**Physical Meaning:** Binary mask indicating areas exceeding tornado-significant UH threshold.

---

### Crosswind Component

**Formula:** `Crosswind = u × sin(θ) + v × cos(θ)` where θ = reference direction

**Inputs:**
- `u10` - 10m U wind component (m/s)
- `v10` - 10m V wind component (m/s)

**Physical Meaning:** Wind component perpendicular to a reference direction (default: North).

---

## Usage Notes

1. **Primary Sources:** Direct HRRR CAPE/CIN fields are preferred over backup calculations
2. **Units:** All energies in J/kg, winds in m/s, heights in meters AGL, temperatures in K or °C as specified
3. **Thresholds:** Operational thresholds based on current SPC/NWS criteria and research literature
4. **Validation:** Parameters follow published meteorological literature and SPC operational definitions
5. **Performance:** Derived parameters computed only when constituent fields available
6. **Virtual Temperature:** CAPE calculations use virtual temperature correction when available for improved accuracy (typically 5-15% increase in moist environments)
7. **CIN Calculation:** CIN does not use virtual temperature per SPC operational practice
8. **Effective Layers:** Prefer effective-layer versions of composite parameters (SCP_eff, STP_eff) over fixed-layer versions for better accuracy in capped environments
9. **Parameter Interdependence:** No single parameter guarantees severe weather - consider combinations (e.g., high CAPE with weak shear may produce only pulse storms)
10. **Layer Definitions:** 
    - Effective inflow base: Lowest level with CAPE ≥ 100 J/kg and CIN ≥ -250 J/kg
    - Effective top: Where CAPE drops below 100 J/kg or CIN exceeds -250 J/kg
11. **Storm Motion:** SRH calculations assume typical supercell motion; adjust interpretation for unusual storm motions

## References

- Bolton, D. (1980): The computation of equivalent potential temperature. *Mon. Wea. Rev.*, **108**, 1046-1053.
- Thompson, R. L., et al. (2004): Close proximity soundings within supercell environments obtained from the Rapid Update Cycle. *Wea. Forecasting*, **19**, 1243-1261.
- Craven, J. P., R. E. Jewell, and H. E. Brooks (2004): Comparison between observed convective cloud-base heights and lifting condensation level for two different lifted parcels. *Wea. Forecasting*, **19**, 511-526.
- Rasmussen, E. N. (2003): Refined supercell and tornado forecast parameters. *Wea. Forecasting*, **18**, 530-535.
- Storm Prediction Center Parameter Definitions: https://www.spc.noaa.gov/sfctest/help/sfcoa.html
- National Weather Service Training: https://www.weather.gov/training/
- MetPy Documentation: https://unidata.github.io/MetPy/
- University of Wyoming Upper Air Soundings: http://weather.uwyo.edu/upperair/calc.html

**Key Validation Sources:**
- SPC Mesoanalysis: https://www.spc.noaa.gov/exper/mesoanalysis/
- NOAA/NWS Forecast Office Training Materials
- Operational meteorology textbooks and peer-reviewed literature
- Field verification studies and tornado/severe weather climatologies

---

*Generated: $(date)*  
*HRRR Processor Version: 2.0*