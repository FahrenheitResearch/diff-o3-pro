# Missing Products Analysis - F02 20250607 18z

**Status:** 76/95 products generated (80% success rate)  
**Missing:** 19 products across 5 categories

## 🚨 Missing Products by Category

### 1. UPPER AIR CATEGORY (9/10 missing) ⚠️ **CRITICAL**
**Location:** `parameters/derived.json` and `parameters/surface.json`

| Field Name | Title | Configuration File | Notes |
|------------|-------|-------------------|-------|
| `dewpoint_850` | 850mb Dewpoint Temperature | `parameters/derived.json` | Pressure level access |
| `freezing_level` | Freezing Level Height | `parameters/derived.json` | Derived parameter |
| `height_700` | 700mb Geopotential Height | `parameters/derived.json` | Pressure level access |
| `lapse_rate_700_500` | 700-500mb Lapse Rate | `parameters/derived.json` | Multi-level derived |
| `rh_500` | 500mb Relative Humidity | `parameters/derived.json` | Pressure level access |
| `temp_500` | 500mb Temperature | `parameters/derived.json` | Pressure level access |
| `temp_700` | 700mb Temperature | `parameters/derived.json` | Pressure level access |
| `temp_850` | 850mb Temperature | `parameters/derived.json` | Pressure level access |
| `u_500` | 500mb U Wind Component | `parameters/surface.json` | Pressure level access |
| `v_500` | 500mb V Wind Component | `parameters/surface.json` | Pressure level access |

**Investigation Priority:** HIGH - Entire category failing suggests pressure level access issues

---

### 2. BACKUP CATEGORY (4/5 missing)
**Location:** `parameters/derived.json`

| Field Name | Title | Configuration File | Notes |
|------------|-------|-------------------|-------|
| `mlcape_backup` | Mixed-Layer CAPE (Backup) | `parameters/derived.json` | Backup instability param |
| `mlcin_backup` | Mixed-Layer CIN (Backup) | `parameters/derived.json` | Backup instability param |
| `mucape_backup` | Most-Unstable CAPE (Backup) | `parameters/derived.json` | Backup instability param |
| `sbcape_backup` | Surface-Based CAPE (Backup) | `parameters/derived.json` | Backup instability param |

**Investigation Priority:** MEDIUM - Backup fields, may have template issues

---

### 3. PERSONALITY CATEGORY (3/6 missing)
**Location:** `parameters/personality.json`

| Field Name | Title | Configuration File | Notes |
|------------|-------|-------------------|-------|
| `destroyer_reality_check` | Destroyer Reality-Check Composite (DRC) | `parameters/personality.json` | Custom composite param |
| `samuel_outflow_propensity` | Samuel Outflow Propensity Index (S-OPI) | `parameters/personality.json` | Custom composite param |
| `seqouigrove_weird_west` | Seqouigrove Weird-West Composite (SW²C) | `parameters/personality.json` | Custom composite param |

**Investigation Priority:** MEDIUM - Custom composites, may have calculation issues

---

### 4. HEAT CATEGORY (1/4 missing)
**Location:** `parameters/derived.json`

| Field Name | Title | Configuration File | Notes |
|------------|-------|-------------------|-------|
| `mixing_ratio_2m` | 2m Mixing Ratio | `parameters/derived.json` | Surface-level derived param |

**Investigation Priority:** LOW - Single field, likely config issue

---

### 5. UPDRAFT HELICITY CATEGORY (1/5 missing)
**Location:** `parameters/derived.json`

| Field Name | Title | Configuration File | Notes |
|------------|-------|-------------------|-------|
| `uh_2_5` | Updraft Helicity 2–5 km (max 1 h) | `parameters/derived.json` | Layer-specific UH field |

**Investigation Priority:** LOW - Single field, likely layer access issue

---

## 🔧 Files to Investigate

### Configuration Files
- `parameters/derived.json` - Contains most missing field definitions
- `parameters/surface.json` - Contains u_500, v_500 definitions  
- `parameters/personality.json` - Contains missing personality composites

### Processing Code
- `hrrr_processor_refactored.py` - Main processing logic
- `process_all_products.py` - Category-level processing
- `field_registry.py` - Field loading and validation
- `derived_parameters.py` - Derived parameter calculations

### Derived Parameter Functions
- `derived_params/lapse_rate_700_500.py` - 700-500mb lapse rate calculation
- `derived_params/` directory - Contains calculation functions for derived fields

### Log Files to Check
- `outputs/hrrr/20250607/18z/logs/` - Processing logs
- `**/timing_results_*.json` - Timing data per category

---

## 🚀 Investigation Strategy

### 1. **Check Upper Air Category First (Highest Impact)**
```bash
# Check if upper_air category is being processed
grep -r "upper_air" outputs/hrrr/20250607/18z/F02/

# Check pressure level access patterns
grep -A 5 -B 5 "pressure_level" parameters/derived.json
```

### 2. **Check Category Processing**
```bash
# Check which categories were attempted
ls outputs/hrrr/20250607/18z/F02/F02/
```

### 3. **Check Field Configurations**
```bash
# Validate missing field configs
python -c "
from field_registry import FieldRegistry
registry = FieldRegistry()
missing_fields = ['temp_500', 'u_500', 'v_500', 'destroyer_reality_check']
for field in missing_fields:
    config = registry.get_field(field)
    print(f'{field}: {config is not None}')
"
```

### 4. **Check Processing Logs**
Look for error messages related to:
- Pressure level access failures
- GRIB variable not found errors  
- Template resolution failures
- Category processing skips

---

## 📋 Quick Action Items

1. **Verify upper_air category exists and processes**
2. **Check if pressure level access patterns work**
3. **Validate backup field templates resolve correctly**
4. **Test personality composite calculations**
5. **Add error logging for missing categories**

---

*Generated: 2025-06-07*  
*Analysis based on F02 20250607 18z products*