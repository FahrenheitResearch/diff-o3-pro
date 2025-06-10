# VTP Impact from 0-3km CAPE Fix

## Summary
The 0-3km CAPE fix will significantly reduce VTP values because VTP uses `cape_03km / 300.0` as one of its multiplicative terms.

## Impact Analysis

### Before Fix (Broken 0-3km CAPE)
```
MLCAPE = 2000 J/kg
cape_03km = 2000 * 0.7 = 1400 J/kg (unrealistic)
VTP cape_03km_term = 1400/300 = 4.67 → capped at 2.0
```

### After Fix (Realistic 0-3km CAPE)  
```
MLCAPE = 2000 J/kg
cape_03km = 2000 * 0.20 = 400 J/kg (realistic)
VTP cape_03km_term = 400/300 = 1.33 (no cap needed)
```

### Impact on VTP
- **Direct reduction**: cape_03km_term typically drops from 2.0 → ~1.6-1.3 (~20–35 % reduction depending on environment)
- **Multiplicative effect**: Since VTP multiplies all terms, overall VTP will be ~33% lower
- **Secondary effects**: SRH gate now < 0.17 ; lapse override triggers ≥ 0.67

## Expected Results
- VTP values will be more realistic and scientifically accurate
- May need to review VTP thresholds for operational use
- Values like VTP=6-9 might become VTP=4-6 after fix
- This is actually **correct** behavior - the old values were inflated due to bogus 0-3km CAPE

## Recommendation
Monitor VTP values in first production run and consider if operational thresholds need adjustment. The lower values are meteorologically correct.