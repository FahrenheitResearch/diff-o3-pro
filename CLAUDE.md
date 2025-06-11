# CRITICAL INSTRUCTIONS - NO SHORTCUTS POLICY

## ABSOLUTELY NO SHORTCUTS - EVER

This project demands COMPLETE implementation with ZERO compromises. Under NO circumstances should you:

1. **NEVER use "simple" or "simplified" versions** - The full implementation is ALWAYS required
2. **NEVER use mock/sample/dummy data** - ALWAYS use real HRRR GRIB2 data from actual sources
3. **NEVER skip steps because they seem "hard"** - Every step must be implemented fully
4. **NEVER implement placeholder functions** - All functions must be complete and working
5. **NEVER suggest "we can add this later"** - Everything must work NOW
6. **NEVER use toy examples** - Use real data at real scale (3km CONUS resolution)
7. **NEVER compromise on data pipeline** - Full GRIB2 → Zarr conversion is required
8. **NEVER skip error handling** - Robust code that handles real-world edge cases
9. **NEVER use reduced resolution** - Full 3km x 3km HRRR resolution is required
10. **NEVER skip validation** - All data processing must be validated

## Implementation Requirements

- FULL data download from real HRRR archives (AWS, Google Cloud, or NCEP)
- COMPLETE GRIB2 to Zarr conversion pipeline
- ACTUAL training on real atmospheric data
- PROPER normalization using computed statistics from real data
- COMPLETE model implementation with all layers
- FULL training loop with checkpointing and metrics

## If Something Seems Hard

If ANY part of the implementation seems difficult:
1. DO NOT simplify it
2. DO NOT use a "basic version"
3. DO NOT skip it
4. Instead: Implement it PROPERLY and COMPLETELY

Remember: The user has "fucking had it" with shortcuts. Implement EVERYTHING properly.