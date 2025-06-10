from .common import *
from .violent_tornado_parameter import violent_tornado_parameter

def validate_vtp_implementation(mlcape: np.ndarray, mlcin: np.ndarray,
                               lcl_height: np.ndarray, srh_03km: np.ndarray,
                               wind_shear_06km: np.ndarray, cape_03km: np.ndarray,
                               lapse_rate_03km: np.ndarray,
                               case_name: str = "Unknown") -> Dict[str, Any]:
    """
    Validate VTP implementation against expected SPC behavior
    
    Quick validation cases per expert guidance:
    1. Quiet summer HRRR run → VTP ~0 everywhere (maybe <0.5 in humid Gulf)
    2. 3 Mar 2019 20 UTC → VTP max 4-6 in Lee Co., AL corridor  
    3. 27 Apr 2011 18 UTC → VTP ribbon 6-9 in central AL/MS, STP ~4-6
    
    Args:
        All VTP input arrays
        case_name: Description of the case being validated
        
    Returns:
        Dictionary with validation results and diagnostic flags
    """
    
    # Compute VTP
    vtp = violent_tornado_parameter(mlcape, mlcin, lcl_height, srh_03km, 
                                   wind_shear_06km, cape_03km, lapse_rate_03km)
    
    # Basic statistics
    valid_vtp = vtp[vtp > 0]
    total_points = vtp.size
    valid_points = len(valid_vtp)
    
    validation = {
        'case_name': case_name,
        'total_grid_points': int(total_points),
        'valid_vtp_points': int(valid_points),
        'vtp_coverage_pct': float(valid_points / total_points * 100) if total_points > 0 else 0.0
    }
    
    if valid_points > 0:
        validation['vtp_stats'] = {
            'min': float(np.min(valid_vtp)),
            'max': float(np.max(valid_vtp)),
            'mean': float(np.mean(valid_vtp)),
            'median': float(np.median(valid_vtp)),
            'p95': float(np.percentile(valid_vtp, 95)),
            'p99': float(np.percentile(valid_vtp, 99))
        }
        
        # Count threshold exceedances
        validation['threshold_counts'] = {
            'vtp_gt_0p5': int(np.sum(vtp > 0.5)),
            'vtp_gt_1': int(np.sum(vtp > 1.0)),
            'vtp_gt_2': int(np.sum(vtp > 2.0)),
            'vtp_gt_5': int(np.sum(vtp > 5.0)),
            'vtp_gt_10': int(np.sum(vtp > 10.0))
        }
    else:
        validation['vtp_stats'] = {'no_valid_data': True}
        validation['threshold_counts'] = {k: 0 for k in ['vtp_gt_0p5', 'vtp_gt_1', 'vtp_gt_2', 'vtp_gt_5', 'vtp_gt_10']}
    
    # Diagnostic flags based on expected SPC behavior
    validation['flags'] = {}
    
    # Flag 1: Carpet bombing (too much coverage)
    high_coverage = validation['vtp_coverage_pct'] > 20.0  # >20% of domain with VTP>0
    validation['flags']['potential_carpet_bombing'] = high_coverage
    
    # Flag 2: Extreme values (suspect scaling)
    extreme_values = validation['threshold_counts']['vtp_gt_10'] > 0
    validation['flags']['extreme_values_present'] = extreme_values
    
    # Flag 3: Widespread moderate values (oversensitivity)
    widespread_moderate = validation['threshold_counts']['vtp_gt_2'] > (total_points * 0.05)  # >5% with VTP>2
    validation['flags']['widespread_moderate_vtp'] = widespread_moderate
    
    # Input diagnostics
    cape_stats = _get_array_stats(cape_03km, "0-3km CAPE")
    lapse_stats = _get_array_stats(lapse_rate_03km, "0-3km Lapse Rate")
    
    validation['input_diagnostics'] = {
        'cape_03km': cape_stats,
        'lapse_rate_03km': lapse_stats
    }
    
    # Check for HRRR diagnostic scaling issues
    high_cape_points = np.sum(cape_03km > 400)
    validation['flags']['high_cape_03km_values'] = high_cape_points > 0
    validation['input_diagnostics']['high_cape_03km_points'] = int(high_cape_points)
    
    return validation


def _get_array_stats(arr: np.ndarray, name: str) -> Dict[str, Any]:
    """Get basic statistics for an array"""
    valid_data = arr[np.isfinite(arr) & (arr >= 0)]
    
    if len(valid_data) > 0:
        return {
            'name': name,
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'median': float(np.median(valid_data)),
            'valid_points': int(len(valid_data))
        }
    else:
        return {'name': name, 'no_valid_data': True}


def print_validation_report(validation: Dict[str, Any]) -> None:
    """Print a formatted validation report"""
    print(f"\n{'='*60}")
    print(f"VTP VALIDATION REPORT: {validation['case_name']}")
    print(f"{'='*60}")
    
    print(f"Grid Points: {validation['total_grid_points']:,}")
    print(f"Valid VTP Points: {validation['valid_vtp_points']:,} ({validation['vtp_coverage_pct']:.1f}%)")
    
    if 'vtp_stats' in validation and 'no_valid_data' not in validation['vtp_stats']:
        stats = validation['vtp_stats']
        print(f"\nVTP Statistics:")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
        print(f"  95th percentile: {stats['p95']:.2f}")
        print(f"  99th percentile: {stats['p99']:.2f}")
        
        counts = validation['threshold_counts']
        print(f"\nThreshold Exceedances:")
        print(f"  VTP > 0.5: {counts['vtp_gt_0p5']:,}")
        print(f"  VTP > 1.0: {counts['vtp_gt_1']:,}")
        print(f"  VTP > 2.0: {counts['vtp_gt_2']:,}")
        print(f"  VTP > 5.0: {counts['vtp_gt_5']:,}")
        print(f"  VTP > 10.0: {counts['vtp_gt_10']:,}")
    else:
        print("\nNo valid VTP data found!")
    
    # Flags
    flags = validation['flags']
    print(f"\nDiagnostic Flags:")
    if flags['potential_carpet_bombing']:
        print("  ⚠️  POTENTIAL CARPET BOMBING: >20% coverage")
    if flags['extreme_values_present']:
        print("  ⚠️  EXTREME VALUES: VTP > 10 detected")
    if flags['widespread_moderate_vtp']:
        print("  ⚠️  WIDESPREAD MODERATE VTP: >5% of domain has VTP > 2")
    if flags['high_cape_03km_values']:
        print("  📊 HIGH 0-3KM CAPE VALUES: HRRR diagnostic scaling applied")
    
    if not any(flags.values()):
        print("  ✅ All checks passed")
    
    # Input diagnostics
    input_diag = validation['input_diagnostics']
    print(f"\nInput Field Diagnostics:")
    for field_name, stats in input_diag.items():
        if isinstance(stats, dict) and 'no_valid_data' not in stats:
            print(f"  {stats['name']}: {stats['min']:.1f} - {stats['max']:.1f} "
                  f"(mean: {stats['mean']:.1f}, {stats['valid_points']:,} valid)")


def benchmark_case_expectations() -> Dict[str, Dict[str, Any]]:
    """
    Return expected validation results for benchmark cases
    """
    return {
        'quiet_summer': {
            'description': 'Quiet summer HRRR run (no severe risk)',
            'expected_max_vtp': 0.5,
            'expected_coverage_pct': '<5%',
            'expected_flags': ['minimal_activity']
        },
        'lee_county_20190303': {
            'description': '3 Mar 2019 20 UTC - Lee County, AL EF4',
            'expected_max_vtp': 6.0,
            'expected_coverage_pct': '5-15%',
            'expected_flags': ['focused_high_values']
        },
        'april27_2011': {
            'description': '27 Apr 2011 18 UTC - Historic outbreak',
            'expected_max_vtp': 9.0,
            'expected_coverage_pct': '10-25%',
            'expected_flags': ['multiple_corridors']
        }
    }


# Example usage for validation
if __name__ == "__main__":
    # This would be called with actual HRRR data
    print("VTP Validation Module")
    print("Usage: Call validate_vtp_implementation() with HRRR data arrays")
    print("\nBenchmark cases:")
    benchmarks = benchmark_case_expectations()
    for case_id, details in benchmarks.items():
        print(f"  {case_id}: {details['description']}")
        print(f"    Expected max VTP: {details['expected_max_vtp']}")
        print(f"    Expected coverage: {details['expected_coverage_pct']}")