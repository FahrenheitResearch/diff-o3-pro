#!/usr/bin/env python3
"""
HRRR Training Data Pipeline Requirements Calculator
Estimates storage, bandwidth, and processing requirements for different scales
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import json

class HRRRRequirementsCalculator:
    """Calculate storage and processing requirements for HRRR training pipeline"""
    
    def __init__(self):
        # Base measurements from successful Tier 2 processing
        self.base_measurements = {
            'grib_file_size_mb': 390.2,  # Raw HRRR GRIB file
            'netcdf_size_mb': 20.0,      # Processed NetCDF (5 variables)
            'tier2_estimated_mb': 84.0,   # Estimated for 21 variables (20MB * 21/5)
            'variables_tier1': 13,
            'variables_tier2': 21,
            'grid_points': 1905141,      # 1059 × 1799 native 3km
            'processing_time_minutes': 5, # Estimated per file
        }
        
        # HRRR model specifications
        self.hrrr_specs = {
            'runs_per_day': 24,          # Every hour
            'forecast_hours': 48,        # 0-47 hours
            'days_per_year': 365,
            'archive_start_year': 2016,  # HRRR archive availability
        }
    
    def calculate_single_file_requirements(self) -> dict:
        """Calculate requirements for a single HRRR file"""
        return {
            'raw_grib_mb': self.base_measurements['grib_file_size_mb'],
            'tier1_netcdf_mb': self.base_measurements['netcdf_size_mb'] * (
                self.base_measurements['variables_tier1'] / 5
            ),
            'tier2_netcdf_mb': self.base_measurements['tier2_estimated_mb'],
            'processing_time_min': self.base_measurements['processing_time_minutes'],
            'compression_ratio': self.base_measurements['grib_file_size_mb'] / 
                               self.base_measurements['tier2_estimated_mb']
        }
    
    def calculate_daily_requirements(self, forecast_hours_used: int = 1) -> dict:
        """Calculate requirements for one day of HRRR data"""
        single_file = self.calculate_single_file_requirements()
        files_per_day = self.hrrr_specs['runs_per_day'] * forecast_hours_used
        
        return {
            'files_count': files_per_day,
            'raw_grib_gb': (single_file['raw_grib_mb'] * files_per_day) / 1024,
            'tier1_netcdf_gb': (single_file['tier1_netcdf_mb'] * files_per_day) / 1024,
            'tier2_netcdf_gb': (single_file['tier2_netcdf_mb'] * files_per_day) / 1024,
            'processing_hours': (single_file['processing_time_min'] * files_per_day) / 60,
            'download_time_hours': files_per_day * 2 / 60,  # ~2 min per 390MB file
        }
    
    def calculate_monthly_requirements(self, forecast_hours_used: int = 1) -> dict:
        """Calculate requirements for one month of HRRR data"""
        daily = self.calculate_daily_requirements(forecast_hours_used)
        days = 30  # Average month
        
        return {
            'files_count': daily['files_count'] * days,
            'raw_grib_tb': (daily['raw_grib_gb'] * days) / 1024,
            'tier1_netcdf_tb': (daily['tier1_netcdf_gb'] * days) / 1024,
            'tier2_netcdf_tb': (daily['tier2_netcdf_gb'] * days) / 1024,
            'processing_days': daily['processing_hours'] * days / 24,
            'download_days': daily['download_time_hours'] * days / 24,
        }
    
    def calculate_yearly_requirements(self, forecast_hours_used: int = 1) -> dict:
        """Calculate requirements for one year of HRRR data"""
        daily = self.calculate_daily_requirements(forecast_hours_used)
        days = self.hrrr_specs['days_per_year']
        
        return {
            'files_count': daily['files_count'] * days,
            'raw_grib_pb': (daily['raw_grib_gb'] * days) / (1024 * 1024),
            'tier1_netcdf_pb': (daily['tier1_netcdf_gb'] * days) / (1024 * 1024),
            'tier2_netcdf_pb': (daily['tier2_netcdf_gb'] * days) / (1024 * 1024),
            'processing_weeks': daily['processing_hours'] * days / (24 * 7),
            'download_weeks': daily['download_time_hours'] * days / (24 * 7),
        }
    
    def calculate_multi_year_archive(self, years: int, forecast_hours_used: int = 1) -> dict:
        """Calculate requirements for multi-year HRRR archive"""
        yearly = self.calculate_yearly_requirements(forecast_hours_used)
        
        return {
            'years': years,
            'files_count_millions': (yearly['files_count'] * years) / 1_000_000,
            'raw_grib_pb': yearly['raw_grib_pb'] * years,
            'tier1_netcdf_pb': yearly['tier1_netcdf_pb'] * years,
            'tier2_netcdf_pb': yearly['tier2_netcdf_pb'] * years,
            'processing_months': yearly['processing_weeks'] * years / 4.33,
            'download_months': yearly['download_weeks'] * years / 4.33,
        }
    
    def calculate_training_scenarios(self) -> dict:
        """Calculate requirements for common training scenarios"""
        scenarios = {}
        
        # Scenario 1: Development/Testing (1 week, analysis only)
        dev_daily = self.calculate_daily_requirements(1)
        scenarios['development'] = {
            'name': 'Development/Testing',
            'timeframe': '1 week, analysis only (F00)',
            'files': dev_daily['files_count'] * 7,
            'storage_gb': dev_daily['tier2_netcdf_gb'] * 7,
            'processing_hours': dev_daily['processing_hours'] * 7,
            'description': 'Quick testing and initial model development'
        }
        
        # Scenario 2: Training Set (1 year, multiple forecast hours)
        train_yearly = self.calculate_yearly_requirements(6)  # F00, F01, F02, F06, F12, F18
        scenarios['training'] = {
            'name': 'Training Dataset',
            'timeframe': '1 year, 6 forecast hours',
            'files': train_yearly['files_count'],
            'storage_tb': train_yearly['tier2_netcdf_pb'] * 1024,
            'processing_weeks': train_yearly['processing_weeks'],
            'description': 'Comprehensive training dataset for production model'
        }
        
        # Scenario 3: Research Archive (5 years, comprehensive)
        research_archive = self.calculate_multi_year_archive(5, 12)  # F00-F11
        scenarios['research'] = {
            'name': 'Research Archive',
            'timeframe': '5 years, 12 forecast hours',
            'files_millions': research_archive['files_count_millions'],
            'storage_pb': research_archive['tier2_netcdf_pb'],
            'processing_months': research_archive['processing_months'],
            'description': 'Full research archive for comprehensive climate/weather studies'
        }
        
        return scenarios
    
    def calculate_bandwidth_requirements(self) -> dict:
        """Calculate network bandwidth requirements"""
        single_file_mb = self.base_measurements['grib_file_size_mb']
        
        return {
            'per_file_mb': single_file_mb,
            'hourly_mb': single_file_mb,  # 1 file per hour (analysis)
            'daily_gb': (single_file_mb * 24) / 1024,
            'weekly_gb': (single_file_mb * 24 * 7) / 1024,
            'monthly_tb': (single_file_mb * 24 * 30) / (1024 * 1024),
            'sustained_mbps': single_file_mb * 8 / (60 * 60),  # Assuming 1-hour window
        }
    
    def calculate_processing_requirements(self) -> dict:
        """Calculate computational processing requirements"""
        base_time = self.base_measurements['processing_time_minutes']
        
        return {
            'single_file_minutes': base_time,
            'parallel_factor': 4,  # Estimated speedup with 4 cores
            'memory_gb_per_file': 8,  # Estimated RAM usage
            'disk_io_intensive': True,
            'cpu_cores_recommended': 8,
            'ram_recommended_gb': 32,
            'storage_type': 'SSD recommended for processing speed'
        }
    
    def generate_requirements_report(self) -> dict:
        """Generate comprehensive requirements report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_measurements': self.base_measurements,
            'single_file': self.calculate_single_file_requirements(),
            'daily': self.calculate_daily_requirements(),
            'monthly': self.calculate_monthly_requirements(),
            'yearly': self.calculate_yearly_requirements(),
            'scenarios': self.calculate_training_scenarios(),
            'bandwidth': self.calculate_bandwidth_requirements(),
            'processing': self.calculate_processing_requirements(),
            'recommendations': {
                'development': 'Start with 1-week analysis-only dataset (~59 GB)',
                'training': 'Use 1-year subset with key forecast hours (~10 TB)',
                'production': 'Consider cloud storage and distributed processing',
                'bandwidth': 'Sustained 100+ Mbps recommended for real-time processing',
                'storage': 'Plan for 10-100x data expansion over time'
            }
        }
        
        return report
    
    def print_summary_report(self):
        """Print a human-readable summary"""
        report = self.generate_requirements_report()
        
        print("🌩️  HRRR Training Data Pipeline Requirements Calculator")
        print("=" * 60)
        
        print("\n📊 BASE MEASUREMENTS:")
        print(f"   • Raw HRRR GRIB file: {report['base_measurements']['grib_file_size_mb']:.1f} MB")
        print(f"   • Tier 2 NetCDF file: {report['base_measurements']['tier2_estimated_mb']:.1f} MB (21 variables)")
        print(f"   • Compression ratio: {report['single_file']['compression_ratio']:.1f}:1")
        print(f"   • Processing time: {report['base_measurements']['processing_time_minutes']} min/file")
        
        print("\n⏱️  PROCESSING SCENARIOS:")
        for scenario_key, scenario in report['scenarios'].items():
            print(f"   {scenario['name']}:")
            print(f"      └─ {scenario['description']}")
            print(f"      └─ {scenario['timeframe']}")
            if 'files' in scenario:
                print(f"      └─ Files: {scenario['files']:,}")
            if 'storage_gb' in scenario:
                print(f"      └─ Storage: {scenario['storage_gb']:.1f} GB")
            if 'storage_tb' in scenario:
                print(f"      └─ Storage: {scenario['storage_tb']:.1f} TB")
            if 'storage_pb' in scenario:
                print(f"      └─ Storage: {scenario['storage_pb']:.2f} PB")
            print()
        
        print("🌐 BANDWIDTH REQUIREMENTS:")
        bw = report['bandwidth']
        print(f"   • Per GRIB file: {bw['per_file_mb']:.1f} MB")
        print(f"   • Daily download: {bw['daily_gb']:.1f} GB")
        print(f"   • Monthly download: {bw['monthly_tb']:.2f} TB")
        print(f"   • Sustained bandwidth: {bw['sustained_mbps']:.1f} Mbps")
        
        print("\n💻 PROCESSING REQUIREMENTS:")
        proc = report['processing']
        print(f"   • CPU cores: {proc['cpu_cores_recommended']} recommended")
        print(f"   • RAM: {proc['ram_recommended_gb']} GB recommended")
        print(f"   • Storage: {proc['storage_type']}")
        print(f"   • Processing time: {proc['single_file_minutes']} min/file")
        
        print("\n🎯 RECOMMENDATIONS:")
        for rec in report['recommendations'].values():
            print(f"   • {rec}")
        
        print("\n" + "=" * 60)

def main():
    calc = HRRRRequirementsCalculator()
    calc.print_summary_report()
    
    # Save detailed report
    report = calc.generate_requirements_report()
    output_file = Path("tier2_pipeline_requirements.json")
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved to: {output_file}")

if __name__ == "__main__":
    main()