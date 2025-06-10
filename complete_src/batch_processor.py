#!/usr/bin/env python3
"""
HRRR Batch Processing Pipeline
Handles multi-hour, multi-day, and large-scale HRRR dataset processing
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import concurrent.futures
from typing import List, Dict, Optional, Tuple
import time
import argparse

# Import our Tier 2 pipeline
from hrrr_tier2_pipeline import HRRRTier2Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class HRRRBatchProcessor:
    """Batch processor for large-scale HRRR dataset creation"""
    
    def __init__(self, output_dir: str = "batch_training_data", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.pipeline = HRRRTier2Pipeline(str(self.output_dir))
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'failed_files': []
        }
    
    def generate_date_hour_combinations(self, start_date: str, end_date: str, 
                                      hours: List[int], forecast_hours: List[str]) -> List[Dict]:
        """Generate all combinations of dates, hours, and forecast hours"""
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        combinations = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            
            for hour in hours:
                for forecast_hour in forecast_hours:
                    combinations.append({
                        'date_str': date_str,
                        'hour': hour,
                        'forecast_hour': forecast_hour,
                        'datetime': current_dt.replace(hour=hour)
                    })
            
            current_dt += timedelta(days=1)
        
        return combinations
    
    def process_single_combination(self, combo: Dict) -> Dict:
        """Process a single date/hour/forecast combination"""
        try:
            logger.info(f"Processing {combo['date_str']} {combo['hour']:02d}Z F{combo['forecast_hour']}")
            
            start_time = time.time()
            output_path = self.pipeline.create_tier2_dataset(
                combo['date_str'], 
                combo['hour'], 
                combo['forecast_hour']
            )
            end_time = time.time()
            
            if output_path:
                file_size = Path(output_path).stat().st_size / (1024*1024)  # MB
                return {
                    'status': 'success',
                    'combo': combo,
                    'output_path': output_path,
                    'processing_time': end_time - start_time,
                    'file_size_mb': file_size
                }
            else:
                return {
                    'status': 'failed',
                    'combo': combo,
                    'error': 'Pipeline returned None'
                }
                
        except Exception as e:
            logger.error(f"Failed to process {combo}: {e}")
            return {
                'status': 'failed',
                'combo': combo,
                'error': str(e)
            }
    
    def process_batch_sequential(self, combinations: List[Dict]) -> List[Dict]:
        """Process combinations sequentially (safer for large batches)"""
        results = []
        
        for i, combo in enumerate(combinations):
            logger.info(f"Processing {i+1}/{len(combinations)}")
            result = self.process_single_combination(combo)
            results.append(result)
            
            # Update statistics
            if result['status'] == 'success':
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_files'].append(result)
        
        return results
    
    def process_batch_parallel(self, combinations: List[Dict], max_workers: int = None) -> List[Dict]:
        """Process combinations in parallel (faster but more resource intensive)"""
        if max_workers is None:
            max_workers = self.max_workers
        
        logger.info(f"Starting parallel processing with {max_workers} workers")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_combo = {
                executor.submit(self.process_single_combination, combo): combo 
                for combo in combinations
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    if result['status'] == 'success':
                        self.stats['successful'] += 1
                        logger.info(f"✅ Success: {combo['date_str']} {combo['hour']:02d}Z F{combo['forecast_hour']}")
                    else:
                        self.stats['failed'] += 1
                        self.stats['failed_files'].append(result)
                        logger.warning(f"❌ Failed: {combo['date_str']} {combo['hour']:02d}Z F{combo['forecast_hour']}")
                        
                except Exception as e:
                    logger.error(f"Exception in parallel processing: {e}")
                    self.stats['failed'] += 1
        
        return results
    
    def create_development_dataset(self, start_date: str = None) -> str:
        """Create a small development/testing dataset (1 week, analysis only)"""
        if start_date is None:
            # Use a recent date (adjust as needed)
            start_date = "20250610"
        
        end_date = (datetime.strptime(start_date, '%Y%m%d') + timedelta(days=6)).strftime('%Y%m%d')
        
        logger.info(f"🔧 Creating development dataset: {start_date} to {end_date}")
        
        combinations = self.generate_date_hour_combinations(
            start_date, end_date,
            hours=[0, 6, 12, 18],  # 4 runs per day
            forecast_hours=['00']   # Analysis only
        )
        
        return self.run_batch_processing(combinations, "development")
    
    def create_training_dataset(self, start_date: str, num_days: int = 30) -> str:
        """Create a training dataset (multiple forecast hours)"""
        end_date = (datetime.strptime(start_date, '%Y%m%d') + timedelta(days=num_days-1)).strftime('%Y%m%d')
        
        logger.info(f"🎯 Creating training dataset: {start_date} to {end_date}")
        
        combinations = self.generate_date_hour_combinations(
            start_date, end_date,
            hours=[0, 6, 12, 18],          # 4 runs per day
            forecast_hours=['00', '01', '02', '06', '12', '18']  # Key forecast hours
        )
        
        return self.run_batch_processing(combinations, "training")
    
    def create_custom_dataset(self, start_date: str, end_date: str, 
                            hours: List[int], forecast_hours: List[str]) -> str:
        """Create a custom dataset with specified parameters"""
        logger.info(f"⚙️  Creating custom dataset: {start_date} to {end_date}")
        
        combinations = self.generate_date_hour_combinations(
            start_date, end_date, hours, forecast_hours
        )
        
        return self.run_batch_processing(combinations, "custom")
    
    def run_batch_processing(self, combinations: List[Dict], dataset_type: str) -> str:
        """Run the batch processing and generate summary"""
        self.stats['total_files'] = len(combinations)
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"🚀 Starting batch processing: {len(combinations)} files")
        
        # Choose processing method based on dataset size
        if len(combinations) <= 50:
            results = self.process_batch_parallel(combinations)
        else:
            logger.info("Large dataset detected, using sequential processing")
            results = self.process_batch_sequential(combinations)
        
        self.stats['end_time'] = datetime.now()
        
        # Generate summary
        summary_path = self.generate_batch_summary(results, dataset_type)
        
        return summary_path
    
    def generate_batch_summary(self, results: List[Dict], dataset_type: str) -> str:
        """Generate a comprehensive batch processing summary"""
        successful_results = [r for r in results if r['status'] == 'success']
        
        # Calculate statistics
        if successful_results:
            total_size_mb = sum(r['file_size_mb'] for r in successful_results)
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            total_processing_time = sum(r['processing_time'] for r in successful_results)
        else:
            total_size_mb = avg_processing_time = total_processing_time = 0
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        summary = {
            'dataset_type': dataset_type,
            'timestamp': datetime.now().isoformat(),
            'processing_duration_minutes': duration / 60,
            'statistics': {
                'total_files': self.stats['total_files'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': (self.stats['successful'] / self.stats['total_files']) * 100,
                'total_size_gb': total_size_mb / 1024,
                'avg_processing_time_minutes': avg_processing_time / 60,
                'total_processing_time_hours': total_processing_time / 3600
            },
            'successful_files': [
                {
                    'date': r['combo']['date_str'],
                    'hour': r['combo']['hour'],
                    'forecast_hour': r['combo']['forecast_hour'],
                    'output_path': r['output_path'],
                    'size_mb': r['file_size_mb']
                }
                for r in successful_results
            ],
            'failed_files': [
                {
                    'date': r['combo']['date_str'],
                    'hour': r['combo']['hour'],
                    'forecast_hour': r['combo']['forecast_hour'],
                    'error': r['error']
                }
                for r in self.stats['failed_files']
            ]
        }
        
        # Save summary
        summary_filename = f"batch_summary_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print human-readable summary
        self.print_batch_summary(summary)
        
        return str(summary_path)
    
    def print_batch_summary(self, summary: Dict):
        """Print human-readable batch processing summary"""
        stats = summary['statistics']
        
        print("\n" + "="*60)
        print(f"🎉 BATCH PROCESSING COMPLETE - {summary['dataset_type'].upper()}")
        print("="*60)
        print(f"⏱️  Duration: {summary['processing_duration_minutes']:.1f} minutes")
        print(f"📊 Success Rate: {stats['success_rate']:.1f}%")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"💾 Total Size: {stats['total_size_gb']:.2f} GB")
        print(f"🕐 Avg Processing: {stats['avg_processing_time_minutes']:.1f} min/file")
        
        if summary['failed_files']:
            print(f"\n⚠️  FAILED FILES ({len(summary['failed_files'])}):")
            for failed in summary['failed_files'][:5]:  # Show first 5
                print(f"   • {failed['date']} {failed['hour']:02d}Z F{failed['forecast_hour']}: {failed['error']}")
            if len(summary['failed_files']) > 5:
                print(f"   ... and {len(summary['failed_files']) - 5} more")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='HRRR Batch Processing Pipeline')
    parser.add_argument('--mode', choices=['dev', 'training', 'custom'], default='dev',
                       help='Processing mode')
    parser.add_argument('--start-date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', help='End date (YYYYMMDD)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to process')
    parser.add_argument('--hours', nargs='+', type=int, default=[0, 6, 12, 18],
                       help='Model run hours')
    parser.add_argument('--forecast-hours', nargs='+', default=['00'],
                       help='Forecast hours')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output-dir', default='batch_training_data', help='Output directory')
    
    args = parser.parse_args()
    
    processor = HRRRBatchProcessor(args.output_dir, args.workers)
    
    if args.mode == 'dev':
        summary_path = processor.create_development_dataset(args.start_date)
    elif args.mode == 'training':
        if not args.start_date:
            print("❌ --start-date required for training mode")
            sys.exit(1)
        summary_path = processor.create_training_dataset(args.start_date, args.days)
    elif args.mode == 'custom':
        if not args.start_date or not args.end_date:
            print("❌ --start-date and --end-date required for custom mode")
            sys.exit(1)
        summary_path = processor.create_custom_dataset(
            args.start_date, args.end_date, args.hours, args.forecast_hours
        )
    
    print(f"\n📄 Batch summary saved to: {summary_path}")

if __name__ == "__main__":
    main()