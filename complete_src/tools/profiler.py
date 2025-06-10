#!/usr/bin/env python3
"""
Performance Profiling System for HRRR Processing
Provides CPU, memory, and I/O profiling with detailed breakdowns
"""

import time
import psutil
import os
import sys
import functools
import threading
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union
import cProfile
import pstats
import io


class ProfilerResults:
    """Container for profiling results"""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.function_times = defaultdict(list)
        self.memory_snapshots = []
        self.cpu_snapshots = []
        self.io_snapshots = []
        self.custom_metrics = {}
        self.phase_times = {}
        self.errors = []
        
    def add_function_time(self, func_name: str, duration: float, args: tuple = None):
        """Add function timing"""
        self.function_times[func_name].append({
            'duration': duration,
            'timestamp': time.time(),
            'args': str(args) if args else None
        })
    
    def add_memory_snapshot(self, label: str = None):
        """Add memory usage snapshot"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'system_total_mb': memory.total / 1024 / 1024,
                'system_used_mb': memory.used / 1024 / 1024,
                'system_available_mb': memory.available / 1024 / 1024,
                'system_percent': memory.percent,
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'process_percent': process.memory_percent()
            }
            self.memory_snapshots.append(snapshot)
        except Exception as e:
            self.errors.append(f"Memory snapshot error: {e}")
    
    def add_cpu_snapshot(self, label: str = None):
        """Add CPU usage snapshot"""
        try:
            process = psutil.Process()
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'system_cpu_percent': psutil.cpu_percent(interval=None),
                'process_cpu_percent': process.cpu_percent(),
                'cpu_count': psutil.cpu_count(),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            self.cpu_snapshots.append(snapshot)
        except Exception as e:
            self.errors.append(f"CPU snapshot error: {e}")
    
    def add_io_snapshot(self, label: str = None):
        """Add I/O usage snapshot"""
        try:
            process = psutil.Process()
            io_counters = process.io_counters()
            disk_usage = psutil.disk_usage('.')
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                'disk_total_gb': disk_usage.total / 1024 / 1024 / 1024,
                'disk_used_gb': disk_usage.used / 1024 / 1024 / 1024,
                'disk_free_gb': disk_usage.free / 1024 / 1024 / 1024
            }
            self.io_snapshots.append(snapshot)
        except Exception as e:
            self.errors.append(f"I/O snapshot error: {e}")
    
    def add_phase_time(self, phase: str, duration: float):
        """Add phase timing"""
        self.phase_times[phase] = duration
    
    def add_custom_metric(self, name: str, value: Union[int, float, str]):
        """Add custom metric"""
        self.custom_metrics[name] = value
    
    def finalize(self):
        """Finalize profiling"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        total_time = (self.end_time or time.time()) - self.start_time
        
        # Function timing summary
        func_summary = {}
        for func_name, times in self.function_times.items():
            durations = [t['duration'] for t in times]
            func_summary[func_name] = {
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'call_count': len(durations),
                'percent_of_total': (sum(durations) / total_time * 100) if total_time > 0 else 0
            }
        
        # Memory summary
        memory_summary = {}
        if self.memory_snapshots:
            memory_values = [s['process_memory_mb'] for s in self.memory_snapshots]
            memory_summary = {
                'peak_memory_mb': max(memory_values),
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'min_memory_mb': min(memory_values),
                'memory_growth_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
            }
        
        # CPU summary
        cpu_summary = {}
        if self.cpu_snapshots:
            cpu_values = [s['process_cpu_percent'] for s in self.cpu_snapshots if s['process_cpu_percent'] is not None]
            if cpu_values:
                cpu_summary = {
                    'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                    'max_cpu_percent': max(cpu_values),
                    'min_cpu_percent': min(cpu_values)
                }
        
        return {
            'total_runtime_seconds': total_time,
            'function_timings': func_summary,
            'memory_usage': memory_summary,
            'cpu_usage': cpu_summary,
            'phase_timings': self.phase_times,
            'custom_metrics': self.custom_metrics,
            'snapshot_counts': {
                'memory': len(self.memory_snapshots),
                'cpu': len(self.cpu_snapshots),
                'io': len(self.io_snapshots)
            },
            'errors': self.errors
        }


class HRRRProfiler:
    """Main profiling class for HRRR processing"""
    
    def __init__(self, enabled: bool = True, sample_interval: float = 1.0):
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.results = ProfilerResults()
        self._monitor_thread = None
        self._monitoring = False
        self._cprofile = None
        
    def start(self):
        """Start profiling"""
        if not self.enabled:
            return
            
        print(f"🔍 Starting performance profiling (sample interval: {self.sample_interval}s)")
        
        # Initial snapshots
        self.results.add_memory_snapshot("start")
        self.results.add_cpu_snapshot("start")
        self.results.add_io_snapshot("start")
        
        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # Start cProfile
        self._cprofile = cProfile.Profile()
        self._cprofile.enable()
    
    def stop(self):
        """Stop profiling"""
        if not self.enabled:
            return
            
        # Stop cProfile
        if self._cprofile:
            self._cprofile.disable()
        
        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        # Final snapshots
        self.results.add_memory_snapshot("end")
        self.results.add_cpu_snapshot("end")
        self.results.add_io_snapshot("end")
        
        # Finalize
        self.results.finalize()
        
        print(f"⏹️ Profiling stopped. Total runtime: {self.results.end_time - self.results.start_time:.2f}s")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self.results.add_memory_snapshot()
                self.results.add_cpu_snapshot()
                self.results.add_io_snapshot()
                time.sleep(self.sample_interval)
            except Exception as e:
                self.results.errors.append(f"Monitor loop error: {e}")
                break
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile individual functions"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.results.add_function_time(name, duration, args)
            
            return wrapper
        return decorator
    
    @contextmanager
    def profile_phase(self, phase_name: str):
        """Context manager to profile phases"""
        if not self.enabled:
            yield
            return
            
        print(f"📊 Starting phase: {phase_name}")
        start_time = time.time()
        
        self.results.add_memory_snapshot(f"phase_start_{phase_name}")
        self.results.add_cpu_snapshot(f"phase_start_{phase_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.results.add_phase_time(phase_name, duration)
            
            self.results.add_memory_snapshot(f"phase_end_{phase_name}")
            self.results.add_cpu_snapshot(f"phase_end_{phase_name}")
            
            print(f"✅ Phase '{phase_name}' completed in {duration:.2f}s")
    
    def add_metric(self, name: str, value: Union[int, float, str]):
        """Add custom metric"""
        if self.enabled:
            self.results.add_custom_metric(name, value)
    
    def print_summary(self):
        """Print profiling summary"""
        if not self.enabled:
            return
            
        summary = self.results.get_summary()
        
        print("\n" + "="*60)
        print("🔍 PERFORMANCE PROFILING SUMMARY")
        print("="*60)
        
        print(f"📊 Total Runtime: {summary['total_runtime_seconds']:.2f} seconds")
        
        # Memory usage
        if summary['memory_usage']:
            mem = summary['memory_usage']
            print(f"\n💾 Memory Usage:")
            print(f"  Peak: {mem['peak_memory_mb']:.1f} MB")
            print(f"  Average: {mem['avg_memory_mb']:.1f} MB")
            print(f"  Growth: {mem['memory_growth_mb']:.1f} MB")
        
        # CPU usage
        if summary['cpu_usage']:
            cpu = summary['cpu_usage']
            print(f"\n🖥️ CPU Usage:")
            print(f"  Average: {cpu['avg_cpu_percent']:.1f}%")
            print(f"  Peak: {cpu['max_cpu_percent']:.1f}%")
        
        # Phase timings
        if summary['phase_timings']:
            print(f"\n⏱️ Phase Timings:")
            for phase, duration in summary['phase_timings'].items():
                percent = (duration / summary['total_runtime_seconds'] * 100)
                print(f"  {phase}: {duration:.2f}s ({percent:.1f}%)")
        
        # Top functions by time
        func_timings = summary['function_timings']
        if func_timings:
            print(f"\n🚀 Top Functions by Time:")
            sorted_funcs = sorted(func_timings.items(), 
                                key=lambda x: x[1]['total_time'], 
                                reverse=True)[:10]
            
            for func_name, stats in sorted_funcs:
                print(f"  {func_name}: {stats['total_time']:.2f}s "
                      f"({stats['call_count']} calls, "
                      f"{stats['percent_of_total']:.1f}%)")
        
        # Custom metrics
        if summary['custom_metrics']:
            print(f"\n📈 Custom Metrics:")
            for name, value in summary['custom_metrics'].items():
                print(f"  {name}: {value}")
        
        # Errors
        if summary['errors']:
            print(f"\n⚠️ Errors ({len(summary['errors'])}):")
            for error in summary['errors'][:5]:  # Show first 5 errors
                print(f"  {error}")
        
        print("="*60 + "\n")
    
    def export_detailed_report(self, output_file: Path = None):
        """Export detailed profiling report"""
        if not self.enabled:
            return
            
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(f"hrrr_profile_{timestamp}.json")
        
        # Prepare detailed data
        detailed_data = {
            'summary': self.results.get_summary(),
            'raw_data': {
                'function_times': dict(self.results.function_times),
                'memory_snapshots': self.results.memory_snapshots,
                'cpu_snapshots': self.results.cpu_snapshots,
                'io_snapshots': self.results.io_snapshots,
                'phase_times': self.results.phase_times,
                'custom_metrics': self.results.custom_metrics
            },
            'metadata': {
                'start_time': self.results.start_time,
                'end_time': self.results.end_time,
                'sample_interval': self.sample_interval,
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Export cProfile data if available
        if self._cprofile:
            cprofile_data = io.StringIO()
            stats = pstats.Stats(self._cprofile, stream=cprofile_data)
            stats.sort_stats('cumulative').print_stats(50)  # Top 50 functions
            detailed_data['cprofile_output'] = cprofile_data.getvalue()
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        print(f"📄 Detailed profiling report exported to: {output_file}")
        return output_file
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top performance bottlenecks"""
        if not self.enabled:
            return []
            
        summary = self.results.get_summary()
        bottlenecks = []
        
        # Function bottlenecks
        func_timings = summary['function_timings']
        if func_timings:
            sorted_funcs = sorted(func_timings.items(), 
                                key=lambda x: x[1]['total_time'], 
                                reverse=True)[:top_n]
            
            for func_name, stats in sorted_funcs:
                bottlenecks.append({
                    'type': 'function',
                    'name': func_name,
                    'time_seconds': stats['total_time'],
                    'percent_of_total': stats['percent_of_total'],
                    'call_count': stats['call_count'],
                    'avg_time': stats['avg_time']
                })
        
        # Phase bottlenecks
        if summary['phase_timings']:
            sorted_phases = sorted(summary['phase_timings'].items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_n]
            
            for phase_name, duration in sorted_phases:
                percent = (duration / summary['total_runtime_seconds'] * 100)
                bottlenecks.append({
                    'type': 'phase',
                    'name': phase_name,
                    'time_seconds': duration,
                    'percent_of_total': percent
                })
        
        return sorted(bottlenecks, key=lambda x: x['time_seconds'], reverse=True)[:top_n]


# Global profiler instance
_global_profiler = None

def get_profiler() -> HRRRProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = HRRRProfiler()
    return _global_profiler

def start_profiling(sample_interval: float = 1.0):
    """Start global profiling"""
    global _global_profiler
    _global_profiler = HRRRProfiler(enabled=True, sample_interval=sample_interval)
    _global_profiler.start()

def stop_profiling():
    """Stop global profiling"""
    if _global_profiler:
        _global_profiler.stop()

def profile_function(func_name: str = None):
    """Decorator for profiling functions"""
    return get_profiler().profile_function(func_name)

def profile_phase(phase_name: str):
    """Context manager for profiling phases"""
    return get_profiler().profile_phase(phase_name)


if __name__ == '__main__':
    # Demo the profiler
    profiler = HRRRProfiler(enabled=True, sample_interval=0.5)
    profiler.start()
    
    @profiler.profile_function("demo_function")
    def slow_function():
        time.sleep(1)
        return "done"
    
    with profiler.profile_phase("demo_phase"):
        slow_function()
        time.sleep(0.5)
    
    profiler.add_metric("demo_metric", 42)
    profiler.stop()
    profiler.print_summary()
    
    bottlenecks = profiler.get_bottlenecks()
    print("Top bottlenecks:", bottlenecks)