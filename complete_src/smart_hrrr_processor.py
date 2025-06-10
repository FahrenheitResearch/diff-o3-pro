#!/usr/bin/env python3
"""
Smart HRRR Processor
Organized output structure with duplicate detection and efficient processing
"""

import sys
import argparse
import logging
import traceback
import subprocess
import time
import shutil
import urllib.request
import urllib.error
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
import json


def parse_hour_range(range_str):
    """Parse forecast hour range string into a list of ints."""
    if not range_str:
        return None
    if "-" in range_str:
        start, end = map(int, range_str.split("-"))
        return list(range(start, end + 1))
    return [int(h) for h in range_str.split(",") if h]


FILTER_FILE = Path(__file__).parent / "custom_filters.json"
WORKFLOW_FILE = Path(__file__).parent / "custom_workflows.json"

from tools.profiler import HRRRProfiler, profile_function, profile_phase
from tools.process_all_products import process_all_products
from hrrr_processor_refactored import HRRRProcessor


def setup_logging(debug=False, output_dir=None):
    """Setup logging with organized output"""
    log_level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create logs directory in output structure
    if output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"processing_{timestamp}.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"smart_hrrr_{timestamp}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(log_level)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    logger = logging.getLogger(__name__)
    logger.info(f"Smart HRRR processor initialized. Log: {log_file}")
    return logger


def check_system_memory():
    """Check system memory usage"""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / 1024 / 1024,
            "available_mb": memory.available / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "percent": memory.percent,
        }
    except:
        return None


def create_output_structure(model, date, hour):
    """Create organized output directory structure"""
    base_dir = Path("outputs")
    model_dir = base_dir / model.lower()
    date_dir = model_dir / date
    run_dir = date_dir / f"{hour:02d}z"

    # Create all directories
    run_dir.mkdir(parents=True, exist_ok=True)

    return {"base": base_dir, "model": model_dir, "date": date_dir, "run": run_dir}


def get_forecast_hour_dir(run_dir, forecast_hour):
    """Get directory for specific forecast hour"""
    fhr_dir = run_dir / f"F{forecast_hour:02d}"
    fhr_dir.mkdir(exist_ok=True)
    return fhr_dir


@profile_function("check_existing_products")
def check_existing_products(fhr_dir):
    """Check what products already exist for this forecast hour"""
    if not fhr_dir.exists():
        return []

    # Check both flat structure and nested structure (for compatibility)
    existing_files = list(fhr_dir.glob("*_REFACTORED.png"))
    existing_files.extend(list(fhr_dir.glob("**/*_REFACTORED.png")))  # Recursive search

    existing_products = []

    for file_path in existing_files:
        # Extract product name from filename like "sbcape_f01_REFACTORED.png"
        name_parts = file_path.stem.split("_")
        if len(name_parts) >= 3 and name_parts[-1] == "REFACTORED":
            product_name = "_".join(name_parts[:-2])  # Remove "_f01_REFACTORED"
            existing_products.append(product_name)

    return list(set(existing_products))  # Remove duplicates


def get_missing_products(fhr_dir, all_available_products):
    """Get list of products that haven't been generated yet"""
    existing = check_existing_products(fhr_dir)
    missing = [product for product in all_available_products if product not in existing]
    return missing, existing


def get_available_products():
    """Get all available products from the field registry"""
    try:
        from field_registry import FieldRegistry

        registry = FieldRegistry()
        all_fields = registry.get_all_fields()
        return list(all_fields.keys())
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Could not load field registry: {e}")
        return []


@profile_function("process_forecast_hour_smart")
def process_forecast_hour_smart(
    cycle,
    forecast_hour,
    output_dirs,
    categories=None,
    fields=None,
    force_reprocess=False,
):
    """Process a single forecast hour with smart duplicate detection and shared GRIB downloads"""
    logger = logging.getLogger(__name__)

    fhr_dir = get_forecast_hour_dir(output_dirs["run"], forecast_hour)

    # Check what's already been processed
    if not force_reprocess:
        all_products = get_available_products()
        if fields:
            all_products = fields
        elif categories:
            # Filter products by categories if specified
            from field_registry import FieldRegistry

            registry = FieldRegistry()
            filtered_products = []
            for category in categories:
                cat_products = registry.get_fields_by_category(category)
                filtered_products.extend(cat_products.keys())
            all_products = filtered_products

        missing_products, existing_products = get_missing_products(
            fhr_dir, all_products
        )

        if not missing_products:
            logger.info(
                f"✓ F{forecast_hour:02d} already complete ({len(existing_products)} products)"
            )
            return {
                "success": True,
                "forecast_hour": forecast_hour,
                "skipped": True,
                "existing_count": len(existing_products),
            }

        logger.info(
            f"F{forecast_hour:02d}: {len(existing_products)} existing, {len(missing_products)} missing"
        )

    # STEP 1: Download GRIB files to forecast hour directory (shared by all categories)
    logger.info(f"📥 Downloading GRIB files for F{forecast_hour:02d}")
    grib_download_success = download_grib_to_forecast_dir(cycle, forecast_hour, fhr_dir)

    if not grib_download_success:
        logger.warning(
            f"⚠️ GRIB download failed for F{forecast_hour:02d}, attempting processing anyway"
        )

    # STEP 2: Process all categories using shared GRIB files
    cmd = [
        "python",
        "tools/process_single_hour.py",
        cycle,
        str(forecast_hour),
        "--output-dir",
        str(fhr_dir),
        "--use-local-grib",
    ]

    if categories:
        cmd.extend(["--categories", ",".join(categories)])
    if fields:
        cmd.extend(["--fields", ",".join(fields)])

    logger.info(f"Processing F{forecast_hour:02d}: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=Path(__file__).parent,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            final_products = check_existing_products(fhr_dir)
            logger.info(
                f"✅ F{forecast_hour:02d} completed in {duration:.1f}s ({len(final_products)} products)"
            )
            return {
                "success": True,
                "forecast_hour": forecast_hour,
                "duration": duration,
                "product_count": len(final_products),
                "skipped": False,
            }
        else:
            logger.error(
                f"❌ F{forecast_hour:02d} failed with return code {result.returncode}"
            )
            if result.stderr:
                logger.error(f"Error: {result.stderr[-500:]}")
            return {
                "success": False,
                "forecast_hour": forecast_hour,
                "error": f"Return code {result.returncode}",
            }
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ F{forecast_hour:02d} timed out")
        return {"success": False, "forecast_hour": forecast_hour, "error": "Timeout"}
    except Exception as e:
        logger.error(f"💥 F{forecast_hour:02d} crashed: {str(e)}")
        return {"success": False, "forecast_hour": forecast_hour, "error": str(e)}


def download_grib_to_forecast_dir(cycle, forecast_hour, fhr_dir):
    """Download GRIB files directly to forecast hour directory"""
    logger = logging.getLogger(__name__)

    try:
        # Import here to avoid circular imports
        from hrrr_processor_refactored import HRRRProcessor

        processor = HRRRProcessor()

        # Download both required GRIB file types to forecast hour directory
        file_types = ["wrfprs", "wrfsfc"]
        downloaded_any = False

        for file_type in file_types:
            try:
                processor.download_hrrr_file(cycle, forecast_hour, fhr_dir, file_type)
                downloaded_any = True
                logger.debug(f"📥 Downloaded {file_type} for F{forecast_hour:02d}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to download {file_type} for F{forecast_hour:02d}: {e}"
                )

        return downloaded_any

    except Exception as e:
        logger.error(f"❌ GRIB download failed for F{forecast_hour:02d}: {e}")
        return False


def process_forecast_hour_worker(args):
    """Worker function for multiprocessing"""
    cycle, forecast_hour, output_dirs, categories, fields, force_reprocess = args
    return process_forecast_hour_smart(
        cycle, forecast_hour, output_dirs, categories, fields, force_reprocess
    )


def process_model_run(
    model,
    date,
    hour,
    forecast_hours,
    categories=None,
    fields=None,
    max_workers=1,
    force_reprocess=False,
    profiler=None,
):
    """Process an entire model run with smart organization"""
    logger = logging.getLogger(__name__)

    with profile_phase("model_run_setup"):
        # Create output structure
        output_dirs = create_output_structure(model, date, hour)
        cycle = f"{date}{hour:02d}"

    logger.info(f"Processing {model.upper()} model run")
    logger.info(f"Date: {date}, Hour: {hour:02d}Z")
    logger.info(
        f"Forecast hours: F{min(forecast_hours):02d}-F{max(forecast_hours):02d} ({len(forecast_hours)} total)"
    )
    logger.info(f"Output directory: {output_dirs['run']}")
    logger.info(f"Categories: {categories if categories else 'all'}")
    if fields:
        logger.info(f"Fields: {fields}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Force reprocess: {force_reprocess}")

    # Check system resources
    sys_memory = check_system_memory()
    if sys_memory:
        logger.info(
            f"System memory: {sys_memory['used_mb']:.0f}MB/{sys_memory['total_mb']:.0f}MB ({sys_memory['percent']:.1f}%)"
        )

    cpu_count = mp.cpu_count()
    logger.info(f"Available CPUs: {cpu_count}")

    with profile_phase("pipeline_processing"):
        if max_workers > 1:
            # Parallel processing (unchanged for now)
            logger.info(f"Starting parallel processing with {max_workers} workers")

            # Prepare arguments
            args_list = [
                (cycle, fhr, output_dirs, categories, fields, force_reprocess)
                for fhr in forecast_hours
            ]

            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_fhr = {
                    executor.submit(process_forecast_hour_worker, args): args[1]
                    for args in args_list
                }

                for future in as_completed(future_to_fhr):
                    fhr = future_to_fhr[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result["success"]:
                            if result.get("skipped"):
                                logger.info(
                                    f"⚡ F{result['forecast_hour']:02d} skipped (already complete)"
                                )
                            else:
                                logger.info(
                                    f"✅ F{result['forecast_hour']:02d} completed ({result.get('product_count', 0)} products)"
                                )
                        else:
                            logger.error(
                                f"❌ F{result['forecast_hour']:02d} failed: {result['error']}"
                            )

                    except Exception as e:
                        logger.error(f"💥 F{fhr:02d} crashed: {str(e)}")
                        results.append(
                            {"success": False, "forecast_hour": fhr, "error": str(e)}
                        )
        else:
            # Pipeline processing (NEW DEFAULT)
            logger.info(
                "🚀 Starting pipeline processing (download + process parallelism)"
            )
            results = []

            # Download queue and completed tracking
            download_queue = queue.Queue()
            completed_downloads = set()
            download_errors = set()

            def download_worker():
                """Background download worker"""
                while True:
                    try:
                        fhr = download_queue.get(timeout=1.0)
                        if fhr is None:  # Shutdown signal
                            break

                        # TEMP: Skip background downloads until GRIB integration is complete
                        # success = download_forecast_hour(cycle, fhr, profiler)
                        # if success:
                        #     completed_downloads.add(fhr)
                        # else:
                        #     download_errors.add(fhr)

                        # For now, just mark as completed so processing continues
                        completed_downloads.add(fhr)

                        download_queue.task_done()

                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"💥 Download worker error: {e}")
                        break

            # Start download worker thread
            download_thread = threading.Thread(target=download_worker, daemon=True)
            download_thread.start()

            # Queue first download
            if forecast_hours:
                download_queue.put(forecast_hours[0])
                logger.info(f"📥 Queued download: F{forecast_hours[0]:02d}")

            try:
                for i, fhr in enumerate(forecast_hours, 1):
                    # Queue next download (if not last)
                    if i < len(forecast_hours):
                        next_fhr = forecast_hours[i]
                        download_queue.put(next_fhr)
                        logger.info(f"📥 Queued download: F{next_fhr:02d}")

                    # Wait for current forecast hour download
                    logger.info(
                        f"\n📅 Processing F{fhr:02d} ({i}/{len(forecast_hours)})"
                    )

                    # Wait for download (with timeout)
                    timeout = 300  # 5 minute timeout
                    start_wait = time.time()

                    while fhr not in completed_downloads and fhr not in download_errors:
                        if time.time() - start_wait > timeout:
                            logger.error(f"⏰ Timeout waiting for F{fhr:02d} download")
                            break
                        time.sleep(0.5)

                    # Process forecast hour
                    if fhr in download_errors:
                        logger.warning(
                            f"⚠️ Processing F{fhr:02d} despite download issues"
                        )

                    result = process_forecast_hour_smart(
                        cycle, fhr, output_dirs, categories, fields, force_reprocess
                    )
                    results.append(result)

                    if result["success"]:
                        if result.get("skipped"):
                            logger.info(f"⚡ F{fhr:02d} skipped (already complete)")
                        else:
                            logger.info(
                                f"✅ F{fhr:02d} completed ({result.get('product_count', 0)} products)"
                            )
                    else:
                        logger.error(f"❌ F{fhr:02d} failed: {result['error']}")

                    # Progress update
                    percent = (i / len(forecast_hours)) * 100
                    successful = sum(1 for r in results if r["success"])
                    logger.info(
                        f"Progress: {i}/{len(forecast_hours)} ({percent:.1f}%) - ✅{successful}"
                    )

                    # Add profiling metrics
                    if profiler:
                        profiler.add_metric(
                            f"forecast_hour_f{fhr:02d}_success", result["success"]
                        )
                        if result["success"] and "duration" in result:
                            profiler.add_metric(
                                f"forecast_hour_f{fhr:02d}_duration", result["duration"]
                            )

            finally:
                # Shutdown download worker
                download_queue.put(None)  # Shutdown signal
                download_thread.join(timeout=5.0)

    # Final summary
    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r.get("skipped", False))
    failed = len(results) - successful

    logger.info(f"\n🎉 Model run processing complete!")
    logger.info(f"Total forecast hours: {len(forecast_hours)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped (already complete): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(forecast_hours)*100:.1f}%")
    logger.info(f"Output directory: {output_dirs['run']}")

    # Show failed forecast hours if any
    failed_hours = [r["forecast_hour"] for r in results if not r["success"]]
    if failed_hours:
        logger.warning(
            f"Failed forecast hours: F{', F'.join([f'{h:02d}' for h in sorted(failed_hours)])}"
        )

    return results


def get_latest_cycle():
    """Get the most recent HRRR cycle that should be available"""
    now = datetime.utcnow()

    print(f"🕐 Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # HRRR runs every hour - check current hour first, then work backwards
    # Start from current hour and go back to account for processing delays
    for hours_back in range(0, 12):  # Check up to 12 hours back
        cycle_time = now - timedelta(hours=hours_back)
        cycle_str = cycle_time.strftime("%Y%m%d%H")

        if hours_back == 0:
            print(
                f"🔍 Checking current cycle: {cycle_time.strftime('%Y-%m-%d %HZ')} (current hour)"
            )
        else:
            print(
                f"🔍 Checking cycle: {cycle_time.strftime('%Y-%m-%d %HZ')} ({hours_back}h ago)"
            )

        # Quick check if this cycle might exist
        if check_cycle_availability(cycle_str):
            print(
                f"✅ Found latest available cycle: {cycle_time.strftime('%Y-%m-%d %HZ UTC')}"
            )
            return cycle_str, cycle_time
        else:
            if hours_back == 0:
                print(
                    f"❌ Current cycle {cycle_time.strftime('%HZ')} not yet available"
                )
            else:
                print(f"❌ Cycle {cycle_time.strftime('%HZ')} not available")

    # Fallback
    fallback_time = now - timedelta(hours=6)
    print(f"⚠️ Using fallback cycle: {fallback_time.strftime('%Y-%m-%d %HZ UTC')}")
    return fallback_time.strftime("%Y%m%d%H"), fallback_time


@profile_function("check_cycle_availability")
def check_cycle_availability(cycle):
    """Check if a cycle is available by testing F00 file"""
    try:
        cycle_dt = datetime.strptime(cycle, "%Y%m%d%H")
        date_str = cycle_dt.strftime("%Y%m%d")
        filename = f"hrrr.t{cycle[-2:]}z.wrfprsf00.grib2"
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}"

        request = urllib.request.Request(url)
        request.get_method = lambda: "HEAD"
        response = urllib.request.urlopen(request, timeout=10)
        return response.getcode() == 200
    except:
        return False


def get_grib_download_dir(cycle):
    """Get centralized GRIB download directory"""
    cycle_dt = datetime.strptime(cycle, "%Y%m%d%H")
    date_str = cycle_dt.strftime("%Y%m%d")
    hour = cycle[-2:]

    # Create centralized GRIB directory
    grib_dir = Path("grib_files") / date_str / f"{hour}z"
    grib_dir.mkdir(parents=True, exist_ok=True)

    return grib_dir


def check_grib_files_exist(cycle, forecast_hour):
    """Check if GRIB files already exist for forecast hour"""
    grib_dir = get_grib_download_dir(cycle)
    hour = cycle[-2:]

    # Check for both required file types
    required_files = ["wrfprs", "wrfsfc"]
    existing_files = []

    for file_type in required_files:
        filename = f"hrrr.t{hour}z.{file_type}f{forecast_hour:02d}.grib2"
        file_path = grib_dir / filename
        if file_path.exists():
            existing_files.append(file_type)

    return len(existing_files) > 0, existing_files


def download_grib_files(cycle, forecast_hour, profiler=None):
    """Download GRIB files to centralized location"""
    logger = logging.getLogger(__name__)

    grib_dir = get_grib_download_dir(cycle)
    hour = cycle[-2:]

    # Check what files we need
    file_types = ["wrfprs", "wrfsfc"]
    files_to_download = []

    for file_type in file_types:
        filename = f"hrrr.t{hour}z.{file_type}f{forecast_hour:02d}.grib2"
        file_path = grib_dir / filename
        if not file_path.exists():
            files_to_download.append((file_type, filename, file_path))

    if not files_to_download:
        logger.debug(f"📁 F{forecast_hour:02d} GRIB files already exist in {grib_dir}")
        if profiler:
            profiler.add_metric(f"download_f{forecast_hour:02d}_duration", 0.0)
            profiler.add_metric(f"download_f{forecast_hour:02d}_cached", True)
        return True

    logger.info(
        f"📥 Downloading {len(files_to_download)} GRIB files for F{forecast_hour:02d}"
    )
    download_start = time.time()

    try:
        # Check availability first
        available_files = check_forecast_hour_availability(
            cycle, forecast_hour, [f[0] for f in files_to_download]
        )

        if not available_files:
            logger.warning(f"⚠️  F{forecast_hour:02d} not yet available on server")
            if profiler:
                profiler.add_metric(f"download_f{forecast_hour:02d}_available", False)
            return False

        # Download each required file
        downloaded_count = 0
        for file_type, filename, file_path in files_to_download:
            if file_type in available_files:
                try:
                    # Create URL for download
                    cycle_dt = datetime.strptime(cycle, "%Y%m%d%H")
                    date_str = cycle_dt.strftime("%Y%m%d")
                    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}"

                    logger.info(f"📥 Downloading {filename}...")

                    # Download with progress
                    response = urllib.request.urlopen(url, timeout=120)

                    # Write to temporary file first, then move (atomic operation)
                    temp_file = file_path.with_suffix(".tmp")

                    with open(temp_file, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)

                    # Move to final location
                    temp_file.rename(file_path)
                    downloaded_count += 1

                    # Check file size
                    file_size_mb = file_path.stat().st_size / 1024 / 1024
                    logger.info(f"✅ Downloaded {filename} ({file_size_mb:.1f}MB)")

                except Exception as e:
                    logger.error(f"❌ Failed to download {filename}: {e}")
                    # Clean up partial file
                    if temp_file.exists():
                        temp_file.unlink()
                    continue

        download_time = time.time() - download_start

        if downloaded_count > 0:
            logger.info(
                f"📥 F{forecast_hour:02d} download completed: {downloaded_count} files in {download_time:.1f}s"
            )

        if profiler:
            profiler.add_metric(
                f"download_f{forecast_hour:02d}_duration", download_time
            )
            profiler.add_metric(
                f"download_f{forecast_hour:02d}_files_downloaded", downloaded_count
            )
            profiler.add_metric(f"download_f{forecast_hour:02d}_available", True)

        return downloaded_count > 0

    except Exception as e:
        logger.error(f"❌ F{forecast_hour:02d} download failed: {e}")
        if profiler:
            profiler.add_metric(f"download_f{forecast_hour:02d}_error", str(e))
        return False


def download_forecast_hour(cycle, forecast_hour, profiler=None):
    """Download GRIB files for a forecast hour (wrapper for centralized download)"""
    exists, existing_files = check_grib_files_exist(cycle, forecast_hour)

    if exists:
        # Files already exist, no download needed
        logger = logging.getLogger(__name__)
        logger.debug(
            f"📁 F{forecast_hour:02d} GRIB files already exist ({', '.join(existing_files)})"
        )
        if profiler:
            profiler.add_metric(f"download_f{forecast_hour:02d}_duration", 0.0)
            profiler.add_metric(f"download_f{forecast_hour:02d}_cached", True)
        return True

    # Download the files
    return download_grib_files(cycle, forecast_hour, profiler)


def check_forecast_hour_availability(
    cycle, forecast_hour, file_types=["wrfprs", "wrfsfc"]
):
    """Check if specific forecast hour is available"""
    cycle_dt = datetime.strptime(cycle, "%Y%m%d%H")
    date_str = cycle_dt.strftime("%Y%m%d")
    hour = cycle[-2:]

    available_files = []

    for file_type in file_types:
        filename = f"hrrr.t{hour}z.{file_type}f{forecast_hour:02d}.grib2"
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/{filename}"

        try:
            request = urllib.request.Request(url)
            request.get_method = lambda: "HEAD"
            response = urllib.request.urlopen(request, timeout=10)
            if response.getcode() == 200:
                available_files.append(file_type)
        except:
            continue

    return available_files


def get_expected_max_forecast_hour(cycle):
    """Get expected maximum forecast hour for this cycle"""
    cycle_dt = datetime.strptime(cycle, "%Y%m%d%H")
    hour = cycle_dt.hour

    # 6-hourly runs (00, 06, 12, 18Z) go to 48 hours
    if hour in [0, 6, 12, 18]:
        return 48
    else:
        return 18


def monitor_and_process_latest(
    categories=None,
    fields=None,
    workers=1,
    check_interval=30,
    force_reprocess=False,
    hour_range=None,
    max_hours=None,
):
    """Monitor for new forecast hours and process them as they become available"""
    logger = logging.getLogger(__name__)

    logger.info("🔄 Starting live monitoring mode")
    logger.info(f"Check interval: {check_interval} seconds")
    logger.info(f"Categories: {categories if categories else 'all'}")
    if fields:
        logger.info(f"Fields: {fields}")
    logger.info(f"Workers: {workers}")

    # Get latest cycle
    cycle, cycle_time = get_latest_cycle()
    date_str = cycle_time.strftime("%Y%m%d")
    hour = cycle_time.hour

    logger.info(f"📡 Monitoring cycle: {cycle_time.strftime('%Y-%m-%d %HZ UTC')}")

    # Create output structure
    output_dirs = create_output_structure("HRRR", date_str, hour)

    expected_max_fhr = get_expected_max_forecast_hour(cycle)

    if hour_range is not None:
        forecast_hours = [h for h in hour_range if h <= expected_max_fhr]
    elif max_hours is not None:
        forecast_hours = list(range(0, min(max_hours, expected_max_fhr) + 1))
    else:
        forecast_hours = list(range(0, expected_max_fhr + 1))

    logger.info(
        f"Target forecast hours: F{forecast_hours[0]:02d}-F{forecast_hours[-1]:02d}"
    )

    processed_hours = set()
    available_hours = set()
    consecutive_no_new = 0
    max_consecutive_no_new = 10  # Stop after 10 checks with no new data

    try:
        while True:
            logger.info(
                f"\n🔍 Checking for new forecast hours... ({datetime.now().strftime('%H:%M:%S')})"
            )

            new_hours_found = False

            # Check each forecast hour
            for fhr in forecast_hours:
                if fhr in processed_hours:
                    continue

                # Check if this forecast hour is available
                available_files = check_forecast_hour_availability(cycle, fhr)

                if available_files:
                    if fhr not in available_hours:
                        logger.info(
                            f"🆕 New forecast hour detected: F{fhr:02d} ({', '.join(available_files)} files)"
                        )
                        available_hours.add(fhr)
                        new_hours_found = True

                    # Check if we should process this hour
                    fhr_dir = get_forecast_hour_dir(output_dirs["run"], fhr)

                    if not force_reprocess:
                        # Check if already processed
                        all_products = get_available_products()
                        if fields:
                            all_products = fields
                        elif categories:
                            from field_registry import FieldRegistry

                            registry = FieldRegistry()
                            filtered_products = []
                            for category in categories:
                                cat_products = registry.get_fields_by_category(category)
                                filtered_products.extend(cat_products.keys())
                            all_products = filtered_products

                        missing_products, existing_products = get_missing_products(
                            fhr_dir, all_products
                        )

                        if not missing_products:
                            logger.info(f"⚡ F{fhr:02d} already complete, skipping")
                            processed_hours.add(fhr)
                            continue

                    # Process this forecast hour
                    logger.info(f"🚀 Processing F{fhr:02d}...")

                    result = process_forecast_hour_smart(
                        cycle, fhr, output_dirs, categories, fields, force_reprocess
                    )

                    if result["success"]:
                        if result.get("skipped"):
                            logger.info(f"⚡ F{fhr:02d} was already complete")
                        else:
                            logger.info(
                                f"✅ F{fhr:02d} processed successfully ({result.get('product_count', 0)} products)"
                            )
                        processed_hours.add(fhr)
                    else:
                        logger.error(f"❌ F{fhr:02d} failed: {result['error']}")
                        # Don't add to processed_hours so we can retry later

            # Update counters
            if new_hours_found:
                consecutive_no_new = 0
            else:
                consecutive_no_new += 1

            # Progress update
            total_available = len(available_hours)
            total_processed = len(processed_hours)

            logger.info(
                f"📊 Status: {total_processed}/{total_available} processed, checking F{min(forecast_hours):02d}-F{forecast_hours[-1]:02d}"
            )

            # Check if we're done
            if len(processed_hours) >= len(forecast_hours):
                logger.info(
                    f"🎉 All forecast hours processed! (F{forecast_hours[0]:02d}-F{forecast_hours[-1]:02d})"
                )
                break

            # Check if no new data for too long
            if consecutive_no_new >= max_consecutive_no_new:
                logger.warning(
                    f"⏰ No new forecast hours for {consecutive_no_new * check_interval / 60:.1f} minutes"
                )
                logger.info(
                    f"Stopping monitoring. Processed: F{', F'.join([f'{h:02d}' for h in sorted(processed_hours)])}"
                )
                break

            # Wait before next check
            logger.info(f"⏳ Waiting {check_interval} seconds for next check...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info(f"\n👋 Monitoring stopped by user")
        logger.info(
            f"Processed forecast hours: F{', F'.join([f'{h:02d}' for h in sorted(processed_hours)])}"
        )

    return list(processed_hours), date_str, hour, forecast_hours


def move_old_files():
    """Move old processing files to old_files directory"""
    logger = logging.getLogger(__name__)

    old_files_dir = Path("old_files")
    old_files_dir.mkdir(exist_ok=True)

    # Files/patterns to move
    patterns_to_move = [
        "hrrr_processed_*",
        "all_products_*",
        "memory_safe_*",
        "debug_output_*",
        "hrrr_debug_*",
        "single_hour_*",
        "parallel_output*",
    ]

    moved_count = 0
    for pattern in patterns_to_move:
        for item in Path(".").glob(pattern):
            if item.is_dir() or item.is_file():
                try:
                    dest = old_files_dir / item.name
                    if dest.exists():
                        # Add timestamp if destination exists
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest = old_files_dir / f"{item.stem}_{timestamp}{item.suffix}"

                    shutil.move(str(item), str(dest))
                    moved_count += 1
                    logger.debug(f"Moved {item} -> {dest}")
                except Exception as e:
                    logger.warning(f"Could not move {item}: {e}")

    if moved_count > 0:
        logger.info(f"📁 Moved {moved_count} old files to {old_files_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Smart HRRR processor with organized output and duplicate detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 20250602 17                    # Process 17Z run (F00-F18)
  %(prog)s 20250602 12 --max-hours 48     # Process 12Z run (F00-F48)
  %(prog)s 20250602 17 --categories severe,smoke  # Process specific categories
  %(prog)s 20250602 17 --workers 4        # Use 4 parallel workers
  %(prog)s 20250602 17 --force             # Reprocess everything
  %(prog)s --latest                       # Monitor and process latest model run
  %(prog)s --latest --categories smoke    # Monitor latest run, smoke products only
        """,
    )

    parser.add_argument(
        "date", nargs="?", help="Date in YYYYMMDD format (or use --latest)"
    )
    parser.add_argument("hour", type=int, nargs="?", help="Model run hour (0-23)")
    parser.add_argument(
        "--latest", action="store_true", help="Monitor and process latest model run"
    )
    parser.add_argument("--model", default="HRRR", help="Model name (default: HRRR)")
    parser.add_argument(
        "--max-hours",
        type=int,
        help="Maximum forecast hours (auto-detected by default)",
    )
    parser.add_argument("--hours", help="Forecast hour range (e.g. 6-12 or 0,1,2)")
    parser.add_argument("--categories", help="Categories to process (comma-separated)")
    parser.add_argument("--fields", help="Specific fields to process (comma-separated)")
    parser.add_argument("--filter", help="Saved filter name")
    parser.add_argument("--workflow", help="Workflow preset name")
    parser.add_argument(
        "--gifs",
        "--gif",
        action="store_true",
        dest="gifs",
        help="Create GIFs after processing",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocess existing files"
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument(
        "--cleanup", action="store_true", help="Move old files before processing"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Check interval for latest mode (seconds, default: 30)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=1.0,
        help="Profiling sample interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--profile-output", help="Export detailed profiling report to file"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.latest:
        if args.date or args.hour:
            parser.error("Cannot specify date/hour with --latest mode")
    else:
        if not args.date or args.hour is None:
            parser.error("Must specify date and hour (or use --latest)")

    # Setup logging first (before we know the exact output dir for latest mode)
    if args.latest:
        logger = setup_logging(debug=args.debug)
    else:
        # Create output structure for logging
        output_dirs = create_output_structure(args.model, args.date, args.hour)
        logger = setup_logging(debug=args.debug, output_dir=output_dirs["run"])

    try:
        # Initialize profiler if requested
        profiler = None
        if args.profile:
            profiler = HRRRProfiler(enabled=True, sample_interval=args.profile_interval)
            profiler.start()
            logger.info(
                f"🔍 Performance profiling enabled (interval: {args.profile_interval}s)"
            )

        # Cleanup old files if requested
        if args.cleanup:
            move_old_files()

        # Parse categories
        categories = None
        if args.categories:
            categories = [cat.strip() for cat in args.categories.split(",")]

        # Parse fields
        fields = None
        if args.fields:
            fields = [f.strip() for f in args.fields.split(",")]

        # Load filter if specified
        if args.filter:
            if FILTER_FILE.exists():
                with open(FILTER_FILE, "r") as f:
                    filters = json.load(f)
                if args.filter in filters:
                    filt = filters[args.filter]
                    if not fields:
                        fields = filt.get("fields")
                    if not categories:
                        categories = filt.get("categories")
                else:
                    logger.error(f"Filter not found: {args.filter}")
            else:
                logger.error("No filter file available")

        workflow_gifs = False
        if args.workflow:
            if WORKFLOW_FILE.exists():
                with open(WORKFLOW_FILE, "r") as f:
                    workflows = json.load(f)
                if args.workflow in workflows:
                    wf = workflows[args.workflow]
                    if not fields and wf.get("fields"):
                        fields = wf["fields"]
                    if not categories and wf.get("categories"):
                        categories = wf["categories"]
                    if wf.get("generate_gifs"):
                        workflow_gifs = True
                else:
                    logger.error(f"Workflow not found: {args.workflow}")
            else:
                logger.error("No workflow file available")

        generate_gifs = args.gifs or workflow_gifs

        # Validate workers
        max_cpu_workers = mp.cpu_count()
        workers = min(args.workers, max_cpu_workers) if args.workers > 0 else 1

        if args.workers > max_cpu_workers:
            logger.warning(
                f"Requested {args.workers} workers, using {workers} (CPU limit)"
            )

        if args.latest:
            # Live monitoring mode
            logger.info(f"Starting smart HRRR processor in LIVE MODE")

            hour_range = parse_hour_range(args.hours)

            processed_hours, live_date, live_hour, used_hours = (
                monitor_and_process_latest(
                    categories=categories,
                    fields=fields,
                    workers=workers,
                    check_interval=args.check_interval,
                    force_reprocess=args.force,
                    hour_range=hour_range,
                    max_hours=args.max_hours,
                )
            )

            logger.info(
                f"🎯 Live monitoring result: {len(processed_hours)} forecast hours processed"
            )

            if generate_gifs and processed_hours:
                from tools.create_gifs import create_gifs_for_model_run

                create_gifs_for_model_run(
                    ".",
                    live_date,
                    f"{live_hour:02d}z",
                    max_hours=max(used_hours),
                    categories=categories,
                )
        else:
            # Regular mode - process specific date/hour
            logger.info(f"Starting smart HRRR processor")

            # Determine forecast hours based on model run type
            hour_range = parse_hour_range(args.hours)

            if hour_range is not None:
                forecast_hours = hour_range
                max_hours = max(forecast_hours)
            else:
                if args.max_hours is None:
                    # Auto-detect: 6-hourly runs go to 48 hours, others go to 18
                    if args.hour in [0, 6, 12, 18]:
                        max_hours = 48
                    else:
                        max_hours = 18
                else:
                    max_hours = args.max_hours
                forecast_hours = list(range(0, max_hours + 1))

            # Process model run
            results = process_model_run(
                model=args.model,
                date=args.date,
                hour=args.hour,
                forecast_hours=forecast_hours,
                categories=categories,
                fields=fields,
                max_workers=workers,
                force_reprocess=args.force,
                profiler=profiler,
            )

            # Final success check
            successful = sum(1 for r in results if r["success"])
            logger.info(
                f"🎯 Final result: {successful}/{len(results)} forecast hours processed successfully"
            )

            if generate_gifs:
                from tools.create_gifs import create_gifs_for_model_run

                create_gifs_for_model_run(
                    ".",
                    args.date,
                    f"{args.hour:02d}z",
                    max_hours=max(forecast_hours),
                    categories=categories,
                )

            # Add final profiling metrics
            if profiler:
                profiler.add_metric("total_forecast_hours", len(results))
                profiler.add_metric("successful_forecast_hours", successful)
                profiler.add_metric(
                    "success_rate_percent", (successful / len(results)) * 100
                )

    except KeyboardInterrupt:
        logger.info("Processing cancelled by user")
        if profiler:
            profiler.stop()
            profiler.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}")
        if profiler:
            profiler.stop()
            profiler.print_summary()
        sys.exit(1)
    finally:
        # Finalize profiling
        if profiler:
            profiler.stop()
            profiler.print_summary()

            # Show bottlenecks
            bottlenecks = profiler.get_bottlenecks(top_n=5)
            if bottlenecks:
                logger.info("🐌 Top Performance Bottlenecks:")
                for i, bottleneck in enumerate(bottlenecks, 1):
                    logger.info(
                        f"  {i}. {bottleneck['name']} ({bottleneck['type']}): "
                        f"{bottleneck['time_seconds']:.2f}s ({bottleneck['percent_of_total']:.1f}%)"
                    )

            # Export detailed report if requested
            if args.profile_output:
                output_file = Path(args.profile_output)
                profiler.export_detailed_report(output_file)
            elif args.profile:
                # Auto-export with timestamp
                profiler.export_detailed_report()


if __name__ == "__main__":
    main()
