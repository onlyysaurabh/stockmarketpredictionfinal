#!/usr/bin/env python3
# filepath: /home/skylap/Downloads/stockmarketprediction/train-model/train.py
import argparse
import subprocess
import sys
import os
import concurrent.futures
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_jobs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_trainer")

# Default max workers - adjust based on system resources
DEFAULT_MAX_WORKERS = 4

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    
    # Job scheduling parameters
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS,
                      help=f'Maximum number of concurrent training jobs (default: {DEFAULT_MAX_WORKERS})')
    
    # Model selection options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model', type=str,
                      choices=['xgboost', 'svm', 'lstm', 'arima'],
                      help='Single model type to train')
    model_group.add_argument('--models', type=str,
                      help='Comma-separated list of model types (e.g., xgboost,lstm,arima)')
    model_group.add_argument('--all-models', action='store_true',
                      help='Train all available model types')
    
    # Stock symbol selection options
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument('--symbol', type=str, 
                      help='Single stock symbol to train on (e.g., AAPL)')
    symbol_group.add_argument('--symbols', type=str,
                      help='Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)')
    symbol_group.add_argument('--symbols-file', type=str,
                      help='Path to a file containing stock symbols (one per line)')
    
    # Date range parameters
    parser.add_argument('--start-date', type=str, 
                      help='Start date for training data (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                      help='End date for training data (YYYY-MM-DD format)')
    
    # Batch job configuration
    parser.add_argument('--config', type=str,
                      help='Path to JSON configuration file for batch training jobs')
    
    # Pass through additional arguments
    parser.add_argument('extra_args', nargs='*',
                      help='Additional arguments to pass to the model training script')
    
    return parser.parse_args()

def load_symbols_from_file(file_path):
    """Load stock symbols from a file (one symbol per line)."""
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        return symbols
    except Exception as e:
        logger.error(f"Failed to load symbols from file {file_path}: {e}")
        return []

def load_config_file(config_path):
    """Load job configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return None

def get_model_types(args):
    """Determine which model types to train based on arguments."""
    available_models = ['xgboost', 'svm', 'lstm', 'arima']
    
    if args.all_models:
        return available_models
    elif args.models:
        models = [m.strip().lower() for m in args.models.split(',')]
        # Validate models
        valid_models = [m for m in models if m in available_models]
        if len(valid_models) != len(models):
            logger.warning(f"Skipping unknown model types")
        return valid_models
    elif args.model:
        return [args.model]
    else:
        # Default to xgboost if not specified
        logger.info("No model type specified, defaulting to xgboost")
        return ['xgboost']

def get_stock_symbols(args):
    """Determine which stock symbols to train on based on arguments."""
    if args.symbol:
        return [args.symbol.upper()]
    elif args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(',')]
    elif args.symbols_file:
        return load_symbols_from_file(args.symbols_file)
    else:
        # Try to use default symbols file if it exists
        default_file = "stock_symbols.csv"
        if os.path.exists(default_file):
            import pandas as pd
            try:
                df = pd.read_csv(default_file)
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].tolist()
                    logger.info(f"Using {len(symbols)} symbols from default file: {default_file}")
                    return symbols
            except Exception as e:
                logger.warning(f"Failed to load symbols from default file: {e}")
        
        logger.error("No stock symbols specified")
        return []

def create_training_jobs(args, models, symbols):
    """Create a list of training job configurations."""
    jobs = []
    
    for model in models:
        for symbol in symbols:
            job = {
                'model': model,
                'symbol': symbol,
                'start_date': args.start_date,
                'end_date': args.end_date,
                'extra_args': args.extra_args
            }
            jobs.append(job)
    
    return jobs

def create_jobs_from_config(config):
    """Create jobs from a configuration dictionary."""
    jobs = []
    
    # Global default settings
    default_start_date = config.get('start_date', None)
    default_end_date = config.get('end_date', None)
    
    # Process each job in the config
    for job_config in config.get('jobs', []):
        model = job_config.get('model')
        if not model:
            logger.warning("Skipping job with no model specified")
            continue
            
        # Get symbols for this job
        symbols = job_config.get('symbols', [])
        symbol = job_config.get('symbol')
        symbols_file = job_config.get('symbols_file')
        
        if symbol:
            job_symbols = [symbol.upper()]
        elif symbols:
            job_symbols = [s.strip().upper() for s in symbols]
        elif symbols_file:
            job_symbols = load_symbols_from_file(symbols_file)
        else:
            logger.warning(f"No symbols specified for {model} job, skipping")
            continue
            
        # Get job parameters
        start_date = job_config.get('start_date', default_start_date)
        end_date = job_config.get('end_date', default_end_date)
        extra_args = job_config.get('extra_args', [])
        
        # Create a job for each symbol
        for symbol in job_symbols:
            job = {
                'model': model,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'extra_args': extra_args
            }
            jobs.append(job)
            
    return jobs

def execute_job(job):
    """Execute a single training job."""
    model = job['model']
    symbol = job['symbol']
    start_date = job['start_date']
    end_date = job['end_date']
    extra_args = job['extra_args']
    
    logger.info(f"Starting job: model={model}, symbol={symbol}")
    
    script_path = os.path.join(os.path.dirname(__file__), f"train-{model}.py")
    if not os.path.exists(script_path):
        logger.error(f"Training script not found: {script_path}")
        return {
            'status': 'failed',
            'model': model,
            'symbol': symbol,
            'error': f"Script not found: {script_path}"
        }
        
    cmd = [sys.executable, script_path]
    
    # Add symbol
    cmd.extend(["--symbol", symbol])
    
    # Add dates if provided
    if start_date:
        cmd.extend(["--start-date", start_date])
        
    if end_date:
        cmd.extend(["--end-date", end_date])
        
    # Add extra arguments
    cmd.extend(extra_args)
    
    # Execute the command
    start_time = time.time()
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log stdout and stderr even on success
        if process.stdout:
            logger.info(f"Job stdout: model={model}, symbol={symbol}\n{process.stdout.strip()}")
        if process.stderr:
            logger.warning(f"Job stderr: model={model}, symbol={symbol}\n{process.stderr.strip()}")
            
        logger.info(f"Job completed: model={model}, symbol={symbol}")
        
        end_time = time.time()
        return {
            'status': 'success',
            'model': model,
            'symbol': symbol,
            'duration': end_time - start_time
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in job model={model}, symbol={symbol}: {e}")
        # Log stdout and stderr on error
        if e.stdout:
            logger.error(f"Job stdout on error: model={model}, symbol={symbol}\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Job stderr on error: model={model}, symbol={symbol}\n{e.stderr.strip()}")
            
        end_time = time.time()
        return {
            'status': 'failed',
            'model': model,
            'symbol': symbol,
            'error': str(e),
            'duration': end_time - start_time
        }
    except Exception as e:
        logger.error(f"Exception in job model={model}, symbol={symbol}: {e}")
        
        end_time = time.time()
        return {
            'status': 'failed',
            'model': model,
            'symbol': symbol,
            'error': str(e),
            'duration': end_time - start_time
        }

def main():
    """Run the training jobs based on arguments."""
    args = parse_args()
    
    # Handle configuration file if provided
    if args.config:
        config = load_config_file(args.config)
        if not config:
            logger.error("Failed to load configuration file.")
            sys.exit(1)
            
        jobs = create_jobs_from_config(config)
        max_workers = config.get('max_workers', DEFAULT_MAX_WORKERS)
    else:
        # Determine models and symbols from arguments
        models = get_model_types(args)
        if not models:
            logger.error("No valid model types specified")
            sys.exit(1)
            
        symbols = get_stock_symbols(args)
        if not symbols:
            logger.error("No stock symbols specified")
            sys.exit(1)
            
        jobs = create_training_jobs(args, models, symbols)
        max_workers = args.max_workers
    
    if not jobs:
        logger.error("No valid training jobs to execute")
        sys.exit(1)
        
    logger.info(f"Prepared {len(jobs)} training jobs with {max_workers} max workers")
    
    # Record start time
    start_time = time.time()
    
    # Execute jobs in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Job failed: {e}")
                results.append({
                    'status': 'failed',
                    'model': job['model'],
                    'symbol': job['symbol'],
                    'error': str(e)
                })
    
    # Record end time
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Summarize results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Training Summary:")
    logger.info(f"  Total jobs: {len(jobs)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info(f"  Total duration: {total_duration:.2f} seconds")
    
    if failed:
        logger.info("Failed jobs:")
        for job in failed:
            logger.info(f"  Model: {job['model']}, Symbol: {job['symbol']}, Error: {job.get('error', 'Unknown error')}")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
