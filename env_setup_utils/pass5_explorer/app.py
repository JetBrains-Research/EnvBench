from flask import Flask, render_template, request, jsonify
from huggingface_hub import HfApi, hf_hub_download, HfFileSystem
import json
import os
import logging
from collections import defaultdict
import plotly.graph_objects as go
import plotly.utils
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import tempfile
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# HuggingFace API configuration
DATASET_NAME = "JetBrains-Research/envbench-rl-trajectories"
hf_api = HfApi()
fs = HfFileSystem()

# Global file list - populated once at startup
_results_files = None
CACHE_FILE = Path(__file__).resolve().parent / "results_files_cache.json"

def walk_dataset_for_results():
    """Walk the dataset directory looking for results.jsonl files, avoiding trajectories folders"""
    global _results_files
    
    # Check if we have cached results
    if _results_files is not None:
        return _results_files
    
    # Try to load from cache file first
    if os.path.exists(CACHE_FILE):
        try:
            logger.info(f"Loading results files from cache: {CACHE_FILE}")
            with open(CACHE_FILE, 'r') as f:
                _results_files = json.load(f)
            logger.info(f"Loaded {len(_results_files)} results files from cache")
            return _results_files
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # If no cache, walk the dataset
    logger.info("No cache found, walking dataset...")
    
    try:
        logger.info(f"Walking dataset {DATASET_NAME} to find results.jsonl files...")
        
        # Try different path formats
        possible_paths = [
            f"datasets/{DATASET_NAME}",
            DATASET_NAME,
            f"datasets/{DATASET_NAME}/main",
            f"{DATASET_NAME}/main"
        ]
        
        dataset_path = None
        for path in possible_paths:
            try:
                logger.info(f"Trying path: {path}")
                items = fs.ls(path, detail=True)
                logger.info(f"Successfully listed {len(items)} items in {path}")
                dataset_path = path
                break
            except Exception as e:
                logger.warning(f"Failed to list {path}: {e}")
        
        if dataset_path is None:
            logger.error("Could not find valid dataset path")
            return []
        
        def walk_directory(path):
            """Recursively walk directory, avoiding trajectories folders"""
            results = []
            try:
                logger.info(f"Walking directory: {path}")
                items = fs.ls(path, detail=True)
                logger.info(f"Found {len(items)} items in {path}")
                
                for item in items:
                    item_path = item['name']
                    item_type = item['type']
                    
                    logger.debug(f"Item: {item_path} (type: {item_type})")
                    
                    # Skip trajectories folders
                    if item_type == 'directory' and 'trajectories' == item_path.split('/')[-1]:
                        logger.info(f"Skipping trajectories folder: {item_path}")
                        continue
                    
                    if item_type == 'file' and item_path.endswith('results.jsonl'):
                        logger.info(f"Found results.jsonl: {item_path}")
                        # Remove dataset prefix for cleaner paths
                        clean_path = item_path.replace(f"{dataset_path}/", "")
                        # Extract date from filename
                        import re
                        from datetime import datetime
                        
                        mtime = ''
                        # Try multiple date patterns
                        patterns = [
                            (r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', '%Y-%m-%d_%H-%M-%S'),  # 2025-07-18_23-09-01
                            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),  # 2025-07-18
                            (r'(\d{8})', '%Y%m%d'),  # 20250718
                        ]
                        
                        for pattern, format_str in patterns:
                            date_match = re.search(pattern, clean_path)
                            if date_match:
                                try:
                                    date_str = date_match.group(1)
                                    dt = datetime.strptime(date_str, format_str)
                                    mtime = dt.isoformat()
                                    logger.debug(f"Extracted date from {clean_path} using pattern {pattern}: {mtime}")
                                    break
                                except Exception as e:
                                    logger.debug(f"Failed to parse date {date_str} with format {format_str}: {e}")
                                    continue
                        
                        if not mtime:
                            logger.warning(f"No date pattern found in {clean_path}")
                            # Use a very old date for files without dates so they sort to the end
                            mtime = '1900-01-01T00:00:00'
                        
                        logger.info(f"Final mtime for {clean_path}: {mtime}")
                        
                        results.append({
                            'path': clean_path,
                            'size': item.get('size', 0),
                            'last_modified': mtime
                        })
                    elif item_type == 'directory':
                        logger.info(f"Entering directory: {item_path}")
                        # Recursively walk subdirectories
                        sub_results = walk_directory(item_path)
                        results.extend(sub_results)
                        
            except Exception as e:
                logger.error(f"Error walking {path}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            return results
        
        _results_files = walk_directory(dataset_path)
        logger.info(f"Found {len(_results_files)} results.jsonl files in dataset")
        
        # Sort by newest first (last_modified descending)
        # Files with fallback date (1900-01-01) will naturally sort to the end
        _results_files.sort(key=lambda x: x.get('last_modified', '1900-01-01T00:00:00'), reverse=True)
        logger.info("Sorted files by newest first")
        
        # Save to cache file
        try:
            logger.info(f"Saving {len(_results_files)} results files to cache: {CACHE_FILE}")
            with open(CACHE_FILE, 'w') as f:
                json.dump(_results_files, f, indent=2)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
        
        return _results_files
        
    except Exception as e:
        logger.error(f"Error walking dataset: {e}")
        return []

def get_results_files():
    """Get all results.jsonl files - cached from startup walk"""
    return walk_dataset_for_results()

def search_results_files(query=""):
    """Search for results.jsonl files in the dataset"""
    logger.info(f"Searching for results.jsonl files with query: '{query}'")
    files = get_results_files()
    results_files = []
    
    logger.info(f"Filtering {len(files)} cached results.jsonl files")
    
    for file_info in files:
        if query.lower() in file_info['path'].lower():
            results_files.append(file_info)
    
    logger.info(f"Found {len(results_files)} results.jsonl files matching query")
    return results_files

def download_results_file(file_path: str) -> List[Dict]:
    """Download and parse a results.jsonl file using HfApi"""
    try:
        # Download the file using hf_hub_download which handles caching
        local_path = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=file_path
        )
        
        results = []
        with open(local_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return results
    except Exception as e:
        print(f"Error downloading file {file_path}: {e}")
        return []

def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze results and return statistics"""
    passed = 0
    has_issues = 0
    failed = 0
    total = len(results)
    total_issues_for_passed = 0  # Sum of issues_count for exit_code=0
    passed_with_exit_0 = 0  # Count of repos with exit_code=0 (passed + has_issues)
    
    logger.info(f"Analyzing {total} results")
    
    for i, result in enumerate(results):
        # Debug first few results to understand structure
        if i < 3:
            logger.info(f"Result {i}: {result}")
        
        # Try different possible structures
        exit_code = None
        issues_count = None
        
        # Structure 1: result.results.exit_code
        if 'results' in result and isinstance(result['results'], dict):
            exit_code = result['results'].get('exit_code')
            issues_count = result['results'].get('issues_count', 0)
        # Structure 2: result.exit_code (direct)
        elif 'exit_code' in result:
            exit_code = result['exit_code']
            issues_count = result.get('issues_count', 0)
        # Structure 3: result.results might be a string or different format
        else:
            logger.warning(f"Unknown result structure: {result}")
            exit_code = 1  # Assume failed if we can't parse
            issues_count = 0
        
        # Default to failed if we can't find exit_code
        if exit_code is None:
            exit_code = 1
            issues_count = 0
        
        logger.debug(f"Repo: {result.get('repo_name', 'unknown')}, exit_code: {exit_code}, issues: {issues_count}")
        
        if exit_code == 0 and issues_count == 0:
            passed += 1
            passed_with_exit_0 += 1
            total_issues_for_passed += issues_count
        elif exit_code == 0:
            has_issues += 1
            passed_with_exit_0 += 1
            total_issues_for_passed += issues_count
        else:
            failed += 1
    
    # Calculate average errors for repos with exit_code=0
    avg_errs = total_issues_for_passed / passed_with_exit_0 if passed_with_exit_0 > 0 else 0
    
    logger.info(f"Analysis complete: {passed} passed, {has_issues} has issues, {failed} failed")
    logger.info(f"AvgErrs: {avg_errs:.2f} (total issues: {total_issues_for_passed}, exit_code=0 count: {passed_with_exit_0})")
    
    return {
        'passed': passed,
        'has_issues': has_issues,
        'failed': failed,
        'total': total,
        'pass_rate': (passed / total * 100) if total > 0 else 0,
        'avg_errs': round(avg_errs, 2)
    }

def calculate_cross_run_stats(selected_files: List[str]) -> Dict[str, Any]:
    """Calculate cross-run statistics"""
    all_repos = set()
    passed_repos = set()
    run_pass_counts = []  # Number of repos that passed in each run
    run_has_issues_counts = []  # Number of repos with issues in each run
    run_failed_counts = []  # Number of repos that failed in each run
    run_avg_errs = []  # AvgErrs for each run
    
    for file_path in selected_files:
        results = download_results_file(file_path)
        run_passed = 0  # Count repos that passed in this run
        
        # Calculate analysis for this run
        run_analysis = analyze_results(results)
        run_avg_errs.append(run_analysis['avg_errs'])
        run_has_issues_counts.append(run_analysis['has_issues'])
        run_failed_counts.append(run_analysis['failed'])
        
        for result in results:
            repo_name = result.get('repo_name', '')
            if repo_name:
                all_repos.add(repo_name)
                
                # Use same parsing logic as analyze_results
                exit_code = None
                issues_count = None
                
                # Structure 1: result.results.exit_code
                if 'results' in result and isinstance(result['results'], dict):
                    exit_code = result['results'].get('exit_code')
                    issues_count = result['results'].get('issues_count', 0)
                # Structure 2: result.exit_code (direct)
                elif 'exit_code' in result:
                    exit_code = result['exit_code']
                    issues_count = result.get('issues_count', 0)
                else:
                    exit_code = 1  # Assume failed if we can't parse
                    issues_count = 0
                
                # Default to failed if we can't find exit_code
                if exit_code is None:
                    exit_code = 1
                    issues_count = 0
                
                if exit_code == 0 and issues_count == 0:
                    passed_repos.add(repo_name)
                    run_passed += 1
        
        run_pass_counts.append(run_passed)
    
    k = len(selected_files)
    pass_at_k = len(passed_repos) / len(all_repos) * 100 if all_repos else 0
    
    # Calculate pass@5 if k >= 5
    pass_at_5 = None
    pass_at_5_repos = None
    if k >= 5:
        # For pass@5, find repos that passed in at least one of the first 5 runs
        pass_5_repos = set()
        for i, file_path in enumerate(selected_files[:5]):
            results = download_results_file(file_path)
            for result in results:
                repo_name = result.get('repo_name', '')
                if repo_name:
                    # Use same parsing logic
                    exit_code = None
                    issues_count = None
                    
                    if 'results' in result and isinstance(result['results'], dict):
                        exit_code = result['results'].get('exit_code')
                        issues_count = result['results'].get('issues_count', 0)
                    elif 'exit_code' in result:
                        exit_code = result['exit_code']
                        issues_count = result.get('issues_count', 0)
                    else:
                        exit_code = 1
                        issues_count = 0
                    
                    if exit_code is None:
                        exit_code = 1
                        issues_count = 0
                    
                    if exit_code == 0 and issues_count == 0:
                        pass_5_repos.add(repo_name)
        
        pass_at_5_repos = len(pass_5_repos)
        pass_at_5 = pass_at_5_repos / len(all_repos) * 100 if all_repos else 0
    
    # Calculate average pass@1 (average number of repos that passed per run)
    avg_passed_repos = sum(run_pass_counts) / len(run_pass_counts) if run_pass_counts else 0
    avg_pass_at_1 = avg_passed_repos / len(all_repos) * 100 if all_repos else 0
    
    # Calculate means for all metrics
    mean_avg_errs = sum(run_avg_errs) / len(run_avg_errs) if run_avg_errs else 0
    mean_has_issues = sum(run_has_issues_counts) / len(run_has_issues_counts) if run_has_issues_counts else 0
    mean_failed = sum(run_failed_counts) / len(run_failed_counts) if run_failed_counts else 0
    
    # Calculate standard deviations
    import math
    
    def calculate_std(values, mean_val):
        if len(values) <= 1:
            return 0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    std_passed = calculate_std(run_pass_counts, avg_passed_repos)
    std_has_issues = calculate_std(run_has_issues_counts, mean_has_issues)
    std_failed = calculate_std(run_failed_counts, mean_failed)
    std_avg_errs = calculate_std(run_avg_errs, mean_avg_errs)
    
    logger.info(f"Cross-run stats: {len(all_repos)} total repos, {len(passed_repos)} passed repos")
    logger.info(f"Pass@k: {len(passed_repos)}/{len(all_repos)} = {pass_at_k:.1f}%")
    if pass_at_5 is not None:
        logger.info(f"Pass@5: {pass_at_5_repos}/{len(all_repos)} = {pass_at_5:.1f}%")
    logger.info(f"Run counts - Passed: {run_pass_counts}, Has Issues: {run_has_issues_counts}, Failed: {run_failed_counts}")
    logger.info(f"Avg passed: {avg_passed_repos:.1f}±{std_passed:.1f}, Has issues: {mean_has_issues:.1f}±{std_has_issues:.1f}, Failed: {mean_failed:.1f}±{std_failed:.1f}")
    logger.info(f"Avg pass@1: {avg_passed_repos:.1f} repos passed per run out of {len(all_repos)} = {avg_pass_at_1:.1f}%")
    logger.info(f"Avg errors: {mean_avg_errs:.2f}±{std_avg_errs:.2f}")
    
    result = {
        'total_repos': len(all_repos),
        'passed_repos': len(passed_repos),
        'pass_at_k': pass_at_k,
        'avg_pass_at_1': avg_pass_at_1,
        'avg_passed_repos': round(avg_passed_repos, 1),
        'avg_errs': round(mean_avg_errs, 2),
        'avg_has_issues': round(mean_has_issues, 1),
        'avg_failed': round(mean_failed, 1),
        'std_passed': round(std_passed, 1),
        'std_has_issues': round(std_has_issues, 1),
        'std_failed': round(std_failed, 1),
        'std_avg_errs': round(std_avg_errs, 2),
        'k': k
    }
    
    if pass_at_5 is not None:
        result['pass_at_5'] = pass_at_5
        result['pass_at_5_repos'] = pass_at_5_repos
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

def get_cache_info():
    """Get cache age and status information"""
    if os.path.exists(CACHE_FILE):
        try:
            stat = os.stat(CACHE_FILE)
            cache_age = time.time() - stat.st_mtime
            cache_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            return {
                'exists': True,
                'age_seconds': cache_age,
                'last_modified': cache_modified
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'exists': False}
    else:
        return {'exists': False}

@app.route('/api/search_files')
def api_search_files():
    query = request.args.get('query', '')
    logger.info(f"API request to search files with query: '{query}'")
    
    try:
        files = search_results_files(query)
        cache_info = get_cache_info()
        logger.info(f"Returning {len(files)} files to frontend")
        return jsonify({
            'files': files,
            'cache_info': cache_info,
            'total_count': len(get_results_files())
        })
    except Exception as e:
        logger.error(f"Error in search_files API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_cache', methods=['POST'])
def api_clear_cache():
    """Clear the file cache and re-walk the dataset"""
    global _results_files
    _results_files = None
    
    # Also remove cache file
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            logger.info(f"Removed cache file: {CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to remove cache file: {e}")
    
    logger.info("File cache cleared, re-walking dataset...")
    
    # Immediately re-walk the dataset to rebuild cache
    try:
        files = walk_dataset_for_results()
        logger.info(f"Cache rebuilt with {len(files)} files")
        cache_info = get_cache_info()
        return jsonify({
            'success': True,
            'message': f'Cache rebuilt successfully with {len(files)} files',
            'file_count': len(files),
            'cache_info': cache_info
        })
    except Exception as e:
        logger.error(f"Error rebuilding cache: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to rebuild cache: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    selected_files = data.get('files', [])
    
    if not selected_files:
        return jsonify({'error': 'No files selected'})
    
    # Analyze individual files
    file_analyses = {}
    for file_path in selected_files:
        results = download_results_file(file_path)
        analysis = analyze_results(results)
        file_analyses[file_path] = analysis
    
    # Calculate cross-run statistics
    cross_run_stats = calculate_cross_run_stats(selected_files)
    
    return jsonify({
        'file_analyses': file_analyses,
        'cross_run_stats': cross_run_stats
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0') 