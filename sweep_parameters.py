#!/usr/bin/env python3
"""
Parameter sweeping script for GPT model training.
This script runs the main.py training script with different combinations of n_layer and n_head parameters.
"""

import subprocess
import sys
import os
import time
import re
from datetime import datetime

def extract_average_test_accuracy(output_file):
    """Extract the average test accuracy from an output file."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # First try to find the specific sweep result line
        match = re.search(r'SWEEP_RESULT_AVG_TEST_ACCURACY: (\d+\.\d+)', content)
        if match:
            return float(match.group(1))
        
        # Fallback to the regular "Average test accuracy:" line
        match = re.search(r'Average test accuracy: (\d+\.\d+)', content)
        if match:
            return float(match.group(1))
        else:
            return None
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return None

def extract_average_train_accuracy(output_file):
    """Extract the average train accuracy from an output file."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # First try to find the specific sweep result line
        match = re.search(r'SWEEP_RESULT_AVG_TRAIN_ACCURACY: (\d+\.\d+)', content)
        if match:
            return float(match.group(1))
        
        # Fallback to the regular "Average train accuracy:" line
        match = re.search(r'Average train accuracy: (\d+\.\d+)', content)
        if match:
            return float(match.group(1))
        else:
            return None
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return None

def run_sweep():
    # Define the parameter ranges to sweep
    n_layer_values = [2, 3, 4, 5]
    n_head_values = [2, 4, 6, 8, 10, 12, 14, 16]
    n_embd_factor_values = [4, 8, 12, 16]  # Factors to multiply n_head
    dropout_values = [0.1, 0.2, 0.3]    # Dropout rates
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"sweep_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a summary file
    summary_file = os.path.join(results_dir, "sweep_summary.txt")
    
    print(f"Starting parameter sweep...")
    print(f"n_layer values: {n_layer_values}")
    print(f"n_head values: {n_head_values}")
    print(f"n_embd_factor values: {n_embd_factor_values}")
    print(f"dropout values: {dropout_values}")
    total_combinations = len(n_layer_values) * len(n_head_values) * len(n_embd_factor_values) * len(dropout_values)
    print(f"Total combinations: {total_combinations}")
    print(f"Results will be saved to: {results_dir}")
    print("-" * 80)
    
    # Store results for summary
    results = []
    
    # Run each combination
    for n_layer in n_layer_values:
        for n_head in n_head_values:
            for n_embd_factor in n_embd_factor_values:
                for dropout in dropout_values:
                    print(f"\n{'='*60}")
                    print(f"Running with n_layer={n_layer}, n_head={n_head}, n_embd_factor={n_embd_factor}, dropout={dropout}")
                    print(f"n_embd will be: {n_embd_factor * n_head}")
                    print(f"{'='*60}")
                    
                    # Create output file for this combination
                    output_file = os.path.join(results_dir, f"n_layer_{n_layer}_n_head_{n_head}_n_embd_factor_{n_embd_factor}_dropout_{dropout}.txt")
                    
                    # Build the command
                    cmd = [
                        sys.executable, "main.py",
                        "--n_layer", str(n_layer),
                        "--n_head", str(n_head),
                        "--n_embd_factor", str(n_embd_factor),
                        "--dropout", str(dropout)
                    ]
                    
                    start_time = time.time()
                    
                    try:
                        # Run the command and capture output
                        with open(output_file, 'w') as f:
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True,
                                bufsize=1
                            )
                            
                            # Stream output to both console and file
                            for line in process.stdout:
                                print(line.rstrip())
                                f.write(line)
                                f.flush()
                            
                            process.wait()
                            
                            if process.returncode == 0:
                                print(f"✓ Successfully completed n_layer={n_layer}, n_head={n_head}, n_embd_factor={n_embd_factor}, dropout={dropout}")
                                
                                # Extract average test accuracy from output file
                                avg_test_acc = extract_average_test_accuracy(output_file)
                                avg_train_acc = extract_average_train_accuracy(output_file)
                                
                                results.append({
                                    'n_layer': n_layer,
                                    'n_head': n_head,
                                    'n_embd_factor': n_embd_factor,
                                    'dropout': dropout,
                                    'n_embd': n_embd_factor * n_head,
                                    'status': 'success',
                                    'output_file': output_file,
                                    'avg_test_accuracy': avg_test_acc,
                                    'avg_train_accuracy': avg_train_acc
                                })
                            else:
                                print(f"✗ Failed with n_layer={n_layer}, n_head={n_head}, n_embd_factor={n_embd_factor}, dropout={dropout}")
                                results.append({
                                    'n_layer': n_layer,
                                    'n_head': n_head,
                                    'n_embd_factor': n_embd_factor,
                                    'dropout': dropout,
                                    'n_embd': n_embd_factor * n_head,
                                    'status': 'failed',
                                    'output_file': output_file,
                                    'avg_test_accuracy': None,
                                    'avg_train_accuracy': None
                                })
                                
                    except Exception as e:
                        print(f"✗ Error running n_layer={n_layer}, n_head={n_head}, n_embd_factor={n_embd_factor}, dropout={dropout}: {e}")
                        results.append({
                            'n_layer': n_layer,
                            'n_head': n_head,
                            'n_embd_factor': n_embd_factor,
                            'dropout': dropout,
                            'n_embd': n_embd_factor * n_head,
                            'status': 'error',
                            'error': str(e),
                            'avg_test_accuracy': None,
                            'avg_train_accuracy': None
                        })
                    
                    end_time = time.time()
                    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Write summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    
    with open(summary_file, 'w') as f:
        f.write("Parameter Sweep Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations: {total_combinations}\n\n")
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        errors = [r for r in results if r['status'] == 'error']
        
        f.write(f"Successful runs: {len(successful)}\n")
        f.write(f"Failed runs: {len(failed)}\n")
        f.write(f"Error runs: {len(errors)}\n\n")
        
        # Sort successful results by average test accuracy (descending)
        successful_sorted = sorted(successful, key=lambda x: x['avg_test_accuracy'] if x['avg_test_accuracy'] is not None else -1, reverse=True)
        
        f.write("Successful combinations (sorted by average test accuracy):\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(successful_sorted, 1):
            test_acc_str = f"{result['avg_test_accuracy']:.4f}" if result['avg_test_accuracy'] is not None else "N/A"
            train_acc_str = f"{result['avg_train_accuracy']:.4f}" if result['avg_train_accuracy'] is not None else "N/A"
            f.write(f"{i:2d}. n_layer={result['n_layer']}, n_head={result['n_head']}, n_embd_factor={result['n_embd_factor']}, dropout={result['dropout']}, n_embd={result['n_embd']}, test_acc={test_acc_str}, train_acc={train_acc_str}\n")
        
        if failed:
            f.write("\nFailed combinations:\n")
            for result in failed:
                f.write(f"  n_layer={result['n_layer']}, n_head={result['n_head']}, n_embd_factor={result['n_embd_factor']}, dropout={result['dropout']}, n_embd={result['n_embd']}\n")
        
        if errors:
            f.write("\nError combinations:\n")
            for result in errors:
                f.write(f"  n_layer={result['n_layer']}, n_head={result['n_head']}, n_embd_factor={result['n_embd_factor']}, dropout={result['dropout']}, n_embd={result['n_embd']}: {result['error']}\n")
        
        # Summary statistics
        if successful_sorted:
            f.write("\n" + "="*50 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("="*50 + "\n")
            
            test_accuracies = [r['avg_test_accuracy'] for r in successful_sorted if r['avg_test_accuracy'] is not None]
            
            if test_accuracies:
                f.write(f"Best average test accuracy: {max(test_accuracies):.4f}\n")
                f.write(f"Worst average test accuracy: {min(test_accuracies):.4f}\n")
                f.write(f"Mean average test accuracy: {sum(test_accuracies)/len(test_accuracies):.4f}\n")
                
                # Find best configuration
                best_result = successful_sorted[0]
                f.write(f"\nBest configuration:\n")
                f.write(f"  n_layer={best_result['n_layer']}, n_head={best_result['n_head']}, n_embd_factor={best_result['n_embd_factor']}, dropout={best_result['dropout']}, n_embd={best_result['n_embd']}\n")
                f.write(f"  Average test accuracy: {best_result['avg_test_accuracy']:.4f}\n")
                f.write(f"  Average train accuracy: {best_result['avg_train_accuracy']:.4f}\n")
    
    # Print summary to console
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs: {len(failed)}")
    print(f"Error runs: {len(errors)}")
    
    if successful_sorted:
        best_result = successful_sorted[0]
        best_test_acc = best_result['avg_test_accuracy']
        if best_test_acc is not None:
            print(f"Best average test accuracy: {best_test_acc:.4f}")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"All output files saved to: {results_dir}")

if __name__ == "__main__":
    run_sweep()
