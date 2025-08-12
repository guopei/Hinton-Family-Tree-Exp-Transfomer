# Parameter Sweeping for GPT Model Training

This document explains how to use the parameter sweeping functionality for training the GPT model with different combinations of `n_layer`, `n_head`, `n_embd_factor`, and `dropout` parameters.

## Overview

The parameter sweeping system consists of two main components:

1. **Modified `main.py`**: Now accepts command-line arguments for `n_layer`, `n_head`, `n_embd_factor`, and `dropout`
2. **`sweep_parameters.py`**: A separate script that runs the main training script with different parameter combinations

## Parameter Relationship

The embedding dimension (`n_embd`) is calculated as:
```
n_embd = n_embd_factor × n_head
```

This ensures that the embedding dimension is always divisible by the number of attention heads, which is required for the multi-head attention mechanism.

## Usage

### Running Individual Training Runs

You can now run the main training script with specific parameters:

```bash
# Run with default parameters (n_layer=2, n_head=2, n_embd_factor=32, dropout=0.2)
python main.py

# Run with custom parameters
python main.py --n_layer 3 --n_head 4 --n_embd_factor 16 --dropout 0.1

# Run with different combinations
python main.py --n_layer 1 --n_head 8 --n_embd_factor 64 --dropout 0.3
```

### Running Parameter Sweep

To run a comprehensive parameter sweep across multiple combinations:

```bash
python sweep_parameters.py
```

This will:
- Test all combinations of `n_layer` values: [1, 2]
- Test all combinations of `n_head` values: [1, 2, 4, 8]
- Test all combinations of `n_embd_factor` values: [16, 32, 64]
- Test all combinations of `dropout` values: [0.1, 0.2, 0.3]
- Create a timestamped results directory
- Save individual output files for each combination
- Generate a summary report

## Parameter Ranges

The current sweep configuration tests:

- **n_layer**: [1, 2] (number of transformer layers)
- **n_head**: [1, 2, 4, 8] (number of attention heads)
- **n_embd_factor**: [16, 32, 64] (factor to multiply n_head)
- **dropout**: [0.1, 0.2, 0.3] (dropout rate)

This results in 72 total combinations (2 × 4 × 3 × 3).

## Embedding Dimension Examples

With the current parameter ranges, the resulting embedding dimensions will be:

| n_head | n_embd_factor | n_embd (n_head × n_embd_factor) |
|--------|---------------|-----------------------------------|
| 1      | 16            | 16                               |
| 1      | 32            | 32                               |
| 1      | 64            | 64                               |
| 2      | 16            | 32                               |
| 2      | 32            | 64                               |
| 2      | 64            | 128                              |
| 4      | 16            | 64                               |
| 4      | 32            | 128                              |
| 4      | 64            | 256                              |
| 8      | 16            | 128                              |
| 8      | 32            | 256                              |
| 8      | 64            | 512                              |

## Output Structure

When you run the sweep, it creates a directory structure like:

```
sweep_results_20231201_143022/
├── sweep_summary.txt                                                                    # Overall summary of all runs
├── n_layer_1_n_head_1_n_embd_factor_16_dropout_0.1.txt                               # Output for specific combination
├── n_layer_1_n_head_1_n_embd_factor_16_dropout_0.2.txt                               # Output for specific combination
├── n_layer_1_n_head_1_n_embd_factor_16_dropout_0.3.txt                               # Output for specific combination
└── ... (continues for all combinations)
```

## Customizing the Sweep

To modify the parameter ranges, edit the `sweep_parameters.py` file:

```python
# Change these lines in sweep_parameters.py
n_layer_values = [1, 2]                    # Modify as needed
n_head_values = [1, 2, 4, 8]              # Modify as needed
n_embd_factor_values = [16, 32, 64]       # Modify as needed
dropout_values = [0.1, 0.2, 0.3]          # Modify as needed
```

## Important Notes

1. **Compatibility**: The `n_embd_factor × n_head` relationship ensures that `n_embd` is always divisible by `n_head`, which is required for multi-head attention.

2. **Resource Usage**: Each combination runs 50 training iterations, so the full sweep will take significant time and computational resources. With 72 combinations, this could take several hours.

3. **Memory Considerations**: Larger embedding dimensions (higher `n_embd_factor` or `n_head`) will require more GPU memory.

4. **Error Handling**: The sweep script captures and reports any errors that occur during individual runs.

5. **Output**: All console output is saved to individual files for later analysis.

## Example Output

The sweep script provides real-time feedback:

```
Starting parameter sweep...
n_layer values: [1, 2]
n_head values: [1, 2, 4, 8]
n_embd_factor values: [16, 32, 64]
dropout values: [0.1, 0.2, 0.3]
Total combinations: 72
Results will be saved to: sweep_results_20231201_143022
--------------------------------------------------------------------------------

============================================================
Running with n_layer=1, n_head=1, n_embd_factor=16, dropout=0.1
n_embd will be: 16
============================================================
Training and evaluating with 50 random seeds, n_layer=1, n_head=1, n_embd_factor=16, dropout=0.1
...
✓ Successfully completed n_layer=1, n_head=1, n_embd_factor=16, dropout=0.1
Time taken: 45.23 seconds

============================================================
Running with n_layer=1, n_head=1, n_embd_factor=16, dropout=0.2
============================================================
...
```

## Analyzing Results

After the sweep completes, you can:

1. Check the `sweep_summary.txt` file for an overview
2. Examine individual output files for detailed results
3. Compare performance across different parameter combinations
4. Identify the best performing configurations
5. Analyze the relationship between embedding dimension and performance

## Troubleshooting

- **Memory Issues**: If you encounter memory problems, reduce the `n_embd_factor` or `n_head` values
- **CUDA Errors**: Ensure your GPU has enough memory for larger models
- **Timeouts**: The sweep can take several hours depending on your hardware
- **Compatibility**: All combinations ensure `n_embd % n_head == 0` automatically
