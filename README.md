# Hinton Family Tree Experiment Transformer

A research project exploring whether small transformer models can learn family relationships. This is a sister project of my [Hinton Family Tree Repro](https://github.com/guopei/Hinton-Family-Tree-Exp-Repro) project.

## Overview
Minimal GPT-style transformer trained on synthetic family tree data (20 people, 12 relationship types). Investigates minimum viable architecture for relational reasoning.

## Quick Start
```bash
uv run main.py
```

I was able to get 0.715 average accuracy using transformers. 

## Features
- Configurable layers, heads, embedding dimensions
- Multi-label classification for relationships
- Reproducible training (50 random seeds)
- Hyperparameter sweep results included

## Files
- `main.py` - Training script
- `model.py` - Transformer implementation  
- `data.py` - Family tree dataset
- `pyproject.toml` - Dependencies

Thing I found help with validation accuracy:

Loss on both outputs
Weight decay
Hyperparameter searching (num of layer, num of head, embedding size, dropout rate)
learning rate warmup
gradient clipping
Do not touch anything in the model
