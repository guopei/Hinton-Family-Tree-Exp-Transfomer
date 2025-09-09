# Hinton Family Tree Experiment Transformer

This is a sister repo of my [Hinton Family Tree Repro](https://github.com/guopei/Hinton-Family-Tree-Exp-Repro) experiment. This code explores using small transformers to learn to predict family relations proposed by Hinton in his 1986 paper.

## Quick Start
```bash
uv run main.py
```

Due to the discrepency of the low-level math accelation library on different hardware, the accuracy number ranges from 0.715 (A Windows VM) to 0.7 (Macbook air M4 2025) to 0.665 (Ubuntu 24.04.03 LTS on Intel i7-12700K CPU). Details on reproducibility can be found [here](https://peiguo.me/posts/hinton-family-tree-experiment/#reproducibility-the-ghost-that-haunts-my-mind) in my blogpost.

More details on this experiment can be found [here](https://peiguo.me/posts/hinton-family-tree-experiment/#transformer-architecture).