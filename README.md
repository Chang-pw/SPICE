<p align="center">
  <img src="https://img.shields.io/badge/ICLR-2026-blue" alt="ICLR 2026">
  <a href="https://arxiv.org/abs/2601.23155"><img src="https://img.shields.io/badge/arXiv-2601.23155-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

# SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training

English | [中文](README_zh.md)

our work is accepted at **ICLR 2026**.

> **TL;DR:** SPICE selects high-quality training subsets by maximizing Fisher information while penalizing gradient conflicts. Using only 10% of the data, it matches or exceeds full-data fine-tuning and 6 baseline methods across 8 benchmarks.

## Overview

Information-based data selection for instruction tuning is compelling: maximizing the log-determinant of the Fisher information yields a monotone submodular objective with a greedy $(1-1/e)$ approximation guarantee. In practice, however, gradient conflicts between samples cause the marginal information gains to decay rapidly, wasting budget on redundant or conflicting data.

SPICE addresses this by:

- Formalizing an **ε-decomposition** that quantifies deviation from ideal submodularity as a function of conflict statistics
- Proposing a **conflict-aware greedy selector** that maximizes information while penalizing gradient misalignment
- Supporting **early stopping** and **proxy models** for computational efficiency

## Project Structure

```
SPICE/
├── train.py                 # Main training entry point
├── configs/
│   └── default.yaml         # Default configuration
├── spice/
│   ├── adafisher.py         # AdaFisher gradient computation and projection
│   ├── config.py            # Configuration management with validation
│   ├── data.py              # Dataset loading (JSON/JSONL/Parquet)
│   ├── metrics.py           # Logging adapters (TensorBoard / SwanLab)
│   ├── models.py            # Model loading with LoRA support
│   └── select.py            # Selection algorithms (top-k, conflict penalty)
├── train/
│   └── sft_data.sh          # Downstream SFT training script (LLaMA-Factory)
└── run.sh                   # Example multi-GPU launch script
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `configs/default.yaml` to specify your model, dataset, and training parameters. All parameters can also be overridden via command-line arguments.

### Data Format

Training data should be in JSONL format:
```json
{"instruction": "Your prompt", "response": "Expected response"}
```

Parquet and JSON formats are also supported.

### Training

```bash
# Single GPU
python train.py --config configs/default.yaml

# Multi-GPU with Accelerate
accelerate launch --mixed_precision fp16 --multi_gpu --num_processes 4 train.py \
  --config configs/default.yaml
```

See `run.sh` for a full multi-GPU example with all parameters.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `selection_method` | `top_k` | Selection strategy: `top_k` or `conflict_penalty` |
| `alpha_fisher` | `1.0` | Fisher information regularization coefficient |
| `pool_size` | `128` | Number of candidate samples per selection step |
| `select_k` | `64` | Number of samples to select (must be ≤ pool_size) |
| `conflict_penalty` | `0.1` | Penalty weight for gradient conflicts |
| `update_frequency` | `1` | Accumulate N selection steps before one optimizer update |

## Downstream SFT

After data selection, use `train/sft_data.sh` to fine-tune with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The script runs SFT for all baseline methods and SPICE-selected data.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chang2026spicesubmodularpenalizedinformationconflict,
      title={SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training}, 
      author={Powei Chang and Jinpeng Zhang and Bowen Chen and Chenyu Wang and Chenlu Guo and Yixing Zhang and Yukang Gao and JianXiang Xiang and Yue Gao and Chaoqun Sun and Yiyi Chen and Dongying Kong},
      year={2026},
      eprint={2601.23155},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.23155}, 
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
