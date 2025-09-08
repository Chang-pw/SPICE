# SPICE: Submodular Penalized Information–Conflict Selection

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `configs/default.yaml` to specify your model, dataset, and training parameters.

### Training

```bash
# Single GPU
python train.py --config configs/default.yaml

# Multi-GPU
accelerate launch --multi_gpu train.py --config configs/default.yaml
```

## Configuration

Key parameters in `configs/default.yaml`:

- `model_name`: Base model for fine-tuning
- `dataset_path`: Path to training data (JSONL format)
- `pool_size`: Number of samples in selection pool
- `select_k`: Number of samples to select per iteration
- `selection_method`: Selection strategy (original, conflict_penalty, top_k, etc.)
- `alpha_fisher`: Fisher information scaling factor

## Data Format

Training data should be in JSONL format with fields:
```json
{"instruction": "Your prompt", "response": "Expected response"}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.