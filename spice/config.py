import argparse
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _str2bool(v: str) -> bool:
	return str(v).lower() in {"1", "true", "yes", "y"}


@dataclass
class TrainConfig:
	model_name: str = "Qwen/Qwen2-0.5B"
	use_lora: bool = True
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	target_modules: Optional[list] = None

	# Data
	dataset_name: Optional[str] = None
	dataset_path: Optional[str] = None
	split: str = "train"
	text_field_name: str = "text"
	label_field_name: Optional[str] = None
	max_length: int = 1024

	# Selection
	alpha_fisher: float = 1.0
	pool_size: int = 64
	select_k: int = 16
	drop_k: int = 2
	fisher_mode: str = "diag"  # or "scalar"
	
	# Conflict handling
	selection_method: str = "top_k"  # conflict_penalty | top_k
	conflict_penalty: float = 0.1  # Conflict penalty weight
	
	# Halflife early stopping
	halflife_threshold: Optional[float] = None  # Halflife threshold, None means no early stopping

	# Train
	per_device_train_batch_size: int = 4
	gradient_accumulation_steps: int = 1
	update_frequency: int = 1  # Update optimizer every n steps, 1 means every step
	num_epochs: int = 0  # Use epochs to calculate steps; 0 means not used
	num_train_steps: int = 1000
	learning_rate: float = 2.0e-4
	weight_decay: float = 0.01
	warmup_ratio: float = 0.03
	logging_step_freq: int = 10
	
	# Gradient computation optimization
	use_batch_gradient_optimization: bool = True  # Whether to use batch gradient optimization
	batch_gradient_size: int = 8  # Internal batch size for batch gradient computation
	use_gradient_projection: bool = False  # Whether to use gradient projection to low-dimensional space
	gradient_projection_dim: int = 8192  # Gradient projection dimension

	# IO
	output_dir: str = "./outputs/ick"
	logging_dir: str = "./runs/ick"
	seed: int = 42
	device: str = "cuda"
	
	# Checkpoint
	save_checkpoint_freq: int = 100  # Save checkpoint every n steps
	save_total_limit: int = 3  # Maximum number of checkpoints to save
	checkpoint_dir: str = "./checkpoints"  # Checkpoint save directory
	resume_from_checkpoint: Optional[str] = None  # Which checkpoint to resume training from

	# Logging
	logger: str = "tensorboard"  # tensorboard | swanlab
	project: Optional[str] = None
	run_name: Optional[str] = None
	dump_delta_sequences: bool = False

	# Internal/advanced
	fp16: bool = True


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
	if not path:
		return {}
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	res = dict(base)
	for k, v in override.items():
		if isinstance(v, dict) and isinstance(res.get(k), dict):
			res[k] = merge_dict(res[k], v)
		else:
			res[k] = v
	return res


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser()
	p.add_argument("--config", type=str, default=None)
	p.add_argument("--model_name", type=str, default=None)
	p.add_argument("--dataset_name", type=str, default=None)
	p.add_argument("--dataset_path", type=str, default=None)
	p.add_argument("--split", type=str, default=None)
	p.add_argument("--text_field_name", type=str, default=None)
	p.add_argument("--label_field_name", type=str, default=None)
	p.add_argument("--max_length", type=int, default=None)
	p.add_argument("--alpha_fisher", type=float, default=None)
	p.add_argument("--pool_size", type=int, default=None)
	p.add_argument("--select_k", type=int, default=None)
	p.add_argument("--drop_k", type=int, default=None)
	p.add_argument("--fisher_mode", type=str, default=None)
	p.add_argument("--selection_method", type=str, default=None)
	p.add_argument("--conflict_penalty", type=float, default=None)
	p.add_argument("--halflife_threshold", type=float, default=None)
	p.add_argument("--per_device_train_batch_size", type=int, default=None)
	p.add_argument("--gradient_accumulation_steps", type=int, default=None)
	p.add_argument("--update_frequency", type=int, default=None)
	p.add_argument("--num_epochs", type=int, default=None)
	p.add_argument("--num_train_steps", type=int, default=None)
	p.add_argument("--learning_rate", type=float, default=None)
	p.add_argument("--weight_decay", type=float, default=None)
	p.add_argument("--warmup_ratio", type=float, default=None)
	p.add_argument("--logging_step_freq", type=int, default=None)
	p.add_argument("--use_batch_gradient_optimization", type=_str2bool, default=None)
	p.add_argument("--batch_gradient_size", type=int, default=None)
	p.add_argument("--use_gradient_projection", type=_str2bool, default=None)
	p.add_argument("--gradient_projection_dim", type=int, default=None)
	p.add_argument("--output_dir", type=str, default=None)
	p.add_argument("--logging_dir", type=str, default=None)
	p.add_argument("--seed", type=int, default=None)
	p.add_argument("--device", type=str, default=None)
	p.add_argument("--save_checkpoint_freq", type=int, default=None)
	p.add_argument("--save_total_limit", type=int, default=None)
	p.add_argument("--checkpoint_dir", type=str, default=None)
	p.add_argument("--resume_from_checkpoint", type=str, default=None)
	p.add_argument("--use_lora", type=_str2bool, default=None)
	p.add_argument("--lora_r", type=int, default=None)
	p.add_argument("--lora_alpha", type=int, default=None)
	p.add_argument("--lora_dropout", type=float, default=None)
	p.add_argument("--target_modules", type=str, nargs="*", default=None)
	p.add_argument("--fp16", type=_str2bool, default=None)
	p.add_argument("--logger", type=str, default=None)
	p.add_argument("--project", type=str, default=None)
	p.add_argument("--run_name", type=str, default=None)
	p.add_argument("--dump_delta_sequences", type=_str2bool, default=None)
	return p


def load_config() -> TrainConfig:
	# defaults
	cfg_dict: Dict[str, Any] = TrainConfig().__dict__

	# yaml
	args = build_argparser().parse_args()
	yaml_dict = load_yaml(args.config)
	cfg_dict = merge_dict(cfg_dict, yaml_dict)

	# Map nested lora block to flat fields if present
	lora_block = cfg_dict.pop("lora", None)
	if isinstance(lora_block, dict):
		if "r" in lora_block:
			cfg_dict["lora_r"] = lora_block["r"]
		if "alpha" in lora_block:
			cfg_dict["lora_alpha"] = lora_block["alpha"]
		if "dropout" in lora_block:
			cfg_dict["lora_dropout"] = lora_block["dropout"]
		if "target_modules" in lora_block:
			cfg_dict["target_modules"] = lora_block["target_modules"]

	# cli overrides (exclude the parser's own --config key)
	cli = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
	cfg_dict = merge_dict(cfg_dict, cli)

	cfg_dict.pop("config", None)

	# normalize target modules
	if isinstance(cfg_dict.get("target_modules"), str):
		cfg_dict["target_modules"] = [m.strip() for m in cfg_dict["target_modules"].split(",") if m.strip()]

	return TrainConfig(**cfg_dict)