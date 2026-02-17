from typing import Any, Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
import torch
import pandas as pd
import os


def format_example(ex: Dict[str, Any], text_field: str, label_field: Optional[str]) -> str:
	"""Format a single example into instruction-response text for SFT."""
	if label_field and label_field in ex and ex[label_field] is not None:
		return f"Instruction:\n{ex[text_field]}\n\nResponse:\n{ex[label_field]}"
	return str(ex[text_field])


def load_sft_dataset(dataset_name: Optional[str], dataset_path: Optional[str], split: str,
					text_field_name: str, label_field_name: Optional[str]) -> Dataset:
	"""Load an SFT dataset from HuggingFace Hub or a local file (JSON/JSONL/Parquet)."""
	if dataset_name:
		ds = load_dataset(dataset_name, split=split)
	elif dataset_path:
		# Check file extension
		file_ext = os.path.splitext(dataset_path)[1].lower()
		
		if file_ext == '.parquet':
			# Read parquet file
			df = pd.read_parquet(dataset_path)
			ds = Dataset.from_pandas(df)
		elif file_ext == '.json' or file_ext == '.jsonl':
			# Read JSON file
			ds = load_dataset("json", data_files=dataset_path, split=split)
		else:
			# Try to auto-detect format
			try:
				# Try parquet first
				df = pd.read_parquet(dataset_path)
				ds = Dataset.from_pandas(df)
			except Exception:
				try:
					# Try JSON next
					ds = load_dataset("json", data_files=dataset_path, split=split)
				except Exception as e:
					raise ValueError(f"Unsupported file format: {dataset_path}. Supported formats: .parquet, .json, .jsonl. Error: {e}")
	else:
		raise ValueError("Either dataset_name or dataset_path must be provided")
	
	# ensure text field exists
	assert text_field_name in ds.column_names, f"Missing text_field_name={text_field_name}"
	return ds


def collate_texts(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int,
					text_field_name: str, label_field_name: Optional[str]) -> Dict[str, torch.Tensor]:
	"""Tokenize and collate a list of examples into a model-ready batch."""
	texts = [format_example(ex, text_field_name, label_field_name) for ex in batch]
	tok = tokenizer(
		texts,
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)
	labels = tok["input_ids"].clone()
	labels[labels == tokenizer.pad_token_id] = -100
	return {**tok, "labels": labels}