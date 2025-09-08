from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch


def load_model_and_tokenizer(model_name: str, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float,
							target_modules: Optional[List[str]] = None) -> Tuple[torch.nn.Module, AutoTokenizer]:
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	
	# Set data type
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	
	# Check if it is a distributed environment
	import os
	is_distributed = (
		os.environ.get("WORLD_SIZE") is not None and 
		os.environ.get("LOCAL_RANK") is not None and
		os.environ.get("RANK") is not None
	)
	
	if is_distributed:
		# Distributed environment: do not specify device_map, let eval.py manually control device allocation
		model = AutoModelForCausalLM.from_pretrained(
			model_name, 
			torch_dtype=torch_dtype,
			trust_remote_code=True
		)
	else:
		# Single GPU training: use device_map="auto"
		model = AutoModelForCausalLM.from_pretrained(
			model_name, 
			torch_dtype=torch_dtype,
			device_map="auto",
			trust_remote_code=True
		)
	
	if use_lora:
		if target_modules is None:
			# Common for Qwen/LLaMA style blocks
			target_modules = [
				"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
			]
		lora_cfg = LoraConfig(
			r=lora_r,
			lora_alpha=lora_alpha,
			lora_dropout=lora_dropout,
			target_modules=target_modules,
			task_type="CAUSAL_LM",
		)
		model = get_peft_model(model, lora_cfg)
	
	return model, tokenizer