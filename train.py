import os
import glob
import json
import logging
import shutil
import time
from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

from spice.config import load_config
from spice.models import load_model_and_tokenizer
from spice.data import load_sft_dataset, collate_texts
from spice.adafisher import (
	compute_per_sample_grads,
	compute_per_sample_grads_batch_optimized,
	compute_per_sample_grads_with_projection,
	compute_per_sample_grads_batch_optimized_with_projection,
	cosine_similarity,
)
from spice.select import (
	greedy_select_with_metrics,
	greedy_select_with_conflict_penalty,
	compute_conflict_metrics,
	top_k_select,
)
from spice.metrics import log_step, LoggerAdapter

logger = logging.getLogger(__name__)


def _ddp_barrier():
	if torch.distributed.is_available() and torch.distributed.is_initialized():
		try:
			torch.distributed.barrier()
		except Exception:
			pass


def _bcast_start_end(device: torch.device, start: int, end: int) -> (int, int):
	if not torch.distributed.is_available() or not torch.distributed.is_initialized():
		return start, end
	src = 0
	buf = torch.tensor([start, end], dtype=torch.long, device=device)
	torch.distributed.broadcast(buf, src=src)
	return int(buf[0].item()), int(buf[1].item())


def save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch_idx, 
                   checkpoint_dir: str, save_total_limit: int = 3):
	"""
	Save checkpoint
	"""
	# Ensure checkpoint directory exists
	os.makedirs(checkpoint_dir, exist_ok=True)
	
	checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
	
	# Save model and optimizer state
	accelerator.save_state(checkpoint_path)
	
	# Save additional training state
	training_state = {
		"global_step": global_step,
		"epoch_idx": epoch_idx,
		"config": {}
	}
	
	state_path = os.path.join(checkpoint_path, "training_state.json")
	with open(state_path, "w", encoding="utf-8") as f:
		json.dump(training_state, f, indent=2, ensure_ascii=False)
	
	logger.info("Checkpoint saved: %s", checkpoint_path)
	
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	
	if save_total_limit > 0:
		checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
		checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
		
		for checkpoint in checkpoints[save_total_limit:]:
			shutil.rmtree(checkpoint)
			logger.info("Removed old checkpoint: %s", checkpoint)


def load_checkpoint(accelerator, checkpoint_path: str):
	"""
	Load training state from checkpoint
	"""
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
	
	# Load model and optimizer state
	accelerator.load_state(checkpoint_path)
	
	# Load additional training state
	state_path = os.path.join(checkpoint_path, "training_state.json")
	if os.path.exists(state_path):
		with open(state_path, "r", encoding="utf-8") as f:
			training_state = json.load(f)
		return training_state.get("global_step", 0), training_state.get("epoch_idx", 0)
	
	return 0, 0


def main():
	logging.basicConfig(
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
		level=logging.INFO,
	)
	cfg = load_config()
	accelerator = Accelerator()
	torch.manual_seed(cfg.seed)
	device = accelerator.device

	# Load model
	model, tokenizer = load_model_and_tokenizer(
		cfg.model_name, cfg.use_lora, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules
	)
	model.to(device)
	model.train()
	if accelerator.is_main_process:
		logger.info("Model loaded successfully")

	# Load dataset
	dataset = load_sft_dataset(cfg.dataset_name, cfg.dataset_path, cfg.split, cfg.text_field_name, cfg.label_field_name)
	num_samples = len(dataset)
	steps_per_epoch = max(1, (num_samples + cfg.pool_size - 1) // cfg.pool_size)
	
	if getattr(cfg, "num_epochs", 0) and cfg.num_epochs > 0:
		cfg.num_train_steps = int(steps_per_epoch * cfg.num_epochs)
	
	if accelerator.is_main_process:
		logger.info("Dataset loaded successfully")

	# Initialize training components
	logger = LoggerAdapter(cfg.logger, cfg.logging_dir, cfg.project, cfg.run_name, accelerator.is_main_process)
	optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	warmup_steps = max(1, int(cfg.warmup_ratio * cfg.num_train_steps))
	scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, cfg.num_train_steps)

	# Prepare distributed training
	model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

	if cfg.resume_from_checkpoint:
		try:
			global_step, epoch_idx = load_checkpoint(accelerator, cfg.resume_from_checkpoint)
		except Exception:
			global_step = 0
			epoch_idx = 0
	else:
		global_step = 0
		epoch_idx = 0

	def zero_grad_fn():
		optimizer.zero_grad(set_to_none=True)

	def forward_loss_from_examples(examples: List[dict]) -> torch.Tensor:
		batch = collate_texts(examples, tokenizer, cfg.max_length, cfg.text_field_name, cfg.label_field_name)
		batch = {k: v.to(device) for k, v in batch.items()}
		out = model(**batch)
		return out.loss

	# Initialize training loop
	base_seed = int(cfg.seed)
	epoch_step = 0
	train_step = 0  # Number of steps that actually perform optimizer updates
	order: List[int] = []
	
	# Output path setup
	out_path = os.path.join(cfg.output_dir, "delta_sequences.jsonl")
	selected_data_path = os.path.join(cfg.output_dir, "selected_data.jsonl")
	selected_data_summary_path = os.path.join(cfg.output_dir, "selected_data_summary.json")
	
	if accelerator.is_main_process:
		os.makedirs(cfg.output_dir, exist_ok=True)
		# Clean up old files
		for file_path in [out_path, selected_data_path, selected_data_summary_path]:
			if os.path.exists(file_path):
				os.remove(file_path)

	def reseed_epoch():
		nonlocal order, epoch_step, epoch_idx
		# deterministically generate the same permutation on all ranks
		epoch_seed = base_seed + epoch_idx
		gen = torch.Generator(device="cpu").manual_seed(epoch_seed)
		order = torch.randperm(num_samples, generator=gen).tolist()
		_ddp_barrier()
		epoch_step = 0
		epoch_idx += 1

	# initialize first epoch
	reseed_epoch()

	# helper for no_sync via accelerator
	def no_sync_ctx(mdl):
		try:
			return accelerator.no_sync(mdl)
		except Exception:
			from contextlib import nullcontext
			return nullcontext()

	cursor = 0
	pbar = tqdm(total=cfg.num_train_steps, disable=not accelerator.is_main_process, dynamic_ncols=True)
	start_time = time.time()
	last_delta_mean = None
	last_eps_mean = None
	last_kept = 0
	last_epoch = 0
	last_loss_mean = None
	
	# Data for tracking all selections
	all_selected_data = []
	selection_stats = {
		"total_steps": 0,
		"total_selected": 0,
		"selection_method": cfg.selection_method,
		"config": {
			"alpha_fisher": cfg.alpha_fisher,
			"select_k": cfg.select_k,
			"pool_size": cfg.pool_size,
			"dataset_name": cfg.dataset_name,
		}
	}
	
	# Batch update related variables
	update_frequency = getattr(cfg, 'update_frequency', 1)  # Update every n steps, default 1
	accumulated_examples = []  # Accumulated training examples
	selection_step = 0  # Selection step counter
	

	
	while global_step < cfg.num_train_steps:
		if accelerator.is_main_process:
			logger.info("Step %d", global_step)
		
		# 1) Build candidate pool from sequential window
		if accelerator.is_main_process:
			if cursor >= num_samples:
				cursor = 0
				reseed_epoch()
			start = cursor
			end = min(cursor + cfg.pool_size, num_samples)
			cursor = end
			epoch_step += 1
			last_epoch = epoch_idx - 1
		else:
			start = 0
			end = 0
			
		# broadcast start/end, then slice locally
		start, end = _bcast_start_end(device, start, end)
		pool_idx = order[start:end]
		pool_examples = [dataset[i] for i in pool_idx]

		# 2) Backward per-sample to get gradients (no step). Avoid DDP allreduce via no_sync
		if accelerator.is_main_process:
			logger.info("Computing per-sample gradients")
		
		# All processes need to compute gradients, but only main process does data selection
		if cfg.use_gradient_projection:
			# Use gradient projection
			if cfg.use_batch_gradient_optimization:
				g_list = compute_per_sample_grads_batch_optimized_with_projection(
					model, pool_examples, forward_loss_from_examples, zero_grad_fn, device,
					no_sync_ctx=no_sync_ctx, batch_size=cfg.batch_gradient_size, 
					projection_dim=cfg.gradient_projection_dim,
				)
			else:
				g_list = compute_per_sample_grads_with_projection(
					model, pool_examples, forward_loss_from_examples, zero_grad_fn, device,
					no_sync_ctx=no_sync_ctx, projection_dim=cfg.gradient_projection_dim,
				)
		else:
			# Do not use gradient projection
			if cfg.use_batch_gradient_optimization:
				g_list = compute_per_sample_grads_batch_optimized(
					model, pool_examples, forward_loss_from_examples, zero_grad_fn, device,
					no_sync_ctx=no_sync_ctx, batch_size=cfg.batch_gradient_size,
				)
			else:
				g_list = compute_per_sample_grads(
					model, pool_examples, forward_loss_from_examples, zero_grad_fn, device,
					no_sync_ctx=no_sync_ctx,
				)
		
		if accelerator.is_main_process:
			logger.debug("Per-sample gradients computed")

		# 3) Data selection based on configuration
		marginals_raw = None
		selected_rel_raw = None
		
		# Main selection based on method - only on main process
		if accelerator.is_main_process:
			if cfg.selection_method == "conflict_penalty":
				# Conflict penalty method: considers gradient conflicts
				sel_f = greedy_select_with_conflict_penalty(
					g_list, cfg.alpha_fisher, cfg.select_k, cfg.conflict_penalty, cfg.fisher_mode
				)
				marginals_spice = sel_f["marginals"]
				selected_rel_spice = sel_f["selected"]
				kept_rel = selected_rel_spice
				
				# For delta sequences comparison, need separate raw selection
				if cfg.dump_delta_sequences:
					sel_raw = greedy_select_with_metrics(g_list, cfg.alpha_fisher, cfg.select_k, cfg.fisher_mode, cfg.halflife_threshold)
					marginals_raw = sel_raw["marginals"]
					selected_rel_raw = sel_raw["selected"]
			
			elif cfg.selection_method == "top_k":
				# Fast top-k selection (no iteration)
				sel_f = top_k_select(g_list, cfg.alpha_fisher, cfg.select_k, cfg.fisher_mode, cfg.halflife_threshold)
				marginals_spice = sel_f["marginals"]
				selected_rel_spice = sel_f["selected"]
				kept_rel = selected_rel_spice
				
				# For delta sequences comparison, need separate raw selection
				if cfg.dump_delta_sequences:
					sel_raw = greedy_select_with_metrics(g_list, cfg.alpha_fisher, cfg.select_k, cfg.fisher_mode, cfg.halflife_threshold)
					marginals_raw = sel_raw["marginals"]
					selected_rel_raw = sel_raw["selected"]
			else:
				raise ValueError(f"Unknown selection_method: {cfg.selection_method}. Supported methods: conflict_penalty, top_k")
		else:
			# Non-main process: initialize empty variables, wait for main process broadcast
			marginals_spice = []
			kept_rel = []
			marginals_raw = None
			selected_rel_raw = None

		if accelerator.is_main_process:
			logger.debug("Data selection completed")
		
		# Calculate required metrics before deleting g_list
		conf_kept = {"conflict_mean": 0.0, "conflict_ratio": 0.0, "avg_cosine": 0.0}
		cos_mean_k = 0.0
		cos_min_k = 0.0
		frac_below = 0.0
		
		# Only calculate metrics on main process with valid data
		if accelerator.is_main_process and kept_rel and len(g_list) > 0:
			# Calculate conflict metrics
			conf_kept = compute_conflict_metrics(kept_rel, g_list)
			
			# Calculate conflict-to-mean metrics
			g_kept = [g_list[i] for i in kept_rel]
			g_mean_k = torch.stack(g_kept, dim=0).mean(0)
			cos_k = [cosine_similarity(g, g_mean_k) for g in g_kept]
			cos_mean_k = float(sum(cos_k) / len(cos_k))
			cos_min_k = float(min(cos_k))
			# frac_below calculation removed (no longer needed without pruning method)
			
			# Clean up intermediate tensors
			del g_kept, g_mean_k, cos_k
		
		# Free gradient memory before training step
		del g_list
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		def _bcast_kept(kept: List[int]) -> List[int]:
			if not torch.distributed.is_available() or not torch.distributed.is_initialized():
				return kept
			
			# Ensure all processes participate in communication
			world_size = accelerator.num_processes
			rank = accelerator.process_index
			
			# Main process broadcasts length
			len_t = torch.tensor([len(kept) if rank == 0 else 0], dtype=torch.long, device=device)
			torch.distributed.broadcast(len_t, src=0)
			L = int(len_t.item())
			
			# Create fixed-size buffer
			buf = torch.full((cfg.select_k,), fill_value=-1, dtype=torch.long, device=device)
			
			# Main process fills data
			if rank == 0 and L > 0:
				buf[:L] = torch.tensor(kept, dtype=torch.long, device=device)
			
			# Broadcast data
			torch.distributed.broadcast(buf, src=0)
			
			# Extract valid data
			result = buf[:L].cpu().tolist()
			
			# Clean up temporary tensors
			del len_t, buf
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			return result

		# Broadcast selection results to all processes
		kept_rel = _bcast_kept(kept_rel)
		last_kept = len(kept_rel)
		
		if accelerator.is_main_process:
			logger.debug("Computing information metrics")
		
		# Accumulate selected samples - all processes execute
		current_batch_examples = []
		for ridx in kept_rel:
			example = dataset[pool_idx[ridx]]
			current_batch_examples.append(example)
		
		# Add current batch samples to accumulated list
		accumulated_examples.extend(current_batch_examples)
		
		del current_batch_examples
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		selection_step += 1
		
		if accelerator.is_main_process:
			logger.debug("Accumulated samples: %d (target: %d)", len(accumulated_examples), update_frequency * cfg.select_k)
		
		# Save selected data records
		if accelerator.is_main_process and kept_rel:
			step_selected_data = []
			for ridx in kept_rel:
				selected_example = dataset[pool_idx[ridx]]
				data_record = {
					"step": global_step,
					"epoch": last_epoch,
					"pool_index": ridx,
					"global_index": pool_idx[ridx],
					"data": selected_example
				}
				step_selected_data.append(data_record)
				all_selected_data.append(data_record)
			
			# Save to JSONL file in real-time
			with open(selected_data_path, "a", encoding="utf-8") as f:
				for record in step_selected_data:
					json.dump(record, f, ensure_ascii=False)
					f.write("\n")
			
			# Update statistics
			selection_stats["total_steps"] = global_step + 1
			selection_stats["total_selected"] += len(kept_rel)

		# 5) Check if optimizer update is needed
		should_update = (selection_step % update_frequency == 0) or (global_step == cfg.num_train_steps - 1)
		
		# Distributed synchronization: ensure all ranks reach the same update decision point
		if torch.distributed.is_available() and torch.distributed.is_initialized():
			torch.distributed.barrier()
		
		if should_update and len(accumulated_examples) > 0:
			if accelerator.is_main_process:
				logger.info("Optimizer update with %d accumulated samples", len(accumulated_examples))
			
			# Use accumulated samples for optimizer update - efficient batch processing
			zero_grad_fn()
			acc = max(1, cfg.gradient_accumulation_steps)
			world = accelerator.num_processes
			rank = accelerator.process_index
			loss_sum = 0.0
			loss_cnt = 0
			response_length_sum = 0.0
			response_length_cnt = 0
			
			# Distribute samples by rank to avoid duplicate computation
			my_examples = [example for i, example in enumerate(accumulated_examples) if (i % world) == rank]
			
			if len(my_examples) > 0:
				# Batch process samples for current rank
				batch_size = min(len(my_examples), cfg.per_device_train_batch_size)
				num_batches = (len(my_examples) + batch_size - 1) // batch_size
				
				for batch_idx in range(num_batches):
					start_idx = batch_idx * batch_size
					end_idx = min(start_idx + batch_size, len(my_examples))
					batch_examples = my_examples[start_idx:end_idx]
					
					# Batch forward pass
					loss_value = forward_loss_from_examples(batch_examples)
					loss_sum += float(loss_value.item()) * len(batch_examples)
					loss_cnt += len(batch_examples)
					
					# Calculate response length
					for example in batch_examples:
						if cfg.label_field_name in example:
							response_text = example[cfg.label_field_name]
							if isinstance(response_text, str):
								response_length_sum += len(response_text)
								response_length_cnt += 1
					
					# Gradient accumulation
					loss = loss_value / acc
					accelerator.backward(loss)
			
			# Only execute optimizer update once at the end
			optimizer.step()
			scheduler.step()
			zero_grad_fn()
			
			# Increment actual training steps
			train_step += 1
			
			# gather loss mean and response length to main
			if torch.distributed.is_available() and torch.distributed.is_initialized():
				loss_tensor = torch.tensor([loss_sum, loss_cnt, response_length_sum, response_length_cnt], dtype=torch.float64, device=device)
				torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
				if accelerator.is_main_process:
					loss_sum = float(loss_tensor[0].item())
					loss_cnt = int(loss_tensor[1].item())
					response_length_sum = float(loss_tensor[2].item())
					response_length_cnt = int(loss_tensor[3].item())
			
			# Clear accumulated samples
			accumulated_examples = []
			
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			# Distributed synchronization: ensure all ranks have completed optimizer update
			if torch.distributed.is_available() and torch.distributed.is_initialized():
				torch.distributed.barrier()
			
			if accelerator.is_main_process:
				logger.debug("Optimizer update completed")
		else:
			# Default values when not updating
			loss_sum = 0.0
			loss_cnt = 0
			response_length_sum = 0.0
			response_length_cnt = 0
		


		# 6) Metrics and logging (main only)
		if accelerator.is_main_process:
			# derive deltas/eps from spice selection
			if marginals_spice:
				# Handle different marginal formats based on selection method
				if cfg.selection_method == "conflict_penalty":
					# New format: (idx, delta, eps, conflict_penalty_val)
					deltas_q = [d for _, d, _, _ in marginals_spice]
					eps_q = [e for _, _, e, _ in marginals_spice]
					conflict_penalties = [cp for _, _, _, cp in marginals_spice]
				else:
					# Original format: (idx, delta, eps)
					deltas_q = [d for _, d, _ in marginals_spice]
					eps_q = [e for _, _, e in marginals_spice]
					conflict_penalties = [0.0] * len(marginals_spice)
				
				# within-step decay metrics for spice
				delta_first = float(deltas_q[0])
				delta_last = float(deltas_q[-1])
				slope = float((delta_last - delta_first) / max(1, (len(deltas_q) - 1)))
				decay_abs = float(delta_last - delta_first)
				decay_rel = float(delta_last / (delta_first + 1e-8) - 1.0) if delta_first != 0 else 0.0
				# loss mean
				loss_mean = (loss_sum / max(1, loss_cnt)) if loss_cnt > 0 else 0.0
				# response length mean
				response_length_mean = (response_length_sum / max(1, response_length_cnt)) if response_length_cnt > 0 else 0.0
				# conflicts on kept - already calculated above
				# conflict-to-mean (angle-based) on kept - already calculated above
			
				log_info = {
					"train/step": float(train_step),  # Use actual training steps
					"train/selection_step": float(global_step),  # Add selection steps for comparison
					"train/epoch": float(last_epoch),
					"train/epoch_step": float(epoch_step),
					"train/loss_mean": float(loss_mean),
					"train/response_length_mean": float(response_length_mean),
					"train/elapsed_sec": float(time.time() - start_time),
					"select/num_selected": float(len(kept_rel)),
					"select/num_kept": float(len(kept_rel)),
					"theory/delta_first": delta_first,
					"theory/delta_last": delta_last,
					"theory/decay_abs": decay_abs,
					"theory/decay_rel": decay_rel,
					"theory/slope": slope,
					"theory/delta_mean": float(sum(deltas_q) / len(deltas_q)),
					"theory/delta_min": float(min(deltas_q)),
					"theory/epsilon_mean": float(sum(eps_q) / len(eps_q)),
					"theory/epsilon_min": float(min(eps_q)),
					"conflict/mean": conf_kept["conflict_mean"],
					"conflict/ratio": conf_kept["conflict_ratio"],
					"conflict/avg_cosine": conf_kept["avg_cosine"],
					"conflict_to_mean/mean_cos": cos_mean_k,
					"conflict_to_mean/min_cos": cos_min_k,
					"conflict_to_mean/frac_below_threshold": frac_below,
				}
				
				# Add conflict penalty specific metrics
				if cfg.selection_method == "conflict_penalty":
					log_info.update({
						"conflict_penalty/weight": cfg.conflict_penalty,
						"conflict_penalty/mean": float(sum(conflict_penalties) / len(conflict_penalties)),
						"conflict_penalty/max": float(max(conflict_penalties)),
						"conflict_penalty/total": float(sum(conflict_penalties)),
					})
				last_delta_mean = log_info["theory/delta_mean"]
				last_eps_mean = log_info["theory/epsilon_mean"]
				last_loss_mean = log_info["train/loss_mean"]
				log_step(logger, train_step, log_info)

				# optional: stream raw vs spice sequences
				if cfg.dump_delta_sequences:
					# compute raw deltas list if available
					raw_deltas = [float(d) for _, d, _ in (marginals_raw or [])]
					spice_deltas = [float(x) for x in deltas_q]
					record = {
						"step": int(global_step),
						"raw_delta": raw_deltas,
						"spice_delta": spice_deltas,
					}
					with open(out_path, "a", encoding="utf-8") as f:
						json.dump(record, f, ensure_ascii=False)
						f.write("\n")
			else:
				log_info = {
					"train/step": float(train_step),  # Use actual training steps
					"train/selection_step": float(global_step),  # Add selection steps for comparison
					"train/epoch": float(last_epoch),
					"train/epoch_step": float(epoch_step),
					"train/loss_mean": 0.0,
					"train/response_length_mean": 0.0,
					"train/elapsed_sec": float(time.time() - start_time),
					"select/num_selected": 0.0,
					"select/num_kept": float(len(kept_rel)),
				}
				log_step(logger, train_step, log_info)

		# Save checkpoint if needed
		if accelerator.is_main_process and cfg.save_checkpoint_freq > 0:
			if global_step % cfg.save_checkpoint_freq == 0:
				save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch_idx, 
							   cfg.checkpoint_dir, cfg.save_total_limit)

		# update progress bar
		if accelerator.is_main_process:
			pbar.set_postfix({
				"epoch": last_epoch,
				"kept": last_kept,
				"loss": f"{(last_loss_mean if last_loss_mean is not None else 0):.3f}",
				"resp_len": f"{(response_length_mean if 'response_length_mean' in locals() else 0):.0f}",
				"Δmean": f"{(last_delta_mean if last_delta_mean is not None else 0):.4f}",
				"εmean": f"{(last_eps_mean if last_eps_mean is not None else 0):.4f}",
			})
			pbar.update(1)

		global_step += 1

		if global_step % 50 == 0:  # Clean up every 50 steps
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if accelerator.is_main_process:
					allocated = torch.cuda.memory_allocated() / 1024**3
					reserved = torch.cuda.memory_reserved() / 1024**3
					logger.info("Step %d: GPU Memory - Allocated: %.2fGB, Reserved: %.2fGB", global_step, allocated, reserved)

	# Save final checkpoint
	if accelerator.is_main_process and cfg.save_checkpoint_freq > 0:
		save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch_idx, 
					   cfg.checkpoint_dir, cfg.save_total_limit)

	if accelerator.is_main_process:
		pbar.close()
		logger.info("Training completed")
		
		total_time = float(time.time() - start_time)
		steps_per_sec = float(global_step / total_time) if total_time > 0 else 0.0
		# Output average number of selected samples per step (average k) at training end
		avg_k = float(selection_stats["total_selected"]) / float(max(1, selection_step))
		logger.info("Average samples selected per step: %.2f (selection steps: %d)", avg_k, selection_step)
		
		log_step(logger, train_step, {
			"train/total_time_sec": total_time,
			"train/steps_per_sec": steps_per_sec,
		})
	logger.close()
	os.makedirs(cfg.output_dir, exist_ok=True)
	
	# Save final selection data summary
	if accelerator.is_main_process:
		# Add final statistics
		selection_stats.update({
			"final_total_selected": len(all_selected_data),
			"training_completed": True,
			"total_training_time_sec": float(time.time() - start_time),
			"unique_indices": list(set(record["global_index"] for record in all_selected_data)),
			"selection_steps": int(selection_step),
			"average_k": float(selection_stats["total_selected"]) / float(max(1, selection_step)),
		})
		selection_stats["unique_selected_count"] = len(selection_stats["unique_indices"])
		
		# Save summary information
		with open(selected_data_summary_path, "w", encoding="utf-8") as f:
			json.dump(selection_stats, f, indent=2, ensure_ascii=False)
		



if __name__ == "__main__":
	main()