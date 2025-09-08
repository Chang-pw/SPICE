from typing import Dict, List, Tuple
import torch
import math
from .adafisher import adafisher_h_vector, greedy_scores_diag, update_diag_den, cosine_similarity


def greedy_select_with_metrics(
	g_list: List[torch.Tensor],
	alpha: float,
	select_k: int,
	fisher_mode: str = "diag",
	halflife_threshold: float = None,
) -> Dict[str, any]:
	"""
	Original greedy selection method (only for delta sequences comparison)
	
	Args:
		g_list: List of gradients
		alpha: Fisher information regularization parameter
		select_k: Number of samples to select
		fisher_mode: Fisher mode ("diag" or "scalar")
		halflife_threshold: Halflife threshold
		
	Returns:
		Dictionary containing selection results and statistics
	"""
	device = g_list[0].device
	selected: List[int] = []
	remaining: List[int] = list(range(len(g_list)))
	# Precompute norms for base_x
	norm2 = [float((g * g).sum().item()) for g in g_list]
	base = [float(math.log(1.0 + alpha * n2)) for n2 in norm2]

	# diag denominators: start at ones
	if fisher_mode == "diag":
		den = torch.ones_like(g_list[0], device=device)
	else:
		den = None

	marginals: List[Tuple[int, float, float]] = []  # (idx, delta, eps)
	initial_gain = None

	for _ in range(min(select_k, len(remaining))):
		best_idx = None
		best_score = -1e38
		best_delta = None
		for ridx in remaining:
			g = g_list[ridx]
			if fisher_mode == "diag":
				s = greedy_scores_diag(g, den, alpha)
			else:
				# scalar approximation
				s = (g * g).sum() / (1.0 + alpha * 1.0)  # simple conservative denom
			if float(s) > best_score:
				best_score = float(s)
				# marginal gain = log(1 + alpha * s)
				best_delta = float(math.log(1.0 + alpha * float(s)))
				best_idx = ridx
		assert best_idx is not None
		
		# Halflife early stopping check
		if halflife_threshold is not None:
			if initial_gain is None:
				initial_gain = best_delta
			elif best_delta <= initial_gain * halflife_threshold:
				break
		
		selected.append(best_idx)
		remaining.remove(best_idx)
		# epsilon = delta - base
		epsilon = best_delta - base[best_idx]
		marginals.append((best_idx, best_delta, float(epsilon)))
		if fisher_mode == "diag":
			h = adafisher_h_vector(g_list[best_idx])
			den = update_diag_den(den, h, alpha)

	return {
		"selected": selected,
		"marginals": marginals,
		"base": base,
	}


def greedy_select_with_conflict_penalty(
	g_list: List[torch.Tensor],
	alpha: float,
	select_k: int,
	conflict_penalty: float = 0.1,
	fisher_mode: str = "diag",
) -> Dict[str, any]:
	"""
	Greedy selection method with conflict penalty
	
	Args:
		g_list: List of gradients
		alpha: Fisher information regularization parameter
		select_k: Number of samples to select
		conflict_penalty: Conflict penalty weight
		fisher_mode: Fisher mode ("diag" or "scalar")
		
	Returns:
		Dictionary containing selection results and statistics
	"""
	device = g_list[0].device
	selected: List[int] = []
	remaining: List[int] = list(range(len(g_list)))
	
	# Precompute norms for base_x
	norm2 = [float((g * g).sum().item()) for g in g_list]
	base = [float(math.log(1.0 + alpha * n2)) for n2 in norm2]

	# diag denominators: start at ones
	if fisher_mode == "diag":
		den = torch.ones_like(g_list[0], device=device)
	else:
		den = None

	marginals: List[Tuple[int, float, float, float]] = []  # (idx, delta, eps, conflict_penalty_val)
	selected_gradients = []  # For computing conflicts

	for _ in range(min(select_k, len(remaining))):
		best_idx = None
		best_score = -1e38
		best_delta = None
		best_conflict_penalty_val = 0.0
		
		for ridx in remaining:
			g = g_list[ridx]
			
			# 1. Calculate Fisher information gain
			if fisher_mode == "diag":
				s = greedy_scores_diag(g, den, alpha)
			else:
				# scalar approximation
				s = (g * g).sum() / (1.0 + alpha * 1.0)
			
			# Marginal gain = log(1 + alpha * s)
			marginal_gain = float(math.log(1.0 + alpha * float(s)))
			
			# 2. Calculate conflict penalty
			conflict_penalty_val = 0.0
			if selected_gradients:
				# Calculate conflict with average gradient of selected samples
				selected_mean = torch.stack(selected_gradients, dim=0).mean(0)
				cos_sim = cosine_similarity(g, selected_mean)
				
				# Conflict penalty: increase penalty when cosine similarity is negative
				if cos_sim < 0:
					conflict_penalty_val = conflict_penalty * abs(cos_sim)
			
			# 3. Total score = marginal gain - conflict penalty
			total_score = marginal_gain - conflict_penalty_val
			
			if total_score > best_score:
				best_score = total_score
				best_delta = marginal_gain
				best_conflict_penalty_val = conflict_penalty_val
				best_idx = ridx
		
		assert best_idx is not None
		selected.append(best_idx)
		remaining.remove(best_idx)
		selected_gradients.append(g_list[best_idx])
		
		# epsilon = delta - base
		epsilon = best_delta - base[best_idx]
		marginals.append((best_idx, best_delta, float(epsilon), best_conflict_penalty_val))
		
		# Update Fisher information matrix
		if fisher_mode == "diag":
			h = adafisher_h_vector(g_list[best_idx])
			den = update_diag_den(den, h, alpha)

	return {
		"selected": selected,
		"marginals": marginals,
		"base": base,
		"conflict_penalty": conflict_penalty,
	}


def top_k_select(
	g_list: List[torch.Tensor],
	alpha: float,
	select_k: int,
	fisher_mode: str = "diag",
	halflife_threshold: float = None,
) -> Dict[str, any]:
	"""
	Simplified top-k selection method (no iteration, directly select the top k samples with the largest target function value)
	
	Args:
		g_list: List of gradients
		alpha: Fisher information regularization parameter
		select_k: Number of samples to select
		fisher_mode: Fisher mode ("diag" or "scalar")
		
	Returns:
		Dictionary containing selection results and statistics
	"""
	device = g_list[0].device
	
	# Compute target function value for each sample
	scores = []
	for i, g in enumerate(g_list):
		if fisher_mode == "diag":
			# Use fixed Fisher information matrix (all 1)
			den = torch.ones_like(g, device=device)
			s = greedy_scores_diag(g, den, alpha)
		else:
			# scalar approximation
			s = (g * g).sum() / (1.0 + alpha * 1.0)
		
		# Marginal gain = log(1 + alpha * s)
		marginal_gain = float(math.log(1.0 + alpha * float(s)))
		scores.append((i, marginal_gain))
	
	# Sort by scores
	scores.sort(key=lambda x: x[1], reverse=True)
	# Support halflife early stopping (may not select full k)
	if halflife_threshold is not None and len(scores) > 0:
		initial_gain = scores[0][1]
		selected_indices = []
		for idx, gain in scores:
			if len(selected_indices) >= select_k:
				break
			if gain <= initial_gain * halflife_threshold:
				break
			selected_indices.append(idx)
	else:
		selected_indices = [idx for idx, _ in scores[:select_k]]
	
	# Compute baseline value (for epsilon calculation)
	norm2 = [float((g * g).sum().item()) for g in g_list]
	base = [float(math.log(1.0 + alpha * n2)) for n2 in norm2]
	
	# Build marginals information
	marginals = []
	# Read scores from scores dictionary to avoid using sample index as scores index
	score_dict = dict(scores)
	for idx in selected_indices:
		delta = score_dict[idx]
		epsilon = delta - base[idx]
		marginals.append((idx, delta, float(epsilon)))
	
	return {
		"selected": selected_indices,
		"marginals": marginals,
		"base": base,
		"scores": scores,  # All sample scores (for debugging)
	}


def compute_conflict_metrics(selected_indices: List[int], g_list: List[torch.Tensor]) -> Dict[str, float]:
	"""
	Compute conflict metrics for selected samples
	
	Args:
		selected_indices: Indices of selected samples
		g_list: List of gradients for all samples
		
	Returns:
		Conflict statistics
	"""
	if not selected_indices:
		return {"conflict_mean": 0.0, "conflict_ratio": 0.0, "avg_cosine": 0.0}
	
	selected_gradients = [g_list[i] for i in selected_indices]
	
	# Calculate average gradient
	g_mean = torch.stack(selected_gradients, dim=0).mean(0)
	
	# Calculate cosine similarity between each sample and average gradient
	cosine_sims = []
	for g in selected_gradients:
		cos_sim = cosine_similarity(g, g_mean)
		cosine_sims.append(cos_sim)
	
	# Conflict statistics
	conflicts = [abs(cos_sim) for cos_sim in cosine_sims if cos_sim < 0]
	conflict_ratio = len(conflicts) / len(cosine_sims) if cosine_sims else 0.0
	conflict_mean = sum(conflicts) / len(conflicts) if conflicts else 0.0
	avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
	
	return {
		"conflict_mean": conflict_mean,
		"conflict_ratio": conflict_ratio,
		"avg_cosine": avg_cosine,
		"num_conflicts": len(conflicts),
		"total_samples": len(cosine_sims)
	}