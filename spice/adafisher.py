from typing import List, Tuple, Optional, Callable
import torch


def iter_trainable_params(model: torch.nn.Module):
	"""Yield trainable parameters that have computed gradients."""
	for p in model.parameters():
		if p.requires_grad and p.grad is not None:
			yield p


def flatten_grad_vector(model: torch.nn.Module) -> torch.Tensor:
	"""Flatten all trainable gradients into a single 1-D vector."""
	grads = []
	for p in model.parameters():
		if p.requires_grad and p.grad is not None:
			grads.append(p.grad.detach().reshape(-1))
	if not grads:
		return torch.tensor([], device=next(model.parameters()).device)
	return torch.cat(grads, dim=0)


def compute_per_sample_grads(
	model: torch.nn.Module,
	batch_examples: List[dict],
	forward_fn,
	zero_grad_fn,
	device: torch.device,
	no_sync_ctx: Optional[Callable[[torch.nn.Module], torch.cuda.amp.autocast]] = None,
) -> List[torch.Tensor]:
	"""Compute per-sample gradient vectors sequentially (one forward-backward per sample)."""
	g_list: List[torch.Tensor] = []
	for ex in batch_examples:
		zero_grad_fn()
		if no_sync_ctx is not None:
			with no_sync_ctx(model):
				loss = forward_fn([ex])
				loss.backward()
		else:
			loss = forward_fn([ex])
			loss.backward()
		g_vec = flatten_grad_vector(model)
		g_list.append(g_vec)
		zero_grad_fn()
	return g_list


def compute_per_sample_grads_batch_optimized(
	model: torch.nn.Module,
	batch_examples: List[dict],
	forward_fn,
	zero_grad_fn,
	device: torch.device,
	no_sync_ctx: Optional[Callable[[torch.nn.Module], torch.cuda.amp.autocast]] = None,
	batch_size: int = 8,  # Internal batch size
) -> List[torch.Tensor]:
	"""
	Batch-optimized per-sample gradient computation
	
	Args:
		model: Model
		batch_examples: List of examples
		forward_fn: Forward function
		zero_grad_fn: Gradient zero function
		device: Device
		no_sync_ctx: Distributed training context
		batch_size: Internal batch size for balancing memory and speed
		
	Returns:
		List of gradient vectors for each sample
	"""
	g_list: List[torch.Tensor] = []
	
	# Process samples in batches
	for i in range(0, len(batch_examples), batch_size):
		batch = batch_examples[i:i + batch_size]
		
		# Method 1: Use retain_graph=True for batch computation
		if len(batch) == 1:
			# Single sample, use original method
			zero_grad_fn()
			if no_sync_ctx is not None:
				with no_sync_ctx(model):
					loss = forward_fn(batch)
					loss.backward()
			else:
				loss = forward_fn(batch)
				loss.backward()
			g_vec = flatten_grad_vector(model)
			g_list.append(g_vec)
			zero_grad_fn()
		else:
			# Multiple samples, use batch optimization
			zero_grad_fn()
			
			# Calculate loss for each sample
			losses = []
			for ex in batch:
				if no_sync_ctx is not None:
					with no_sync_ctx(model):
						loss = forward_fn([ex])
				else:
					loss = forward_fn([ex])
				losses.append(loss)
			
			# Backward propagate one by one and extract gradients
			for j, loss in enumerate(losses):
				if j == 0:
					# First backward propagation
					loss.backward(retain_graph=(j < len(losses) - 1))
				else:
					# Subsequent backward propagation, accumulate gradients
					loss.backward(retain_graph=(j < len(losses) - 1))
				
				# Extract gradient for current sample
				g_vec = flatten_grad_vector(model)
				g_list.append(g_vec)
				
				# If not the last sample, zero gradients for next one
				if j < len(losses) - 1:
					zero_grad_fn()
			
			# Finally zero gradients
			zero_grad_fn()
	
	return g_list


def adafisher_h_vector(g_vec: torch.Tensor) -> torch.Tensor:
	"""Compute the AdaFisher diagonal element: h = |g| * g."""
	return g_vec.abs() * g_vec


def greedy_scores_diag(g_vec: torch.Tensor, diag_den: torch.Tensor, alpha: float) -> torch.Tensor:
	"""Compute greedy score under diagonal Fisher approximation: sum_j g_j^2 / den_j."""
	return (g_vec * g_vec / diag_den).sum()


def update_diag_den(diag_den: torch.Tensor, h_vec: torch.Tensor, alpha: float) -> torch.Tensor:
	"""Update diagonal denominator: den <- den + alpha * h^2."""
	return diag_den + alpha * (h_vec * h_vec)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
	"""Compute cosine similarity between two 1-D tensors."""
	num = torch.dot(a, b).item()
	den = (a.norm() * b.norm()).item() + eps
	return float(num / den)


def project_gradients_to_low_dim(
	g_list: List[torch.Tensor], 
	projection_dim: int = 8192,
	device: torch.device = None
) -> List[torch.Tensor]:
	"""
	Project gradients to low-dimensional space to reduce computation
	
	Args:
		g_list: Original gradient list
		projection_dim: Projection dimension
		device: Device
		
	Returns:
		Projected gradient list
	"""
	if device is None:
		device = g_list[0].device
	
	# Get original gradient dimension
	original_dim = g_list[0].shape[0]
	
	# If original dimension is already small, return directly
	if original_dim <= projection_dim:
		return g_list
	
	# Create random projection matrix (fixed seed for reproducibility)
	torch.manual_seed(42)
	projection_matrix = torch.randn(original_dim, projection_dim, device=device)
	projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)  # Normalize
	
	# Project all gradients
	projected_g_list = []
	for g in g_list:
		projected_g = torch.matmul(g, projection_matrix)
		projected_g_list.append(projected_g)
	
	return projected_g_list


def compute_per_sample_grads_with_projection(
	model: torch.nn.Module,
	batch_examples: List[dict],
	forward_fn,
	zero_grad_fn,
	device: torch.device,
	no_sync_ctx: Optional[Callable[[torch.nn.Module], torch.cuda.amp.autocast]] = None,
	projection_dim: int = 8192,  # Projection dimension
) -> List[torch.Tensor]:
	"""
	Compute per-sample gradients and project to low-dimensional space
	
	Args:
		model: Model
		batch_examples: List of examples
		forward_fn: Forward function
		zero_grad_fn: Gradient zero function
		device: Device
		no_sync_ctx: Distributed training context
		projection_dim: Projection dimension
		
	Returns:
		List of projected gradient vectors
	"""
	# Compute original gradients
	g_list = compute_per_sample_grads(
		model, batch_examples, forward_fn, zero_grad_fn, device, no_sync_ctx
	)
	
	# Project to low-dimensional space
	projected_g_list = project_gradients_to_low_dim(g_list, projection_dim, device)
	
	return projected_g_list


def compute_per_sample_grads_batch_optimized_with_projection(
	model: torch.nn.Module,
	batch_examples: List[dict],
	forward_fn,
	zero_grad_fn,
	device: torch.device,
	no_sync_ctx: Optional[Callable[[torch.nn.Module], torch.cuda.amp.autocast]] = None,
	batch_size: int = 8,
	projection_dim: int = 8192,  # Projection dimension
) -> List[torch.Tensor]:
	"""
	Batch-optimized per-sample gradient computation + projection to low-dimensional space
	"""
	# Compute original gradients
	g_list = compute_per_sample_grads_batch_optimized(
		model, batch_examples, forward_fn, zero_grad_fn, device, no_sync_ctx, batch_size
	)
	
	# Project to low-dimensional space
	projected_g_list = project_gradients_to_low_dim(g_list, projection_dim, device)
	
	return projected_g_list