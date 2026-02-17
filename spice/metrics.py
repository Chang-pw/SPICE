from typing import Dict, List, Optional
import torch


class LoggerAdapter:
	"""Unified logging adapter supporting TensorBoard and SwanLab backends."""

	def __init__(self, kind: str, logdir: Optional[str], project: Optional[str], run_name: Optional[str], is_main: bool):
		self.kind = kind
		self.is_main = is_main
		self.writer = None
		if not is_main:
			return
		if kind == "tensorboard":
			from torch.utils.tensorboard import SummaryWriter
			self.writer = SummaryWriter(log_dir=logdir)
		elif kind == "swanlab":
			import swanlab
			swanlab.init(project=project or "ick", experiment_name=run_name)
			self.writer = swanlab
		else:
			self.writer = None

	def add_scalar(self, tag: str, val: float, step: int):
		if not self.is_main or self.writer is None:
			return
		if self.kind == "tensorboard":
			self.writer.add_scalar(tag, val, step)
		elif self.kind == "swanlab":
			self.writer.log({tag: val}, step=step)

	def close(self):
		if not self.is_main or self.writer is None:
			return
		if self.kind == "tensorboard":
			self.writer.close()


def compute_conflict_metrics(selected: List[int], g_list: List[torch.Tensor]) -> Dict[str, float]:
	if not selected:
		return {"conflict_sum": 0.0, "conflict_mean": 0.0}
	G = [g_list[i] for i in selected]
	Gmat = torch.stack(G, dim=0)
	# pairwise inner products
	C = Gmat @ Gmat.t()  # [m, m]
	C2 = C * C
	conflict_sum = float(C2.sum().item())
	m = len(selected)
	conflict_mean = float(C2.sum().item() / (m * m))
	return {"conflict_sum": conflict_sum, "conflict_mean": conflict_mean}


def log_step(logger: LoggerAdapter, step: int, info: Dict[str, float]):
	for k, v in info.items():
		logger.add_scalar(k, v, step)