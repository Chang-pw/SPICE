<p align="center">
  <img src="https://img.shields.io/badge/ICLR-2026-blue" alt="ICLR 2026">
  <a href="https://arxiv.org/abs/2601.23155"><img src="https://img.shields.io/badge/arXiv-2601.23155-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

# SPICE：基于次模惩罚信息-冲突选择的高效大语言模型训练

[English](README.md) | 中文

该工作已被 **ICLR 2026** 接收。

> **一句话总结：** SPICE 通过最大化 Fisher 信息同时惩罚梯度冲突来选择高质量训练子集。仅使用 10% 的数据，即可在 8 个基准测试上匹配甚至超越全量数据微调及 6 种基线方法。

## 概述

基于信息的数据选择在指令微调中具有天然优势：最大化 Fisher 信息矩阵的对数行列式是一个单调次模目标函数，贪心算法可以在基数约束下获得 $(1-1/e)$ 的近似保证。然而在实践中，样本间的梯度冲突会导致边际信息增益快速衰减，使得预算浪费在冗余或相互矛盾的数据上。

SPICE 通过以下方式解决这一问题：

- 提出 **ε-分解**，将偏离理想次模性的程度量化为冲突统计量的函数，给出随冲突减小而收紧的数据依赖近似因子
- 设计 **冲突感知的贪心选择器**，在最大化信息的同时惩罚梯度方向的不一致性
- 支持 **提前停止** 和 **代理模型**，提升计算效率

## 项目结构

```
SPICE/
├── train.py                 # 训练主入口
├── configs/
│   └── default.yaml         # 默认配置文件
├── spice/
│   ├── adafisher.py         # AdaFisher 梯度计算与投影
│   ├── config.py            # 配置管理与参数校验
│   ├── data.py              # 数据集加载（JSON/JSONL/Parquet）
│   ├── metrics.py           # 日志适配器（TensorBoard / SwanLab）
│   ├── models.py            # 模型加载与 LoRA 支持
│   └── select.py            # 选择算法（top-k、冲突惩罚）
├── train/
│   └── sft_data.sh          # 下游 SFT 训练脚本（LLaMA-Factory）
└── run.sh                   # 多卡启动示例脚本
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `configs/default.yaml` 指定模型、数据集和训练参数。所有参数均可通过命令行参数覆盖。

### 数据格式

训练数据使用 JSONL 格式：
```json
{"instruction": "你的指令", "response": "期望的回复"}
```

同时支持 Parquet 和 JSON 格式。

### 训练

```bash
# 单卡训练
python train.py --config configs/default.yaml

# 多卡训练（Accelerate）
accelerate launch --mixed_precision fp16 --multi_gpu --num_processes 4 train.py \
  --config configs/default.yaml
```

完整的多卡启动示例见 `run.sh`。

## 核心参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `selection_method` | `top_k` | 选择策略：`top_k` 或 `conflict_penalty` |
| `alpha_fisher` | `1.0` | Fisher 信息正则化系数 |
| `pool_size` | `128` | 每步选择的候选样本数 |
| `select_k` | `64` | 从候选池中选择的样本数（须 ≤ pool_size） |
| `conflict_penalty` | `0.1` | 梯度冲突惩罚权重 |
| `update_frequency` | `1` | 累积 N 步选择后执行一次优化器更新 |

## 下游 SFT

数据选择完成后，使用 `train/sft_data.sh` 配合 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行微调。该脚本会依次运行所有基线方法和 SPICE 选择的数据。

## 引用

如果本工作对您有帮助，请引用：

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

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
