# Spatial-VLA

## 项目简介

Spatial-VLA (Vision-Language-Action Model) 是一个基于视觉-语言-动作模型的机器人学习框架，支持多种机器人平台的数据集训练和推理。

## 项目结构

```
Spatial-VLA/
├── dataset/                    # 数据集模块
│   ├── datasets.py            # 数据集类（RLDSDataset, EpisodicRLDSDataset）
│   ├── data_manager.py        # 数据管理器，统一接口
│   └── utils/                 # 数据集工具类
│       ├── action_tokenizer.py    # 动作离散化和tokenization
│       ├── constants.py           # 常量定义（支持LIBERO/ALOHA/BRIDGE）
│       ├── data_utils.py          # 数据处理工具
│       ├── rlds/                  # RLDS数据集支持
│       └── oxe/                   # Open-X-Embodiment数据集支持
├── models/                    # 模型模块
│   ├── backbones/            # 骨干网络
│   │   ├── llm/             # 语言模型基类
│   │   └── vision/          # 视觉模型基类
│   ├── action_heads/        # 动作预测头
│   └── hf/                  # HuggingFace集成
└── overwatch/               # 日志和监控模块

```

## 核心功能

### 1. 数据集支持

本项目支持基于 RLDS (Robotics Learning Dataset Standard) 格式的数据集，包括：

- **Open-X-Embodiment (OXE)**: 支持多个机器人数据集的混合
- **多视角图像**: 支持主视角和手腕视角图像
- **动作分块 (Action Chunking)**: 支持预测未来多步动作
- **本体感知 (Proprioception)**: 支持关节位置等本体感知信息

#### 使用数据集

```python
from pathlib import Path
from dataset.data_manager import get_vla_dataset_and_collator

# 获取数据集和collator
train_dataset, val_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
    data_root_dir=Path("/path/to/dataset"),
    data_mix="bridge_oxe",  # 或者 "aloha_oxe", "libero_oxe"
    image_transform=image_transform,
    tokenizer=tokenizer,
    prompt_builder_fn=prompt_builder_fn,
    resize_resolution=(224, 224),
    shuffle_buffer_size=100_000,
    image_aug=True,  # 启用图像增强
    use_wrist_image=True,  # 使用手腕相机
    use_proprio=True,  # 使用本体感知
)
```

### 2. 动作离散化 (Action Tokenization)

将连续的机器人动作离散化为token，以便语言模型处理：

```python
from dataset.utils.action_tokenizer import ActionTokenizer

# 初始化动作tokenizer
action_tokenizer = ActionTokenizer(
    tokenizer=base_tokenizer,
    bins=256,  # 将每个动作维度离散化为256个bins
    min_action=-1,
    max_action=1
)

# 编码动作
action = np.array([0.5, -0.3, 0.8, 0.0, 1.0, -1.0, 0.2])
action_string = action_tokenizer(action)

# 解码动作
action_token_ids = [49935, 49744, 49999, 49871, ...]
decoded_action = action_tokenizer.covert_token_ids_to_actions(action_token_ids)
```

### 3. 多平台支持

项目支持多个机器人平台，会根据命令行参数自动检测并设置对应的常量：

| 平台 | 动作分块数 | 动作维度 | 本体感知维度 | 归一化方式 |
|------|-----------|---------|-------------|----------|
| LIBERO | 8 | 7 | 8 | BOUNDS_Q99 |
| ALOHA | 30 | 14 | 14 | BOUNDS |
| BRIDGE | 5 | 7 | 7 | BOUNDS_Q99 |


## 数据格式

### RLDS Batch格式

```python
{
    'observation': {
        'image_primary': np.ndarray,  # 主视角图像
        'image_wrist': np.ndarray,    # 手腕视角图像（可选）
        'proprio': np.ndarray,        # 本体感知（可选）
    },
    'action': np.ndarray,             # 动作序列 [window_size + future_chunk, action_dim]
    'task': {
        'language_instruction': bytes  # 语言指令
    },
    'dataset_name': str               # 数据集名称
}
```

### 模型输入格式

```python
{
    'pixel_values': torch.Tensor,       # [B, C, H, W] 图像
    'input_ids': torch.Tensor,          # [B, seq_len] token IDs
    'labels': torch.Tensor,             # [B, seq_len] 标签（动作部分为真实标签，其他为IGNORE_INDEX）
    'dataset_name': str,                # 数据集名称
    'actions': np.ndarray,              # 原始动作（用于评估）
    'pixel_values_wrist': torch.Tensor, # [B, C, H, W] 手腕图像（可选）
    'proprio': np.ndarray,              # 本体感知（可选）
}
```


## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请通过 Issue 联系我们。