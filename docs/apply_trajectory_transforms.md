# apply_trajectory_transforms 

## 函数位置
`dataset/utils/rlds/dataset.py:212-308`

## 函数作用
在轨迹（trajectory）级别对数据集进行转换和重标记。这些转换需要访问整个轨迹的数据，但不涉及CPU密集型操作（如图像解码），主要是数据的移动和复制。

---

## 📥 输入数据

### 输入参数：
```python
apply_trajectory_transforms(
    dataset: dl.DLataset,           # 输入的轨迹数据集
    train: bool,                    # 是否为训练集
    goal_relabeling_strategy: str,  # 目标重标记策略
    window_size: int = 1,           # 观测窗口大小
    future_action_window_size: int = 0,  # 未来动作窗口大小
    subsample_length: int = None,   # 子采样长度
    skip_unlabeled: bool = False,   # 是否跳过无标签数据
    max_action: float = None,       # 动作最大值
    max_proprio: float = None,      # 本体感知最大值
    task_augment_strategy: str = None,  # 任务增强策略
    ...
)
```

### 输入数据格式（单条轨迹）：
```python
{
    'observation': {
        'image_primary': tf.Tensor([T, H, W, 3], dtype=string),  # T个时间步的编码图像
        'image_wrist': tf.Tensor([T, H, W, 3], dtype=string),    # T个时间步的手腕图像
        'proprio': tf.Tensor([T, proprio_dim], dtype=float32),   # T个时间步的本体感知
        'timestep': tf.Tensor([T], dtype=int32),                 # 时间步索引
    },
    'action': tf.Tensor([T, action_dim], dtype=float32),         # T个时间步的动作
    'task': {
        'language_instruction': tf.Tensor([T], dtype=string),    # T个语言指令（通常相同）
    },
    'dataset_name': tf.Tensor([T], dtype=string),                # 数据集名称
    'absolute_action_mask': tf.Tensor([T, action_dim], dtype=bool),  # 绝对动作掩码
}
```
其中 `T` 是轨迹长度（例如 100 步）

---

## 🔄 处理流程详解

### 步骤 1: 过滤（Filtering）【第 258-268 行】

#### 1.1 跳过无标签数据
```python
if skip_unlabeled:
    dataset = dataset.filter(
        lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
    )
```
**作用**：移除没有语言指令的轨迹

**示例**：
- 输入轨迹 A: language_instruction = "pick up the cup"  ✅ 保留
- 输入轨迹 B: language_instruction = ""                ❌ 移除

#### 1.2 过滤异常动作
```python
if max_action is not None:
    dataset = dataset.filter(
        lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
    )
```
**作用**：移除动作值超过阈值的轨迹（可能是错误数据）

**示例**：max_action = 1.0
- 轨迹 A: actions = [0.5, -0.8, 0.3]  ✅ 保留
- 轨迹 B: actions = [0.5, 1.5, 0.3]   ❌ 移除（1.5 > 1.0）

### 步骤 2: 添加填充掩码（Add Padding Mask）【第 271 行】

```python
dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)
```

**作用**：为每个观测和任务字段添加掩码，标记哪些是真实数据，哪些是填充数据

**转换后的数据**：
```python
{
    'observation': {
        'image_primary': ...,
        'pad_mask_dict': {
            'image_primary': tf.Tensor([T], dtype=bool),  # [True, True, ..., True]
            'proprio': tf.Tensor([T], dtype=bool),
        }
    },
    'task': {
        'language_instruction': ...,
        'pad_mask_dict': {
            'language_instruction': tf.Tensor([T], dtype=bool),
        }
    },
    ...
}
```

### 步骤 3: 目标重标记（Goal Relabeling）【第 274-278 行】

```python
if goal_relabeling_strategy is not None:
    dataset = dataset.traj_map(
        partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
        num_parallel_calls,
    )
```

**作用**：为轨迹的每个时间步添加目标信息（goal）

**常用策略**：
- `"uniform"`: 从当前时间步到轨迹结束随机选择一个未来状态作为目标
- `"last"`: 使用轨迹最后一帧作为目标

**示例**（uniform策略）：
```python
# 原始轨迹（T=5步）
timestep:  0     1     2     3     4
state:    [s0]  [s1]  [s2]  [s3]  [s4]

# 重标记后（添加goal_image等）
timestep:  0     1     2     3     4
state:    [s0]  [s1]  [s2]  [s3]  [s4]
goal:     [s3]  [s4]  [s4]  [s4]  [s4]  # 随机从未来选择
```

### 步骤 4: 任务增强（Task Augmentation）【第 281-289 行】

```python
if train and task_augment_strategy is not None:
    dataset = dataset.traj_map(
        partial(getattr(task_augmentation, task_augment_strategy), **task_augment_kwargs),
        num_parallel_calls,
    )
```

**作用**：增强任务描述的多样性（仅在训练时）

**示例策略**：
- 随机替换语言指令的同义词
- 随机丢弃某些任务描述字段

### 步骤 5: 动作和观测分块（Chunking）【第 293-300 行】⭐️ **核心步骤**

```python
dataset = dataset.traj_map(
    partial(
        traj_transforms.chunk_act_obs,
        window_size=window_size,
        future_action_window_size=future_action_window_size,
    ),
    num_parallel_calls,
)
```

这是**最重要的一步**！让我详细解释：

#### 什么是分块（Chunking）？

**目的**：
1. 为模型提供历史观测上下文（过去的图像）
2. 为模型提供动作序列（action chunking）用于预测未来多步动作

**参数说明**：
- `window_size=1`: 观测窗口大小（当前观测）
- `future_action_window_size=7`: 未来动作窗口大小（预测未来7步）

#### 详细示例：

假设原始轨迹有 5 个时间步：

```python
# 输入（原始轨迹）
{
    'observation': {
        'image_primary': [img_0, img_1, img_2, img_3, img_4],  # shape: [5, H, W, 3]
    },
    'action': [a0, a1, a2, a3, a4],  # shape: [5, 7]
}

# 使用 window_size=2, future_action_window_size=2 进行分块
```

**分块索引计算**：

对于观测（window_size=2）：
```
timestep 0:  取 [img_0, img_0]  (第一个是padding，因为没有-1步)
timestep 1:  取 [img_0, img_1]
timestep 2:  取 [img_1, img_2]
timestep 3:  取 [img_2, img_3]  ← 最后一个有效timestep
# timestep 4 不输出，因为没有足够的future actions
```

对于动作（window_size=2, future=2）：
```
timestep 0:  取 [a0, a0, a1, a2]  (过去1步 + 当前 + 未来2步)
timestep 1:  取 [a0, a1, a2, a3]
timestep 2:  取 [a1, a2, a3, a4]
timestep 3:  取 [a2, a3, a4, a4]  (最后一个action重复)
```

**输出数据结构**：
```python
{
    'observation': {
        'image_primary': [
            [img_0, img_0],  # timestep 0
            [img_0, img_1],  # timestep 1
            [img_1, img_2],  # timestep 2
            [img_2, img_3],  # timestep 3
        ],  # shape: [4, 2, H, W, 3]
        'pad_mask': [
            [False, True],   # timestep 0: 第一个是padding
            [True, True],    # timestep 1
            [True, True],    # timestep 2
            [True, True],    # timestep 3
        ],  # shape: [4, 2]
    },
    'action': [
        [a0, a0, a1, a2],  # timestep 0
        [a0, a1, a2, a3],  # timestep 1
        [a1, a2, a3, a4],  # timestep 2
        [a2, a3, a4, a4],  # timestep 3
    ],  # shape: [4, 4, 7]
}
```

**关键变化**：
1. 轨迹长度从 5 变成 4（减少 future_action_window_size）
2. observation 增加一个维度：`[T, H, W, 3]` → `[T', window_size, H, W, 3]`
3. action 增加一个维度：`[T, action_dim]` → `[T', window_size+future, action_dim]`

#### 真实训练场景（OpenVLA）：

```python
# 配置
window_size = 1
future_action_window_size = 7  # NUM_ACTIONS_CHUNK - 1

# 输入轨迹长度：T = 100
# 输出轨迹长度：T' = 100 - 7 = 93

# 对于每个timestep t (0 <= t < 93)：
observation[t] = [image[t]]        # shape: [1, H, W, 3]
action[t] = [a[t], a[t+1], ..., a[t+7]]  # shape: [8, 7] - 预测当前+未来7步
```

### 步骤 6: 子采样（Subsampling）【第 302-306 行】

```python
if train and subsample_length is not None:
    dataset = dataset.traj_map(
        partial(traj_transforms.subsample, subsample_length=subsample_length),
        num_parallel_calls,
    )
```

**作用**：随机采样轨迹中的部分时间步，缩短轨迹长度

**示例**：subsample_length=50
```python
# 输入：93个分块后的时间步
# 输出：随机选择50个时间步
indices = random_shuffle([0, 1, 2, ..., 92])[:50]
traj = gather(traj, indices)
```

---

## 📤 输出数据

### 最终输出格式：

```python
{
    'observation': {
        'image_primary': tf.Tensor([T', window_size, H, W, 3], dtype=string),
        'image_wrist': tf.Tensor([T', window_size, H, W, 3], dtype=string),
        'proprio': tf.Tensor([T', window_size, proprio_dim], dtype=float32),
        'pad_mask': tf.Tensor([T', window_size], dtype=bool),
        'pad_mask_dict': {...},
    },
    'action': tf.Tensor([T', window_size+future, action_dim], dtype=float32),
    'task': {
        'language_instruction': tf.Tensor([T'], dtype=string),
        'goal_image': tf.Tensor([T', H, W, 3], dtype=string),  # 如果有goal relabeling
        'pad_mask_dict': {...},
    },
    'dataset_name': tf.Tensor([T'], dtype=string),
}
```

其中：
- `T' = T - future_action_window_size`（如果没有subsample）
- 每个observation和action都带有时间窗口维度

---

## 🎯 实际应用示例

### OpenVLA 配置：

```python
traj_transform_kwargs = dict(
    window_size=1,                      # 只使用当前观测
    future_action_window_size=7,        # 预测8步动作（当前+未来7步）
    skip_unlabeled=True,                # 跳过无标签数据
    goal_relabeling_strategy="uniform", # 使用uniform策略添加goal
)

# 输入：一条100步的轨迹
# 输出：93个训练样本，每个样本包含：
#   - 1个当前观测
#   - 8个连续动作（用于action chunking）
#   - 1个goal图像
#   - 1个语言指令
```

### 数据流示意图：

```
原始轨迹 (T=100步)
    ↓
[过滤] 移除无效数据
    ↓
[添加pad_mask] 标记填充数据
    ↓
[goal relabeling] 添加目标信息
    ↓
[chunking] ⭐️ 关键步骤
    → 观测: [100, H, W, 3] → [93, 1, H, W, 3]
    → 动作: [100, 7] → [93, 8, 7]
    ↓
[subsampling] 可选的长度截断
    ↓
输出 (T'=93个训练样本)
```

---

## 💡 关键要点总结

1. **输入**：完整的机器人轨迹（T个时间步）
2. **核心转换**：chunking（分块），将序列数据转换为带时间窗口的训练样本
3. **输出**：T' 个训练样本，每个包含观测窗口和动作序列
4. **长度变化**：T → T' = T - future_action_window_size
5. **维度增加**：所有时间序列数据都增加一个窗口维度

这个函数是训练 VLA 模型的关键预处理步骤，它将原始轨迹数据转换为适合模型训练的格式！
