# Sweep Parameters Reference

这份文档说明 `qkvo_refactor.sweep` 里哪些参数可以放进 `SWEEP_GRID` 做真正的扫参。

## 结论先说

`SWEEP_GRID` 目前只能扫 `ExperimentConfig` 里的字段。

也就是说，key 必须是下面这种 dotted path：

```python
SWEEP_GRID = {
    "quant.beta": (0.5, 1.0, 2.0),
    "quant.init_mode": ("random", "pca"),
    "target.block_indices": ((8, 9, 10, 11), (10, 11)),
}
```

实现上，这些 key 会直接写回：

- `ExperimentConfig`
- `DataConfig`
- `QuantizerConfig`
- `EvalConfig`
- `TargetConfig`

## 特殊解析规则

以下几个字段有额外解析逻辑：

- `quant.codebook`
  可以写 codebook 别名，比如 `"d5"`, `"s8"`, `"4b"`
  也可以直接写数值序列，比如 `(-2.0, -1.0, 0.0, 1.0, 2.0)`

- `target.block_indices`
  可以写 `None`
  也可以写整数 tuple，比如 `(8, 9, 10, 11)`
  通过命令行 / JSON 覆盖时，也支持 `"all"` 或 `"8,9,10,11"`

- `target.target_linear_names`
  推荐直接写 tuple，比如 `("q_proj", "k_proj", "v_proj", "out_proj")`
  通过命令行 / JSON 覆盖时，也支持逗号字符串

- `output_dir`
  可以 sweep，但通常不建议这么做
  因为 `sweep.py` 本身已经会给每个组合自动建独立 run 目录

## 全部可 sweep 参数

下面按配置分组列出当前所有可放入 `SWEEP_GRID` 的字段。

### `data.*`

- `data.model_name`
  类型：`str`
  示例：`"facebook/opt-125m"`

- `data.dataset_name`
  类型：`str`
  示例：`"wikitext"`

- `data.dataset_config`
  类型：`str`
  示例：`"wikitext-2-raw-v1"`

- `data.calib_split`
  类型：`str`
  示例：`"train"`

- `data.eval_split`
  类型：`str`
  示例：`"test"`

- `data.calib_num_tokens`
  类型：`int`
  示例：`2048`, `4096`, `8192`

- `data.eval_num_tokens`
  类型：`Optional[int]`
  示例：`None`, `4096`, `8192`

- `data.tokenizer_use_fast`
  类型：`bool`
  示例：`True`, `False`

### `quant.*`

- `quant.beta`
  类型：`float`
  说明：目标函数里 `J_w` 的权重

- `quant.max_iters`
  类型：`int`
  说明：量化迭代最大轮数

- `quant.tol`
  类型：`float`
  说明：收敛阈值

- `quant.convergence_check_every`
  类型：`int`
  说明：每隔多少轮检查一次收敛

- `quant.log_every`
  类型：`int`
  说明：每隔多少轮打印一次日志

- `quant.codebook`
  类型：`Codebook | str`
  示例：
  `("d5", "s8")`
  `((-2.0, -1.0, 0.0, 1.0, 2.0), (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))`

- `quant.dtype`
  类型：`str`
  常见值：`"float32"`, `"float64"`, `"float16"`, `"bfloat16"`

- `quant.eps`
  类型：`float`

- `quant.init_mode`
  类型：`str`
  常见值：`"random"`, `"pca"`

- `quant.error_mode`
  类型：`str`
  常见值：`"relative"`, `"absolute"`

- `quant.latent_mode`
  类型：`str`
  常见值：`"discrete"`, `"continuous"`

- `quant.ip_reg_gamma`
  类型：`float`
  说明：内积正则项强度

- `quant.ip_reg_inner_iters`
  类型：`int`
  说明：带 IP regularization 时 lambda 内层更新轮数

- `quant.fit_device`
  类型：`str`
  示例：`"cpu"`, `"cuda"`

### `eval.*`

- `eval.stride`
  类型：`int`
  说明：PPL sliding window 的 stride

- `eval.device`
  类型：`str`
  示例：`"cuda"`, `"cpu"`

### `target.*`

- `target.block_indices`
  类型：`Optional[Tuple[int, ...]]`
  示例：
  `None`
  `((8, 9, 10, 11), (10, 11))`

- `target.target_linear_names`
  类型：`Tuple[str, ...]`
  示例：
  `(("q_proj", "k_proj", "v_proj", "out_proj"), ("k_proj", "v_proj"))`

### 顶层 `ExperimentConfig`

- `output_dir`
  类型：`str`
  说明：技术上可 sweep，但通常不建议这么做

- `seed`
  类型：`int`

- `run_sq_baseline`
  类型：`bool`

- `save_plots`
  类型：`bool`

## 推荐 sweep 的参数

如果你是做量化实验，通常更值得 sweep 的是这些：

- `quant.beta`
- `quant.codebook`
- `quant.init_mode`
- `quant.error_mode`
- `quant.latent_mode`
- `quant.ip_reg_gamma`
- `quant.ip_reg_inner_iters`
- `quant.max_iters`
- `data.calib_num_tokens`
- `target.block_indices`
- `target.target_linear_names`

## 不建议 sweep 或很少 sweep 的参数

这些参数技术上能 sweep，但通常更适合固定：

- `output_dir`
- `eval.device`
- `quant.fit_device`
- `data.dataset_name`
- `data.dataset_config`
- `data.calib_split`
- `data.eval_split`
- `data.tokenizer_use_fast`
- `save_plots`

## 不能放进 `SWEEP_GRID` 的配置

下面这些是 sweep 的全局控制项，不是 per-run 的 `ExperimentConfig` 字段。

它们应该放在 `sweep_config.py` 的对应区块里，而不是放进 `SWEEP_GRID`：

- `TRACKING_CONFIG.*`
  例如：
  `track_u`
  `track_u_every`
  `track_u_full_matrix`
  `track_z_flip_stats`

- `OUTPUT_OPTIONS.*`
  例如：
  `save_manifest`
  `save_summary_json`
  `save_u_trace_plots`

- `RUN_CONTROL.*`
  例如：
  `run_index`
  `max_runs`

## 示例

### 例 1：扫 `beta + codebook + init_mode`

```python
SWEEP_GRID = {
    "quant.beta": (0.5, 1.0, 2.0),
    "quant.codebook": ("d5", "s8", "4b"),
    "quant.init_mode": ("random", "pca"),
}
```

### 例 2：扫不同 block 范围

```python
SWEEP_GRID = {
    "target.block_indices": (
        (8, 9, 10, 11),
        (10, 11),
        (11,),
    ),
    "quant.codebook": ("s8",),
}
```

### 例 3：扫离散/连续 latent

```python
SWEEP_GRID = {
    "quant.latent_mode": ("discrete", "continuous"),
    "quant.beta": (1.0,),
    "quant.codebook": ("s8",),
}
```

## 推荐做法

最推荐的方式还是直接改：

- [sweep_config.py](</d:/Documents/HTA/College/Projects/LattiBoxiGusinini Quant/qkvo_refactor/sweep_config.py>)

把你要 sweep 的组合写进 `SWEEP_GRID`，其他固定项写进 `EXPERIMENT_CONFIG_OVERRIDES`。
