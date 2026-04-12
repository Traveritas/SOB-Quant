# QKVO Refactor

这是基于以下两个现有脚本整理出来的简化可运行版本：

- `HTA/OPT/opt_qkvo_refactored_array.py`
- `opt_all_blocks_qkvo_experiment_v1_ipreg.py`

目标不是保留所有实验分支，而是把真正的核心流程拆清楚：

- 配置定义
- QKVO 量化算法
- OPT 目标层定位与激活收集
- SQ baseline
- PPL 评测
- 单次实验入口

运行实验仍然需要现有脚本依赖的环境，至少包括：

- `torch`
- `transformers`
- `datasets`
- `matplotlib`

## 目录结构

- `config.py`: 配置 dataclass 和 CLI 解析辅助函数
- `common.py`: 通用工具函数
- `quantizer.py`: 核心量化器
- `model_utils.py`: 模型、数据、模块替换和线性层包装
- `experiment.py`: 单次实验主流程
- `cli.py`: 命令行入口
- `sweep.py`: 独立扫参与追踪入口
- `sweep_config.py`: 推荐修改的 sweep 配置文件
- `SWEEP_PARAMETERS.md`: 所有可 sweep 参数的完整参考

## 运行方式

在仓库根目录执行：

```bash
python -m qkvo_refactor --model-name facebook/opt-125m --output-dir ./qkvo_refactor_outputs
```

一个更接近你当前脚本默认配置的例子：

```bash
python -m qkvo_refactor \
  --block-indices 8,9,10,11 \
  --target-linear-names q_proj,k_proj,v_proj,out_proj \
  --codebook 4b \
  --init-mode random \
  --error-mode relative \
  --ip-reg-gamma 0.0 \
  --output-dir ./qkvo_refactor_outputs
```

## Sweep 用法

如果你不喜欢长命令行，推荐直接修改：

- [sweep_config.py](</d:/Documents/HTA/College/Projects/LattiBoxiGusinini Quant/qkvo_refactor/sweep_config.py>)
- [SWEEP_PARAMETERS.md](</d:/Documents/HTA/College/Projects/LattiBoxiGusinini Quant/qkvo_refactor/SWEEP_PARAMETERS.md>)

改完后直接运行：

```bash
python -m qkvo_refactor.sweep
```

如果你想把 sweep 配置单独放到别的文件，也可以：

```bash
python -m qkvo_refactor.sweep --config ./my_sweep_config.py
```

或者用 JSON 配置文件：

```bash
python -m qkvo_refactor.sweep --config ./my_sweep_config.json
```

可以直接用独立入口做扫参：

```bash
python -m qkvo_refactor.sweep \
  --output-dir ./qkvo_sweep_outputs \
  --grid-json "{\"quant.beta\":[0.5,1.0],\"quant.init_mode\":[\"random\",\"pca\"],\"quant.codebook\":[\"d5\",\"s8\"]}"
```

如果只想先看看会生成哪些组合：

```bash
python -m qkvo_refactor.sweep --list-runs
```

命令行现在更适合做少量覆盖，而不是承载整套配置。

如果想打开更细的追踪：

```bash
python -m qkvo_refactor.sweep \
  --output-dir ./qkvo_sweep_outputs \
  --track-u \
  --track-u-every 1 \
  --track-u-full-matrix \
  --save-u-trace-plots
```

## 输出内容

运行后会在输出目录生成：

- `experiment.log`
- `results.json`
- `summary.txt`
- `plots/` 下的损失曲线图

Sweep 模式还会额外生成：

- `sweep.log`
- `manifest.json`
- `summary.json`
- `ranking.txt`
- 每个 run 目录下的 `sweep_combo.json`
- 每个 run 目录下的 `tracking_summary.json`
- 如果启用了矩阵追踪，还会有 `tracking/u_matrices/`

## Slurm / Sbatch

如果你在超算平台上用 array job 跑 sweep，可以直接参考：

- [qkvo_refactor_sweep_array.slurm](</d:/Documents/HTA/College/Projects/LattiBoxiGusinini Quant/qkvo_refactor_sweep_array.slurm>)

新的 `sweep.py` 已经支持这些更适合 sbatch 的入口：

- `--print-num-combos`
- `--array-task-id`
- `--rebuild-summary`

典型流程：

```bash
NUM=$(python -m qkvo_refactor.sweep --config ./qkvo_refactor/sweep_config.py --print-num-combos)
sbatch --array=0-$((NUM-1))%4 qkvo_refactor_sweep_array.slurm
```

当所有 array task 跑完后，如果你想单独重建总汇总：

```bash
sbatch --dependency=afterok:<ARRAY_JOB_ID> --export=ALL,REBUILD_SUMMARY_ONLY=1 qkvo_refactor_sweep_array.slurm
```

## 和原脚本相比保留了什么

- 离散/连续 latent mode
- 可选 IP regularization
- SQ-XW baseline
- 全流程 PPL 评测
- 多个 block 的 Q/K/V/O 统一替换

## 主动省掉了什么

- sweep 逻辑
- U-trace / observer 追踪
- 每种实验分支各自堆在一个脚本里的写法

如果后面你还想把 sweep 再接回来，建议在这个版本上单独加一个 `sweep.py`，而不是继续往 `experiment.py` 里塞逻辑。
