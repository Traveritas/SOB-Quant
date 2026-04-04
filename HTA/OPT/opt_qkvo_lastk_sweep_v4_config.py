from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from transformers import AutoConfig


# ============================================================
# User config area
# Modify experiment settings here directly.
# ============================================================
BASE_SCRIPT_PATH = Path(__file__).with_name('opt_all_blocks_qkvo_experiment_v1.py')
OUTPUT_ROOT = Path('./outputs_opt_qkvo_lastk_sweep_v4_config')

SWEEP_CONFIG = {
    'last_ks': (1, 2, 4, 8, 11),#1,2,4,8,11...
    'init_modes': ('pca', 'random'),#pca random
    'betas': (1.0,),
    'error_modes': ('relative',),#relative absolute
}

# If your base ExperimentConfig uses different field paths, adjust here first.
CONFIG_PATH_HINTS = {
    'block_indices': ('target.block_indices', 'target.layers', 'block_indices'),
    'output_dir': ('output_dir',),
    'init_mode': (
        'quant.init_mode',
        'quant.init_type',
        'optimization.init_mode',
        'optimizer.init_mode',
        'init_mode',
        'initialization',
    ),
    'beta': (
        'quant.beta',
        'optimization.beta',
        'optimizer.beta',
        'train.beta',
        'beta',
    ),
    'error_mode': (
        'quant.error_mode',
        'quant.error_type',
        'optimization.error_mode',
        'train.error_mode',
        'metric.error_mode',
        'loss_type',
        'error_mode',
    ),
}

# Only useful when base config is not a dataclass and supports dynamic attrs.
ALLOW_DYNAMIC_ATTACH = False
# ============================================================


def load_base_module(script_path: Path):
    module_name = 'opt_all_blocks_qkvo_experiment_v1'
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import base script from: {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_last_k_values(model_name: str, requested_ks: Sequence[int]) -> Tuple[int, ...]:
    cfg = AutoConfig.from_pretrained(model_name)
    num_layers = int(getattr(cfg, 'num_hidden_layers'))
    valid = tuple(sorted({int(k) for k in requested_ks if 1 <= int(k) <= num_layers}))
    if not valid:
        raise ValueError(f'No valid last-k values for model {model_name}; num_hidden_layers={num_layers}')
    return valid


def block_indices_for_last_k(num_layers: int, k: int) -> Tuple[int, ...]:
    start = num_layers - k
    return tuple(range(start, num_layers))


def sanitize_tag(value: Any) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ('-', '_', '.'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep)


def has_field(dc_obj: Any, field_name: str) -> bool:
    return is_dataclass(dc_obj) and any(f.name == field_name for f in fields(dc_obj))


def set_path_dataclass(obj: Any, path: Sequence[str], value: Any) -> Tuple[Any, bool]:
    if not path:
        return obj, False

    head = path[0]
    tail = path[1:]

    if is_dataclass(obj):
        if not has_field(obj, head):
            return obj, False
        if not tail:
            return replace(obj, **{head: value}), True
        child = getattr(obj, head)
        new_child, changed = set_path_dataclass(child, tail, value)
        if not changed:
            return obj, False
        return replace(obj, **{head: new_child}), True

    if hasattr(obj, head):
        if not tail:
            setattr(obj, head, value)
            return obj, True
        child = getattr(obj, head)
        new_child, changed = set_path_dataclass(child, tail, value)
        if changed and child is not new_child:
            try:
                setattr(obj, head, new_child)
            except Exception:
                pass
        return obj, changed

    return obj, False


def apply_first_matching_path(cfg: Any, candidate_paths: Sequence[str], value: Any) -> Tuple[Any, Optional[str]]:
    for path in candidate_paths:
        parts = tuple(path.split('.'))
        new_cfg, changed = set_path_dataclass(cfg, parts, value)
        if changed:
            return new_cfg, path
    return cfg, None


def maybe_attach_top_level_attr(cfg: Any, attr_name: str, value: Any) -> Tuple[Any, bool]:
    if is_dataclass(cfg):
        return cfg, False
    try:
        setattr(cfg, attr_name, value)
        return cfg, True
    except Exception:
        return cfg, False


def configure_experiment(
    cfg: Any,
    block_indices: Tuple[int, ...],
    run_output_dir: Path,
    init_mode: str,
    beta: float,
    error_mode: str,
) -> Tuple[Any, Dict[str, str]]:
    applied: Dict[str, str] = {}

    cfg, path = apply_first_matching_path(cfg, CONFIG_PATH_HINTS['block_indices'], block_indices)
    if path is None:
        raise AttributeError(f"Could not find block_indices field. Tried: {CONFIG_PATH_HINTS['block_indices']}")
    applied['block_indices'] = path

    cfg, path = apply_first_matching_path(cfg, CONFIG_PATH_HINTS['output_dir'], str(run_output_dir))
    if path is None:
        raise AttributeError(f"Could not find output_dir field. Tried: {CONFIG_PATH_HINTS['output_dir']}")
    applied['output_dir'] = path

    cfg, path = apply_first_matching_path(cfg, CONFIG_PATH_HINTS['init_mode'], init_mode)
    if path is None and ALLOW_DYNAMIC_ATTACH:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'init_mode', init_mode)
        if ok:
            path = 'init_mode'
    if path is None:
        raise AttributeError(f"Could not find init_mode field. Tried: {CONFIG_PATH_HINTS['init_mode']}")
    applied['init_mode'] = path

    cfg, path = apply_first_matching_path(cfg, CONFIG_PATH_HINTS['beta'], float(beta))
    if path is None and ALLOW_DYNAMIC_ATTACH:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'beta', float(beta))
        if ok:
            path = 'beta'
    if path is None:
        raise AttributeError(f"Could not find beta field. Tried: {CONFIG_PATH_HINTS['beta']}")
    applied['beta'] = path

    cfg, path = apply_first_matching_path(cfg, CONFIG_PATH_HINTS['error_mode'], error_mode)
    if path is None and ALLOW_DYNAMIC_ATTACH:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'error_mode', error_mode)
        if ok:
            path = 'error_mode'
    if path is None:
        raise AttributeError(f"Could not find error_mode field. Tried: {CONFIG_PATH_HINTS['error_mode']}")
    applied['error_mode'] = path

    return cfg, applied


def build_summary_rows(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in results:
        rows.append(
            {
                'last_k': int(r['last_k']),
                'block_indices': list(r['block_indices']),
                'init_mode': str(r['init_mode']),
                'beta': float(r['beta']),
                'error_mode': str(r['error_mode']),
                'baseline_ppl': float(r['baseline_ppl']),
                'ours_ppl': float(r['ours_ppl']),
                'sq_ppl': float(r['sq_ppl']),
                'ours_delta': float(r['ours_ppl']) - float(r['baseline_ppl']),
                'sq_delta': float(r['sq_ppl']) - float(r['baseline_ppl']),
                'output_dir': str(r['output_dir']),
                'applied_paths': dict(r['applied_paths']),
            }
        )
    return rows


def group_results(results: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in results:
        key = f"init={r['init_mode']} | beta={r['beta']} | error={r['error_mode']}"
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        group.sort(key=lambda x: int(x['last_k']))
    return groups


def save_group_plots(output_dir: Path, results: List[Dict[str, object]]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    grouped = group_results(results)

    plt.figure(figsize=(10, 6))
    baseline_drawn = False
    for label, group in grouped.items():
        xs = [int(r['last_k']) for r in group]
        ys = [float(r['ours_ppl']) for r in group]
        plt.plot(xs, ys, marker='o', linewidth=1.8, label=label)
        if not baseline_drawn:
            baseline = [float(r['baseline_ppl']) for r in group]
            plt.plot(xs, baseline, marker='o', linewidth=1.8, linestyle='--', label='FP baseline')
            baseline_drawn = True
    plt.xlabel('Number of last blocks quantized')
    plt.ylabel('Perplexity')
    plt.title('OPT last-k QKVO sweep across init / beta / error settings')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    linear_path = output_dir / 'ppl_curve_by_meta_linear.png'
    plt.tight_layout()
    plt.savefig(linear_path, dpi=180)
    plt.close()
    paths['linear'] = str(linear_path)

    plt.figure(figsize=(10, 6))
    baseline_drawn = False
    for label, group in grouped.items():
        xs = [int(r['last_k']) for r in group]
        ys = [float(r['ours_ppl']) for r in group]
        plt.plot(xs, ys, marker='o', linewidth=1.8, label=label)
        if not baseline_drawn:
            baseline = [float(r['baseline_ppl']) for r in group]
            plt.plot(xs, baseline, marker='o', linewidth=1.8, linestyle='--', label='FP baseline')
            baseline_drawn = True
    plt.xlabel('Number of last blocks quantized')
    plt.ylabel('Perplexity (log scale)')
    plt.title('OPT last-k QKVO sweep across init / beta / error settings (log scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    log_path = output_dir / 'ppl_curve_by_meta_log.png'
    plt.tight_layout()
    plt.savefig(log_path, dpi=180)
    plt.close()
    paths['log'] = str(log_path)

    return paths


def build_text_summary(results: List[Dict[str, object]], plot_paths: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append('===== Last-k blocks QKVO PPL meta sweep =====')
    lines.append('')
    for r in results:
        lines.append(
            f"last {int(r['last_k'])} blocks | block_indices={list(r['block_indices'])} | "
            f"init={r['init_mode']} | beta={float(r['beta']):.6g} | error={r['error_mode']} | "
            f"baseline={float(r['baseline_ppl']):.6f} | "
            f"ours={float(r['ours_ppl']):.6f} | "
            f"sq={float(r['sq_ppl']):.6f} | "
            f"ours_delta={float(r['ours_ppl']) - float(r['baseline_ppl']):.6f} | "
            f"sq_delta={float(r['sq_ppl']) - float(r['baseline_ppl']):.6f}"
        )
    lines.append('')
    lines.append('Saved plots:')
    for k, v in plot_paths.items():
        lines.append(f'- {k}: {v}')
    return '\n'.join(lines)


def validate_user_config() -> None:
    required_keys = ('last_ks', 'init_modes', 'betas', 'error_modes')
    for key in required_keys:
        if key not in SWEEP_CONFIG:
            raise KeyError(f'Missing SWEEP_CONFIG[{key!r}]')
        if not SWEEP_CONFIG[key]:
            raise ValueError(f'SWEEP_CONFIG[{key!r}] cannot be empty')



def main() -> None:
    validate_user_config()
    base = load_base_module(BASE_SCRIPT_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    base_config = base.ExperimentConfig()
    valid_last_ks = resolve_last_k_values(base_config.data.model_name, SWEEP_CONFIG['last_ks'])
    num_layers = int(AutoConfig.from_pretrained(base_config.data.model_name).num_hidden_layers)

    results: List[Dict[str, object]] = []

    combos = list(itertools.product(
        valid_last_ks,
        SWEEP_CONFIG['init_modes'],
        SWEEP_CONFIG['betas'],
        SWEEP_CONFIG['error_modes'],
    ))

    for last_k, init_mode, beta, error_mode in combos:
        block_indices = block_indices_for_last_k(num_layers, int(last_k))
        run_name = (
            f'last_{last_k}_blocks__init_{sanitize_tag(init_mode)}'
            f'__beta_{sanitize_tag(beta)}__err_{sanitize_tag(error_mode)}'
        )
        run_output_dir = OUTPUT_ROOT / run_name

        cfg, applied_paths = configure_experiment(
            cfg=base.ExperimentConfig(),
            block_indices=block_indices,
            run_output_dir=run_output_dir,
            init_mode=str(init_mode),
            beta=float(beta),
            error_mode=str(error_mode),
        )

        artifacts = base.run_all_blocks_qkvo_experiment(cfg)
        results.append(
            {
                'last_k': int(last_k),
                'block_indices': block_indices,
                'init_mode': str(init_mode),
                'beta': float(beta),
                'error_mode': str(error_mode),
                'baseline_ppl': float(artifacts.baseline_ppl),
                'ours_ppl': float(artifacts.quantized_ppl),
                'sq_ppl': float(artifacts.sq_baseline_ppl),
                'output_dir': str(run_output_dir),
                'applied_paths': applied_paths,
            }
        )

    rows = build_summary_rows(results)
    plot_paths = save_group_plots(OUTPUT_ROOT, results)

    payload = {
        'model_name': base_config.data.model_name,
        'sweep_config': {
            'last_ks': list(SWEEP_CONFIG['last_ks']),
            'valid_last_ks': list(valid_last_ks),
            'init_modes': list(SWEEP_CONFIG['init_modes']),
            'betas': list(SWEEP_CONFIG['betas']),
            'error_modes': list(SWEEP_CONFIG['error_modes']),
        },
        'config_path_hints': {k: list(v) for k, v in CONFIG_PATH_HINTS.items()},
        'results': rows,
        'plot_paths': plot_paths,
    }

    with open(OUTPUT_ROOT / 'lastk_qkvo_meta_sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = build_text_summary(results, plot_paths)
    (OUTPUT_ROOT / 'lastk_qkvo_meta_sweep_summary.txt').write_text(summary, encoding='utf-8')

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('\n' + summary)


if __name__ == '__main__':
    main()
