from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from transformers import AutoConfig


BASE_SCRIPT_PATH = Path(__file__).with_name('opt_all_blocks_qkvo_experiment_v1.py')
OUTPUT_ROOT = Path('./outputs_opt_qkvo_lastk_sweep_v3_meta')
DEFAULT_LAST_KS: Tuple[int, ...] = (1, 2, 4, 8, 11)
DEFAULT_INIT_MODES: Tuple[str, ...] = ('pca', 'random')
DEFAULT_BETAS: Tuple[float, ...] = (1.0,)
DEFAULT_ERROR_MODES: Tuple[str, ...] = ('relative',)


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


def parse_int_list(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(',') if x.strip())


def parse_float_list(raw: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(',') if x.strip())


def parse_str_list(raw: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(',') if x.strip())


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
    allow_dynamic_attach: bool,
) -> Tuple[Any, Dict[str, str]]:
    applied: Dict[str, str] = {}

    cfg, path = apply_first_matching_path(cfg, ('target.block_indices', 'target.layers', 'block_indices'), block_indices)
    if path is None:
        raise AttributeError('Could not find a block-indices field in ExperimentConfig.')
    applied['block_indices'] = path

    cfg, path = apply_first_matching_path(cfg, ('output_dir',), str(run_output_dir))
    if path is None:
        raise AttributeError('Could not find output_dir in ExperimentConfig.')
    applied['output_dir'] = path

    cfg, path = apply_first_matching_path(
        cfg,
        (
            'quant.init_mode',
            'quant.init_type',
            'optimization.init_mode',
            'optimizer.init_mode',
            'init_mode',
            'initialization',
        ),
        init_mode,
    )
    if path is None and allow_dynamic_attach:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'init_mode', init_mode)
        if ok:
            path = 'init_mode'
    if path is None:
        raise AttributeError(
            'Could not find a config field for initialization mode. '
            'Tried: quant.init_mode, quant.init_type, optimization.init_mode, optimizer.init_mode, init_mode, initialization.'
        )
    applied['init_mode'] = path

    cfg, path = apply_first_matching_path(
        cfg,
        (
            'quant.beta',
            'optimization.beta',
            'optimizer.beta',
            'train.beta',
            'beta',
        ),
        float(beta),
    )
    if path is None and allow_dynamic_attach:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'beta', float(beta))
        if ok:
            path = 'beta'
    if path is None:
        raise AttributeError(
            'Could not find a config field for beta. Tried: quant.beta, optimization.beta, optimizer.beta, train.beta, beta.'
        )
    applied['beta'] = path

    cfg, path = apply_first_matching_path(
        cfg,
        (
            'quant.error_mode',
            'quant.error_type',
            'optimization.error_mode',
            'train.error_mode',
            'metric.error_mode',
            'loss_type',
            'error_mode',
        ),
        error_mode,
    )
    if path is None and allow_dynamic_attach:
        cfg, ok = maybe_attach_top_level_attr(cfg, 'error_mode', error_mode)
        if ok:
            path = 'error_mode'
    if path is None:
        raise AttributeError(
            'Could not find a config field for error mode. '
            'Tried: quant.error_mode, quant.error_type, optimization.error_mode, train.error_mode, metric.error_mode, loss_type, error_mode.'
        )
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


def run_meta_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    base = load_base_module(Path(args.base_script))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = base.ExperimentConfig()
    valid_last_ks = resolve_last_k_values(base_config.data.model_name, args.last_ks)
    num_layers = int(AutoConfig.from_pretrained(base_config.data.model_name).num_hidden_layers)

    results: List[Dict[str, object]] = []

    combos = list(itertools.product(valid_last_ks, args.init_modes, args.betas, args.error_modes))
    for last_k, init_mode, beta, error_mode in combos:
        block_indices = block_indices_for_last_k(num_layers, int(last_k))
        run_name = (
            f'last_{last_k}_blocks__init_{sanitize_tag(init_mode)}'
            f'__beta_{sanitize_tag(beta)}__err_{sanitize_tag(error_mode)}'
        )
        run_output_dir = output_root / run_name

        cfg, applied_paths = configure_experiment(
            cfg=base.ExperimentConfig(),
            block_indices=block_indices,
            run_output_dir=run_output_dir,
            init_mode=init_mode,
            beta=float(beta),
            error_mode=error_mode,
            allow_dynamic_attach=args.allow_dynamic_attach,
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
    plot_paths = save_group_plots(output_root, results)

    payload = {
        'model_name': base_config.data.model_name,
        'requested_last_ks': list(args.last_ks),
        'valid_last_ks': list(valid_last_ks),
        'init_modes': list(args.init_modes),
        'betas': list(args.betas),
        'error_modes': list(args.error_modes),
        'results': rows,
        'plot_paths': plot_paths,
    }

    with open(output_root / 'lastk_qkvo_meta_sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = build_text_summary(results, plot_paths)
    (output_root / 'lastk_qkvo_meta_sweep_summary.txt').write_text(summary, encoding='utf-8')

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('\n' + summary)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sweep last-k QKVO settings over init_mode / beta / error_mode.')
    parser.add_argument('--base-script', type=str, default=str(BASE_SCRIPT_PATH))
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT))
    parser.add_argument('--last-ks', type=parse_int_list, default=DEFAULT_LAST_KS)
    parser.add_argument('--init-modes', type=parse_str_list, default=DEFAULT_INIT_MODES)
    parser.add_argument('--betas', type=parse_float_list, default=DEFAULT_BETAS)
    parser.add_argument('--error-modes', type=parse_str_list, default=DEFAULT_ERROR_MODES)
    parser.add_argument(
        '--allow-dynamic-attach',
        action='store_true',
        help='If a known config field is not found, allow attaching top-level attributes on non-dataclass configs.',
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_meta_sweep(args)


if __name__ == '__main__':
    main()
