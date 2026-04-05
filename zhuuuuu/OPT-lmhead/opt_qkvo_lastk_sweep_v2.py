from __future__ import annotations

import importlib.util
import sys
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from transformers import AutoConfig


BASE_SCRIPT_PATH = Path(__file__).with_name('opt_all_blocks_qkvo_experiment_v1.py')
OUTPUT_ROOT = Path('./outputs_opt_qkvo_lastk_sweep_v1')
DEFAULT_LAST_KS: Tuple[int, ...] = (1,2,4,8,11)


def load_base_module(script_path: Path):
    module_name = 'opt_all_blocks_qkvo_experiment_v1'
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import base script from: {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_last_k_values(model_name: str, requested_ks: Tuple[int, ...]) -> Tuple[int, ...]:
    cfg = AutoConfig.from_pretrained(model_name)
    num_layers = int(getattr(cfg, 'num_hidden_layers'))
    valid = tuple(sorted({k for k in requested_ks if 1 <= k <= num_layers}))
    if not valid:
        raise ValueError(f'No valid last-k values for model {model_name}; num_hidden_layers={num_layers}')
    return valid


def block_indices_for_last_k(num_layers: int, k: int) -> Tuple[int, ...]:
    start = num_layers - k
    return tuple(range(start, num_layers))


def build_summary_rows(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in results:
        rows.append(
            {
                'last_k': int(r['last_k']),
                'block_indices': list(r['block_indices']),
                'baseline_ppl': float(r['baseline_ppl']),
                'ours_ppl': float(r['ours_ppl']),
                'sq_ppl': float(r['sq_ppl']),
                'ours_delta': float(r['ours_ppl']) - float(r['baseline_ppl']),
                'sq_delta': float(r['sq_ppl']) - float(r['baseline_ppl']),
                'output_dir': str(r['output_dir']),
            }
        )
    return rows


def save_plots(output_dir: Path, results: List[Dict[str, object]]) -> Dict[str, str]:
    xs = [int(r['last_k']) for r in results]
    baseline = [float(r['baseline_ppl']) for r in results]
    ours = [float(r['ours_ppl']) for r in results]
    sq = [float(r['sq_ppl']) for r in results]

    paths: Dict[str, str] = {}

    plt.figure(figsize=(8, 5))
    plt.plot(xs, baseline, marker='o', linewidth=1.8, label='FP baseline')
    plt.plot(xs, ours, marker='o', linewidth=1.8, label='Ours QKVO')
    plt.plot(xs, sq, marker='o', linewidth=1.8, label='SQ-XW QKVO')
    plt.xlabel('Number of last blocks quantized')
    plt.ylabel('Perplexity')
    plt.title('OPT last-k blocks QKVO perplexity sweep')
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    linear_path = output_dir / 'ppl_curve_linear.png'
    plt.tight_layout()
    plt.savefig(linear_path, dpi=180)
    plt.close()
    paths['linear'] = str(linear_path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, baseline, marker='o', linewidth=1.8, label='FP baseline')
    plt.plot(xs, ours, marker='o', linewidth=1.8, label='Ours QKVO')
    plt.plot(xs, sq, marker='o', linewidth=1.8, label='SQ-XW QKVO')
    plt.xlabel('Number of last blocks quantized')
    plt.ylabel('Perplexity (log scale)')
    plt.title('OPT last-k blocks QKVO perplexity sweep (log scale)')
    plt.yscale('log')
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    log_path = output_dir / 'ppl_curve_log.png'
    plt.tight_layout()
    plt.savefig(log_path, dpi=180)
    plt.close()
    paths['log'] = str(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, baseline, marker='o', linewidth=1.8, label='FP baseline')
    plt.plot(xs, ours, marker='o', linewidth=1.8, label='Ours QKVO')
    plt.xlabel('Number of last blocks quantized')
    plt.ylabel('Perplexity')
    plt.title('OPT last-k blocks QKVO perplexity sweep (Ours vs FP)')
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    ours_zoom_path = output_dir / 'ppl_curve_ours_vs_fp.png'
    plt.tight_layout()
    plt.savefig(ours_zoom_path, dpi=180)
    plt.close()
    paths['ours_vs_fp'] = str(ours_zoom_path)

    return paths


def build_text_summary(results: List[Dict[str, object]], plot_paths: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append('===== Last-k blocks QKVO PPL sweep =====')
    lines.append('')
    for r in results:
        lines.append(
            f"last {int(r['last_k'])} blocks | block_indices={list(r['block_indices'])} | "
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


def main() -> None:
    base = load_base_module(BASE_SCRIPT_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    base_config = base.ExperimentConfig()
    valid_last_ks = resolve_last_k_values(base_config.data.model_name, DEFAULT_LAST_KS)

    num_layers = AutoConfig.from_pretrained(base_config.data.model_name).num_hidden_layers
    results: List[Dict[str, object]] = []

    for last_k in valid_last_ks:
        block_indices = block_indices_for_last_k(num_layers, last_k)
        run_output_dir = OUTPUT_ROOT / f'last_{last_k}_blocks'

        cfg = replace(
            base_config,
            target=replace(base_config.target, block_indices=block_indices),
            output_dir=str(run_output_dir),
        )

        artifacts = base.run_all_blocks_qkvo_experiment(cfg)
        results.append(
            {
                'last_k': int(last_k),
                'block_indices': block_indices,
                'baseline_ppl': float(artifacts.baseline_ppl),
                'ours_ppl': float(artifacts.quantized_ppl),
                'sq_ppl': float(artifacts.sq_baseline_ppl),
                'output_dir': str(run_output_dir),
            }
        )

    rows = build_summary_rows(results)
    plot_paths = save_plots(OUTPUT_ROOT, results)

    payload = {
        'model_name': base_config.data.model_name,
        'requested_last_ks': list(DEFAULT_LAST_KS),
        'valid_last_ks': list(valid_last_ks),
        'results': rows,
        'plot_paths': plot_paths,
    }

    with open(OUTPUT_ROOT / 'lastk_qkvo_sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = build_text_summary(results, plot_paths)
    (OUTPUT_ROOT / 'lastk_qkvo_sweep_summary.txt').write_text(summary, encoding='utf-8')

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('\n' + summary)


if __name__ == '__main__':
    main()
