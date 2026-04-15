from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import (
    ExperimentConfig,
    INIT_MODES,
    REORTH_METHODS,
    normalize_ip_reg_gamma_overrides,
    normalize_reorth_method,
    parse_block_indices,
    parse_codebook,
    parse_target_linear_names,
)


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--calib-num-tokens", type=int, default=4096)
    parser.add_argument("--eval-num-tokens", type=int, default=None)
    parser.add_argument("--block-indices", default="8,9,10,11", help="Comma-separated block indices, or 'all'.")
    parser.add_argument("--target-linear-names", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-pca", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--codebook", default="s8", help="Codebook alias or comma-separated float values.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--init-mode", choices=INIT_MODES, default="random")
    parser.add_argument("--error-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--latent-mode", choices=("discrete", "continuous"), default="discrete")
    parser.add_argument("--ip-reg-gamma", type=float, default=0.0)
    parser.add_argument(
        "--ip-reg-gamma-overrides",
        default=None,
        help=(
            "JSON object or @path/to/json mapping layer tag or linear name to gamma. "
            'Example: {"q_proj": 0.1, "block8.q_proj": 0.3}'
        ),
    )
    parser.add_argument("--ip-reg-inner-iters", type=int, default=1)
    parser.add_argument(
        "--lambda-quantile-init",
        dest="lambda_quantile_init_enable",
        action="store_true",
        help="Initialize lambda_x/lambda_w from projection quantiles instead of all-ones.",
    )
    parser.add_argument(
        "--lambda-quantile-rebalance",
        dest="lambda_quantile_rebalance_enable",
        action="store_true",
        help="Rebalance lambda_x/lambda_w from projection quantiles after each lambda update.",
    )
    parser.add_argument("--lambda-quantile-p", type=float, default=0.95)
    parser.add_argument("--lambda-quantile-rho", type=float, default=0.8)
    parser.add_argument("--lambda-quantile-alpha", type=float, default=0.0)
    parser.add_argument("--lambda-rebalance-ratio-min", type=float, default=0.8)
    parser.add_argument("--lambda-rebalance-ratio-max", type=float, default=1.25)
    parser.add_argument("--lambda-min-value", type=float, default=1e-4)
    parser.add_argument("--lambda-max-value", type=float, default=1e4)
    parser.add_argument("--fit-device", default="cpu")
    parser.add_argument("--log-orth-error", action="store_true", help="Log U orthogonality diagnostics after each U update.")
    parser.add_argument(
        "--reorth-after-u-update",
        action="store_true",
        help="Re-orthogonalize U immediately after every U update.",
    )
    parser.add_argument("--reorth-method", choices=REORTH_METHODS, default=REORTH_METHODS[0])
    parser.add_argument("--device", default=None, help="Runtime device for model eval. Defaults to torch auto detection.")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./qkvo_refactor_outputs")
    parser.add_argument("--skip-sq-baseline", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean QKVO refactor for OPT attention projection quantization.")
    return add_experiment_args(parser)


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()

    config.data.model_name = args.model_name
    config.data.dataset_name = args.dataset_name
    config.data.dataset_config = args.dataset_config
    config.data.calib_split = args.calib_split
    config.data.eval_split = args.eval_split
    config.data.calib_num_tokens = args.calib_num_tokens
    config.data.eval_num_tokens = args.eval_num_tokens

    config.target.block_indices = parse_block_indices(args.block_indices)
    config.target.target_linear_names = parse_target_linear_names(args.target_linear_names)

    config.quant.beta = args.beta
    config.quant.beta_pca = args.beta_pca
    config.quant.max_iters = args.max_iters
    config.quant.tol = args.tol
    config.quant.codebook = parse_codebook(args.codebook)
    config.quant.dtype = args.dtype
    config.quant.init_mode = args.init_mode
    config.quant.error_mode = args.error_mode
    config.quant.latent_mode = args.latent_mode
    config.quant.ip_reg_gamma = args.ip_reg_gamma
    if args.ip_reg_gamma_overrides is not None:
        config.quant.ip_reg_gamma_overrides = normalize_ip_reg_gamma_overrides(args.ip_reg_gamma_overrides)
    config.quant.ip_reg_inner_iters = args.ip_reg_inner_iters
    config.quant.lambda_quantile_init_enable = args.lambda_quantile_init_enable
    config.quant.lambda_quantile_rebalance_enable = args.lambda_quantile_rebalance_enable
    config.quant.lambda_quantile_p = args.lambda_quantile_p
    config.quant.lambda_quantile_rho = args.lambda_quantile_rho
    config.quant.lambda_quantile_alpha = args.lambda_quantile_alpha
    config.quant.lambda_rebalance_ratio_min = args.lambda_rebalance_ratio_min
    config.quant.lambda_rebalance_ratio_max = args.lambda_rebalance_ratio_max
    config.quant.lambda_min_value = args.lambda_min_value
    config.quant.lambda_max_value = args.lambda_max_value
    config.quant.fit_device = args.fit_device
    config.quant_ext.log_orth_error = args.log_orth_error
    config.quant_ext.reorth_after_u_update = args.reorth_after_u_update
    config.quant_ext.reorth_method = normalize_reorth_method(args.reorth_method)

    if args.device is not None:
        config.eval.device = args.device
    config.eval.stride = args.stride

    config.seed = args.seed
    config.output_dir = args.output_dir
    config.run_sq_baseline = not args.skip_sq_baseline
    config.save_plots = not args.no_plots
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from .experiment import build_summary, run_experiment

    config = args_to_config(args)
    artifacts = run_experiment(config)
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))
    print()
    print(build_summary(artifacts))


if __name__ == "__main__":
    main()
