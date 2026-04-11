from __future__ import annotations

# ============================================================
# User config area
# ============================================================
#
# 你可以直接改这个文件，然后运行：
#
#   python -m qkvo_refactor.sweep
#
# 如果你更喜欢单独的配置文件，也可以复制这个文件并在运行时传：
#
#   python -m qkvo_refactor.sweep --config path/to/my_sweep_config.py


EXPERIMENT_CONFIG_OVERRIDES = {
    "data.model_name": "facebook/opt-125m",
    "data.dataset_name": "wikitext",
    "data.dataset_config": "wikitext-2-raw-v1",
    "data.calib_split": "train",
    "data.eval_split": "test",
    "data.calib_num_tokens": 4096,
    "data.eval_num_tokens": None,
    "target.block_indices": (8, 9, 10, 11),
    "target.target_linear_names": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "quant.beta": 1.0,
    "quant.max_iters": 80,
    "quant.tol": 1e-5,
    "quant.codebook": "d5",
    "quant.dtype": "float32",
    "quant.init_mode": "random",
    "quant.error_mode": "relative",
    "quant.latent_mode": "discrete",
    "quant.ip_reg_gamma": 0.0,
    "quant.ip_reg_inner_iters": 1,
    "quant.fit_device": "cpu",
    "eval.stride": 512,
    "output_dir": "./qkvo_sweep_outputs",
    "seed": 42,
    "run_sq_baseline": True,
    "save_plots": True,
}


SWEEP_GRID = {
    "quant.beta": (1.0,),
    "quant.init_mode": ("random", "pca"),
    "quant.error_mode": ("relative",),
    "quant.codebook": ("s8",),
}


TRACKING_CONFIG = {
    "track_u": False,
    "track_u_every": 1,
    "track_u_full_matrix": False,
    "track_u_save_interval": 10,
    "track_u_save_first": 5,
    "track_z_flip_stats": True,
}


OUTPUT_OPTIONS = {
    "save_manifest": True,
    "save_summary_json": True,
    "save_summary_text": True,
    "save_ranking_text": True,
    "save_run_combo_json": True,
    "save_run_tracking_json": True,
    "save_u_trace_plots": False,
}


RUN_CONTROL = {
    "run_index": None,
    "max_runs": None,
}
