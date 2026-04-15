from .config import (
    CODEBOOKS,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    QuantizerConfig,
    TargetConfig,
    normalize_ip_reg_gamma_overrides,
    parse_block_indices,
    parse_codebook,
    parse_target_linear_names,
)

__all__ = [
    "CODEBOOKS",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "QuantizerConfig",
    "TargetConfig",
    "normalize_ip_reg_gamma_overrides",
    "parse_block_indices",
    "parse_codebook",
    "parse_target_linear_names",
]


def __getattr__(name: str):
    if name in {"ExperimentArtifacts", "run_experiment"}:
        from .experiment import ExperimentArtifacts, run_experiment

        exported = {
            "ExperimentArtifacts": ExperimentArtifacts,
            "run_experiment": run_experiment,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
