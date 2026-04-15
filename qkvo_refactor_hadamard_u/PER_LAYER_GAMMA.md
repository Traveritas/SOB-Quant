## Per-layer gamma

`qkvo_refactor` now supports a global IP regularization strength plus optional
per-layer overrides.

### Config example

```python
EXPERIMENT_CONFIG_OVERRIDES = {
    "quant.ip_reg_gamma": 0.05,
    "quant.ip_reg_gamma_overrides": {
        "q_proj": 0.10,
        "k_proj": 0.20,
        "block8": 0.08,
        "block11.out_proj": 0.02,
    },
}
```

### Supported keys

- `"q_proj"`, `"k_proj"`, `"v_proj"`, `"out_proj"`: apply to all matching linears
- `"block8"`: apply to all selected target linears in one block
- `"block8.q_proj"`: apply to one exact layer

### Priority

1. exact layer name, such as `"block8.q_proj"`
2. block name, such as `"block8"`
3. linear name, such as `"q_proj"`
4. fallback to global `quant.ip_reg_gamma`

### CLI example

```powershell
python -m qkvo_refactor.experiment `
  --ip-reg-gamma 0.05 `
  --ip-reg-gamma-overrides "{\"q_proj\": 0.10, \"block11.out_proj\": 0.02}"
```

You can also pass `--ip-reg-gamma-overrides @path/to/gamma_overrides.json`.
