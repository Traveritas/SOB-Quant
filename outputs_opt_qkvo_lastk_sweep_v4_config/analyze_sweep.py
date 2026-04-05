import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set up matplotlib for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # for CJK support if needed
plt.rcParams['axes.unicode_minus'] = False

base_dir = 'outputs_opt_qkvo_lastk_sweep_v4_config'

# 1. Gather data for metrics vs. k
k_values = [1, 2, 4, 11]
init_methods = ['pca', 'random']
proj_types = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

data_metrics = {proj: {init: {k: {} for k in k_values} for init in init_methods} for proj in proj_types}

convergence_data = {
    'pca': {},
    'random': {}
}

for k in k_values:
    for init in init_methods:
        folder_name = f"last_{k}_blocks__init_{init}__beta_1.0__err_relative"
        file_path = os.path.join(base_dir, folder_name, "all_blocks_qkvo_results.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                quant_metrics = data.get('quant_metrics', {})
                
                proj_sums = {proj: {'rel_recon_error_x': 0, 'rel_recon_error_w': 0, 'rel_linear_error': 0, 'count': 0} for proj in proj_types}
                
                for layer_name, metrics in quant_metrics.items():
                    for proj in proj_types:
                        if layer_name.endswith(proj):
                            proj_sums[proj]['rel_recon_error_x'] += metrics.get('rel_recon_error_x', 0)
                            proj_sums[proj]['rel_recon_error_w'] += metrics.get('rel_recon_error_w', 0)
                            proj_sums[proj]['rel_linear_error'] += metrics.get('rel_linear_error', 0)
                            proj_sums[proj]['count'] += 1
                            break
                            
                for proj in proj_types:
                    count = proj_sums[proj]['count']
                    if count > 0:
                        data_metrics[proj][init][k]['rel_recon_error_x'] = proj_sums[proj]['rel_recon_error_x'] / count
                        data_metrics[proj][init][k]['rel_recon_error_w'] = proj_sums[proj]['rel_recon_error_w'] / count
                        data_metrics[proj][init][k]['rel_linear_error'] = proj_sums[proj]['rel_linear_error'] / count
                
                # For convergence, we'll take it from k=11 to get all blocks
                if k == 11:
                    convergence_data[init] = data.get('convergence_iters', {})

# Plot 1: Metrics vs Number of blocks (k)
metrics_to_plot = [
    ('rel_recon_error_x', 'Relative Recon Error X'),
    ('rel_recon_error_w', 'Relative Recon Error W'),
    ('rel_linear_error', 'Relative Linear Error')
]

colors = {'pca': 'blue', 'random': 'red'}
markers = {'pca': 'o', 'random': 'x'}

for proj in proj_types:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Errors vs Number of Last Blocks Quantized ({proj})', fontsize=16)
    
    for i, (metric_key, metric_title) in enumerate(metrics_to_plot):
        ax = axes[i]
        for init in init_methods:
            x_vals = []
            y_vals = []
            for k in sorted(k_values):
                if metric_key in data_metrics[proj][init][k]:
                    x_vals.append(k)
                    y_vals.append(data_metrics[proj][init][k][metric_key])
                    
            if x_vals:
                ax.plot(x_vals, y_vals, label=f'Init: {init}', color=colors[init], marker=markers[init], linestyle='-', linewidth=2)
                
        ax.set_title(metric_title)
        ax.set_xlabel('Block 数 (k)')
        if i == 0:
            ax.set_ylabel('重建误差')
        ax.set_xticks(k_values)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'metrics_vs_k_{proj}.png'), dpi=300)
    plt.close()

# Plot 2: Convergence Iterations per Layer
fig2, ax2 = plt.subplots(figsize=(16, 6))

all_layers = []
for block_num in range(1, 12):
    for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        all_layers.append(f"block{block_num}.{proj}")

for init in init_methods:
    iters_dict = convergence_data[init]
    y_vals = []
    
    for layer in all_layers:
        y_vals.append(iters_dict.get(layer, None))
            
    ax2.plot(range(len(all_layers)), y_vals, label=f'Init: {init}', linewidth=2)

ax2.set_title('Convergence Iterations per Layer', fontsize=14)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Iterations')
ax2.set_xticks(range(len(all_layers)))
ax2.set_xticklabels(all_layers, rotation=90)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'convergence_all_layers.png'), dpi=300)
plt.close()

print(f"Plots saved to {base_dir}")
