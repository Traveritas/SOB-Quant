import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] # Use sans-serif font for Chinese
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign

sweep_dir = 'outputs_opt_qkvo_general_sweep_u_tracking'
summary_file = os.path.join(sweep_dir, 'sweep_summary.json')

with open(summary_file, 'r') as f:
    runs = json.load(f)

# parse runs
records = []
for run in runs:
    dir_path = run.get('run_dir', os.path.join(sweep_dir, run['run_name']))
    general_file = os.path.join(dir_path, 'general_sweep_results.json')
    if not os.path.exists(general_file):
        continue
        
    with open(general_file, 'r') as f:
        res = json.load(f)
        
    blocks = res['combo']['target.block_indices']
    k = len(blocks)
    beta = res['combo']['quant.beta']
    init_mode = res['combo']['quant_ext.init_mode']
    ppl = res['quantized_ppl']
    
    # parse quant metrics
    quant_metrics = res.get('quant_metrics', {})
    
    # average metrics for q, k, v, out
    for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        errs = {'w': [], 'x': [], 'xw': []}
        for b in blocks:
            layer_name = f'block{b}.{proj}'
            if layer_name in quant_metrics:
                m = quant_metrics[layer_name]
                # map the keys to w, x, xw 
                # According to previous read, keys are 'relative_recon_error_w', 'relative_recon_error_x', 'rel_linear_error'
                if 'relative_recon_error_w' in m:
                    errs['w'].append(m['relative_recon_error_w'])
                    errs['x'].append(m['relative_recon_error_x'])
                    errs['xw'].append(m['rel_linear_error'])
                elif 'rel_recon_error_w' in m:
                    errs['w'].append(m['rel_recon_error_w'])
                    errs['x'].append(m['rel_recon_error_x'])
                    errs['xw'].append(m['rel_linear_error'])
                
        if errs['w']:
            records.append({
                'k': k,
                'beta': beta,
                'init_mode': init_mode,
                'ppl': ppl,
                'proj': proj,
                'err_w': np.mean(errs['w']),
                'err_x': np.mean(errs['x']),
                'err_xw': np.mean(errs['xw'])
            })

df = pd.DataFrame(records)
df = df.sort_values(by='beta')

beta_values = sorted(df['beta'].astype(float).drop_duplicates().tolist())
non_zero_betas = [b for b in beta_values if b > 0]
min_nonzero = non_zero_betas[0] if non_zero_betas else 0.01

# Choose a pseudo-zero value for plotting that is nicely spaced in log scale
zero_plot_val = min_nonzero / 10.0

df['beta_plot'] = df['beta'].apply(lambda x: zero_plot_val if float(x) == 0.0 else float(x))
plot_ticks = [zero_plot_val] + non_zero_betas
plot_labels = ['0'] + [str(b) for b in non_zero_betas]

# 1. PPL vs Beta
figs_dir = os.path.join(sweep_dir, 'visualizations')
os.makedirs(figs_dir, exist_ok=True)

for k_val in df['k'].unique():
    subset = df[df['k'] == k_val].drop_duplicates(subset=['beta', 'init_mode'])
    if subset.empty: continue
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=subset, x='beta_plot', y='ppl', hue='init_mode', marker='o')
    plt.title(f'Beta 对 PPL 的影响 (Last {k_val} Blocks)')
    plt.xlabel('Beta')
    plt.xscale('log')
    plt.xticks(plot_ticks, plot_labels)
    plt.ylabel('Quantized PPL')
    plt.grid(True)
    plt.savefig(os.path.join(figs_dir, f'ppl_vs_beta_k{k_val}.png'), dpi=300)
    plt.close()

# 2. Errors vs Beta
# group by K and proj
for k_val in df['k'].unique():
    k_subset = df[df['k'] == k_val]
    if k_subset.empty: continue
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    projs = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    for i, proj in enumerate(projs):
        ax = axes[i]
        subset = k_subset[k_subset['proj'] == proj].drop_duplicates(subset=['beta', 'init_mode'])
        
        # We want to plot err_w, err_x, err_xw for pca and random. 
        # That's 6 lines. Let's reshape data.
        plot_data = []
        for _, row in subset.iterrows():
            plot_data.append({'beta_plot': row['beta_plot'], 'init_mode': row['init_mode'], 'err_type': 'w', 'error': row['err_w']})
            plot_data.append({'beta_plot': row['beta_plot'], 'init_mode': row['init_mode'], 'err_type': 'x', 'error': row['err_x']})
            plot_data.append({'beta_plot': row['beta_plot'], 'init_mode': row['init_mode'], 'err_type': 'xw', 'error': row['err_xw']})
        
        df_plot = pd.DataFrame(plot_data)
        if df_plot.empty: continue
        
        # combine init_mode and err_type into a single Hue
        df_plot['hue_col'] = df_plot['err_type'] + '-' + df_plot['init_mode']   
        
        sns.lineplot(data=df_plot, x='beta_plot', y='error', hue='hue_col', style='err_type', marker='o', ax=ax)
        ax.set_title(f'{proj} Reconstruction Errors')
        ax.set_xlabel('Beta')
        ax.set_xscale('log')
        ax.set_xticks(plot_ticks)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel('Relative Error')
        ax.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'Beta 对重构误差的影响 (Last {k_val} Blocks)', y=1.02, fontsize=16)
    plt.savefig(os.path.join(figs_dir, f'errors_vs_beta_k{k_val}.png'), dpi=300)
    plt.close()

print('Visualizations saved in', figs_dir)
