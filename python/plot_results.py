"""Plot benchmark.csv -> 3-panel figure: time vs size, speed-up vs size, iterations (compress vs none).
Usage: python plot_results.py benchmark.csv fig_gpu_benchmark.png"""
import sys, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1]); df['time_ms'] = df['time_ms'].astype(float); df['iters'] = df['iters'].astype(int)
g = df.groupby(['kind', 'size', 'mode']).agg(time_ms=('time_ms', 'min'), iters=('iters', 'max')).reset_index()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); kinds = sorted(g['kind'].unique()); col = dict(zip(kinds, ['C0', 'C1', 'C2']))
for kind in kinds:
    s = g[g['kind'] == kind]
    for mode, st in [('cpu-bfs', '--'), ('gpu-compress', '-'), ('gpu-nocompress', ':')]:
        q = s[s['mode'] == mode].sort_values('size')
        if len(q): ax[0].plot(q['size'], q['time_ms'], st, color=col[kind], marker='o', label=f'{kind} · {mode}')
    c = s[s['mode'] == 'cpu-bfs'].set_index('size')['time_ms']; gpu = s[s['mode'] == 'gpu-compress'].set_index('size')['time_ms']
    common = c.index.intersection(gpu.index)
    if len(common): ax[1].plot(common, c[common] / gpu[common], color=col[kind], marker='o', label=kind)
    for mode, st in [('gpu-compress', '-'), ('gpu-nocompress', ':'), ('cpu-prop', '-.')]:
        q = s[s['mode'] == mode].sort_values('size')
        if len(q) and q['iters'].max() > 0: ax[2].plot(q['size'], q['iters'], st, color=col[kind], marker='s', label=f'{kind} · {mode}')
ax[0].set(xscale='log', yscale='log', xlabel='image side (px)', ylabel='time (ms)', title='Total time: CPU BFS vs GPU'); ax[0].legend(fontsize=7)
ax[1].set(xscale='log', xlabel='image side (px)', ylabel='speed-up ×', title='GPU speed-up over CPU BFS'); ax[1].legend(fontsize=8); ax[1].axhline(1, color='k', lw=.5)
ax[2].set(xscale='log', yscale='log', xlabel='image side (px)', ylabel='iterations', title='Propagation iterations: compress vs none'); ax[2].legend(fontsize=7)
for a in ax: a.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=150); print('wrote', sys.argv[2])
