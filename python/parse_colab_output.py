"""Summarise benchmark.csv into the table used in the report.
Usage: python parse_colab_output.py benchmark.csv benchmark_summary.csv"""
import sys, pandas as pd
df = pd.read_csv(sys.argv[1]); df['time_ms'] = df.time_ms.astype(float); df['iters'] = df.iters.astype(int)
g = df.groupby(['kind', 'size', 'mode']).agg(time_ms=('time_ms', 'min'), iters=('iters', 'max'), components=('components', 'first')).reset_index()
p = g.pivot_table(index=['kind', 'size'], columns='mode', values='time_ms').reset_index()
it = g.pivot_table(index=['kind', 'size'], columns='mode', values='iters').reset_index()
for m in ('gpu-compress', 'gpu-nocompress', 'cpu-prop'):
    if m in it: p['iters_' + m] = it[m].values
p['components'] = g[g['mode'] == g['mode'].iloc[0]].set_index(['kind', 'size'])['components'].reindex(list(zip(p['kind'], p['size']))).values
if 'cpu-bfs' in p and 'gpu-compress' in p: p['speedup_cpu/gpu'] = (p['cpu-bfs'] / p['gpu-compress']).round(1)
p.to_csv(sys.argv[2], index=False); print(p.to_string(index=False))
