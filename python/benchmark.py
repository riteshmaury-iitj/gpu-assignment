"""Benchmark sweep: GPU (with / without path compression) vs CPU BFS baseline.
Usage: python benchmark.py --exe ./ccl_cuda --cpu ./ccl_cpu --out benchmark.csv
                           [--sizes 256,512,1024,2048,4096] [--kinds blobs,noise,shapes] [--reps 3]"""
import argparse, subprocess, re, csv, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_image import blobs, noise, shapes; from ccl_io import write_image; import numpy as np

def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r'RESULT (.*)', out); return dict(kv.split('=', 1) for kv in m.group(1).split()) if m else None

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exe', default='./ccl_cuda'); ap.add_argument('--cpu', default='./ccl_cpu')
    ap.add_argument('--out', default='benchmark.csv'); ap.add_argument('--sizes', default='256,512,1024,2048,4096')
    ap.add_argument('--kinds', default='blobs,noise,shapes'); ap.add_argument('--reps', type=int, default=3); ap.add_argument('--gpu-only', action='store_true')
    a = ap.parse_args(); rows = []; tmp = tempfile.mkdtemp()
    for kind in a.kinds.split(','):
        for s in map(int, a.sizes.split(',')):
            rng = np.random.default_rng(42); img = {'blobs': blobs, 'noise': noise, 'shapes': shapes}[kind](s, s, 0.5 if kind == 'noise' else 0.3, rng)
            p = os.path.join(tmp, f'{kind}_{s}.bin'); write_image(p, img)
            modes = [('gpu-compress', []), ('gpu-nocompress', ['--no-compress'])] if os.path.exists(a.exe) else []
            for mode, extra in modes:
                for r in range(a.reps):
                    d = run([a.exe, '--image', p, '--out-labels', os.devnull, '--out-stats', os.devnull] + extra)
                    if d: d.update(kind=kind, size=s, mode=mode, rep=r); rows.append(d); print(kind, s, mode, r, d['time_ms'], 'ms', d['iters'], 'iters', flush=True)
            if not a.gpu_only and os.path.exists(a.cpu):
                for algo in ('bfs', 'prop'):
                    if algo == 'prop' and s > 1024: continue          # serial propagation is very slow on big images
                    d = run([a.cpu, '--image', p, '--algo', algo, '--out-labels', os.devnull, '--out-stats', os.devnull])
                    if d: d.update(kind=kind, size=s, mode='cpu-' + algo, rep=0); rows.append(d); print(kind, s, 'cpu-' + algo, d['time_ms'], 'ms', flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('wrote', a.out)
