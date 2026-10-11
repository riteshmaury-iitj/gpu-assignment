"""Verify a labelled image against the reference.
Usage: python verify.py image.bin labels_test.bin [stats_test.csv] [--ref-labels labels_ref.bin]
Component ids may differ between implementations, so we check that the PARTITION is identical
(one-to-one mapping test<->ref) and that per-component area / centroid / bbox agree."""
import sys, numpy as np
from ccl_io import read_image, read_labels, read_stats
from ccl_numpy import label_numpy, stats_numpy

def partition_equal(a, b):
    if a.shape != b.shape or not np.array_equal(a == 0, b == 0): return False
    fg = a != 0; pa, pb = a[fg].astype(np.int64), b[fg].astype(np.int64)
    pairs = np.unique(pa * (pb.max() + 1) + pb)
    return len(pairs) == len(np.unique(pa)) == len(np.unique(pb))

if __name__ == '__main__':
    args = [x for x in sys.argv[1:] if not x.startswith('--') and x != sys.argv[sys.argv.index('--ref-labels')+1] if '--ref-labels' in sys.argv] if '--ref-labels' in sys.argv else [x for x in sys.argv[1:] if not x.startswith('--')]
    img = read_image(args[0]); test = read_labels(args[1])
    if '--ref-labels' in sys.argv: ref = read_labels(sys.argv[sys.argv.index('--ref-labels') + 1]); K = int(ref.max())
    else: ref, K = label_numpy(img)
    ok = partition_equal(test, ref); print(f'components: test={int(test.max())} ref={K}  partition identical: {ok}')
    if len(args) > 2:
        st = read_stats(args[2]); rs = stats_numpy(ref); bad = 0
        first = {}
        for i, k in enumerate(test.ravel()):
            if k and k not in first: first[k] = i
            if len(first) == len(st): break
        for k in st:
            i = first[k]; r = int(ref.ravel()[i]); q = rs[r]
            if (st[k]['area'] != q['area'] or abs(st[k]['cx']-q['cx']) > 1e-3 or abs(st[k]['cy']-q['cy']) > 1e-3 or
                (st[k]['xmin'], st[k]['ymin'], st[k]['xmax'], st[k]['ymax']) != (q['xmin'], q['ymin'], q['xmax'], q['ymax'])): bad += 1
        print(f'statistics: {len(st)} components checked, {bad} mismatches'); ok = ok and bad == 0
    print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
