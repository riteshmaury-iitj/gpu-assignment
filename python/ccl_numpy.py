"""Pure-NumPy/SciPy reference: labels + statistics for a binary image (8-connectivity).
Uses scipy.ndimage.label if available; otherwise a BFS fallback. Same semantics as ccl_cpu.c."""
import numpy as np

def label_numpy(img):
    img = (np.asarray(img) != 0)
    try:
        from scipy import ndimage
        lab, K = ndimage.label(img, structure=np.ones((3, 3), int)); return lab.astype(np.int32), int(K)
    except ImportError:
        H, W = img.shape; lab = np.zeros((H, W), np.int32); K = 0
        for sy in range(H):
            for sx in range(W):
                if not img[sy, sx] or lab[sy, sx]: continue
                K += 1; st = [(sy, sx)]; lab[sy, sx] = K
                while st:
                    y, x = st.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < H and 0 <= nx < W and img[ny, nx] and not lab[ny, nx]:
                                lab[ny, nx] = K; st.append((ny, nx))
        return lab, K

def stats_numpy(lab):
    K = int(lab.max()); H, W = lab.shape; yy, xx = np.mgrid[0:H, 0:W]; flat = lab.ravel()
    area = np.bincount(flat, minlength=K+1); sx = np.bincount(flat, weights=xx.ravel(), minlength=K+1); sy = np.bincount(flat, weights=yy.ravel(), minlength=K+1)
    order = np.argsort(flat, kind='stable'); bounds = np.searchsorted(flat[order], np.arange(K+2)); out = {}
    for k in range(1, K+1):
        idx = order[bounds[k]:bounds[k+1]]; xs, ys = idx % W, idx // W
        out[k] = dict(area=int(area[k]), cx=sx[k]/area[k], cy=sy[k]/area[k], xmin=int(xs.min()), ymin=int(ys.min()), xmax=int(xs.max()), ymax=int(ys.max()))
    return out
