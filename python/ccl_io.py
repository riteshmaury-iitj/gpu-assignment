"""Binary I/O shared by all Python tools.
Image : int32 W, int32 H, uint8[W*H]   (0 = background, 1 = foreground)
Labels: int32 W, int32 H, int32[W*H]   (0 = background, 1..K = component id)
"""
import numpy as np

def write_image(path, img):
    img = np.asarray(img, dtype=np.uint8); H, W = img.shape
    with open(path, 'wb') as f:
        np.array([W, H], dtype=np.int32).tofile(f); (img != 0).astype(np.uint8).tofile(f)

def read_image(path):
    with open(path, 'rb') as f:
        W, H = np.fromfile(f, dtype=np.int32, count=2); return np.fromfile(f, dtype=np.uint8, count=W*H).reshape(H, W)

def read_labels(path):
    with open(path, 'rb') as f:
        W, H = np.fromfile(f, dtype=np.int32, count=2); return np.fromfile(f, dtype=np.int32, count=W*H).reshape(H, W)

def read_stats(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {int(r['id']): dict(area=int(r['area']), cx=float(r['cx']), cy=float(r['cy']),
            xmin=int(r['xmin']), ymin=int(r['ymin']), xmax=int(r['xmax']), ymax=int(r['ymax'])) for r in rows}
