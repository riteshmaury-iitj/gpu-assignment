"""Generate synthetic binary test images.
Usage: python gen_image.py out.bin W H [--kind blobs|noise|shapes] [--density 0.3] [--seed 42] [--png out.png]
  blobs  : random discs (default) — realistic object-like components
  noise  : i.i.d. pixels with P(fg)=density — worst case, many tiny components
  shapes : rectangles, ring, diagonal lines, one long snake — stresses path compression
"""
import argparse, numpy as np
from ccl_io import write_image

def blobs(W, H, density, rng):
    img = np.zeros((H, W), np.uint8); target = density * W * H; yy, xx = np.mgrid[0:H, 0:W]
    rmax = max(3, min(W, H) // 40)
    while img.sum() < target:
        cx, cy, r = rng.integers(W), rng.integers(H), rng.integers(2, rmax + 1)
        y0, y1, x0, x1 = max(0, cy-r), min(H, cy+r+1), max(0, cx-r), min(W, cx+r+1)
        sub = (xx[y0:y1, x0:x1]-cx)**2 + (yy[y0:y1, x0:x1]-cy)**2 <= r*r
        img[y0:y1, x0:x1] |= sub.astype(np.uint8)
    return img

def noise(W, H, density, rng): return (rng.random((H, W)) < density).astype(np.uint8)

def shapes(W, H, density, rng):
    img = np.zeros((H, W), np.uint8)
    for _ in range(max(1, W*H // 20000)):                     # rectangles
        x, y = rng.integers(W-20), rng.integers(H-20); w, h = rng.integers(5, 20), rng.integers(5, 20)
        img[y:y+h, x:x+w] = 1
    for _ in range(max(1, W // 64)):                          # diagonal lines (8-connectivity!)
        x, y = rng.integers(W//2), rng.integers(H//2); n = rng.integers(10, min(W, H)//2)
        for i in range(n):
            if y+i < H and x+i < W: img[y+i, x+i] = 1
    yy, xx = np.mgrid[0:H, 0:W]; cx, cy, r = W//4, H//4, min(W, H)//8  # ring
    d = (xx-cx)**2 + (yy-cy)**2; img[(d <= r*r) & (d >= (r-3)**2)] = 1
    x, y = W//2, 2                                            # one long snake -> many iterations without compression
    while y < H-2 and 2 < x < W-2:
        img[y, x] = 1; x += int(rng.integers(3)) - 1; y += 1 if rng.random() < 0.5 else 0
    return img

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('W', type=int); ap.add_argument('H', type=int)
    ap.add_argument('--kind', default='blobs', choices=['blobs', 'noise', 'shapes']); ap.add_argument('--density', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--png')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    img = {'blobs': blobs, 'noise': noise, 'shapes': shapes}[a.kind](a.W, a.H, a.density, rng)
    write_image(a.out, img); print(f'wrote {a.out}: {a.W}x{a.H}, foreground={int(img.sum())} ({100*img.mean():.1f}%)')
    if a.png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.imsave(a.png, img, cmap='gray')
