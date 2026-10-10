"""Render image + coloured labels + centroids/bounding boxes.
Usage: python visualize.py image.bin labels.bin stats.csv out.png [--max-boxes 60]"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle
from ccl_io import read_image, read_labels, read_stats
img, lab, st = read_image(sys.argv[1]), read_labels(sys.argv[2]), read_stats(sys.argv[3]); maxb = 60
if '--max-boxes' in sys.argv: maxb = int(sys.argv[sys.argv.index('--max-boxes')+1])
K = int(lab.max()); rng = np.random.default_rng(0); pal = np.vstack([[0.06, 0.06, 0.1], rng.uniform(.3, 1, (K, 3))])
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
ax[0].imshow(img, cmap='gray'); ax[0].set_title(f'binary image {img.shape[1]}×{img.shape[0]}  ({100*img.mean():.0f}% foreground)')
ax[1].imshow(pal[lab]); ax[1].set_title(f'{K} connected components (8-connectivity)')
ax[2].imshow(pal[lab]); ax[2].set_title('centroid (+) and bounding box' + (f' — largest {maxb}' if K > maxb else ''))
for k, s in sorted(st.items(), key=lambda kv: -kv[1]['area'])[:maxb]:
    ax[2].add_patch(Rectangle((s['xmin']-.5, s['ymin']-.5), s['xmax']-s['xmin']+1, s['ymax']-s['ymin']+1, fill=False, ec='yellow', lw=.9))
    ax[2].plot(s['cx'], s['cy'], '+', color='cyan', ms=7, mew=1.3)
for a in ax: a.axis('off')
plt.tight_layout(); plt.savefig(sys.argv[4], dpi=150); print('wrote', sys.argv[4])
