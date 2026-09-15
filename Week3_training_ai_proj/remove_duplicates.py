# ============================================================
# DUPLICATE REMOVER for the chest X-ray dataset
#
# Problem it fixes:
#   The same X-ray can sit in the dataset twice under different names
#   (e.g. Tuberculosis-342.png in train/ AND rahman_Tuberculosis-342.png
#   in test/). Then the "test" score partly measures memory, not skill.
#   add_data.py can't catch these — it compares exact pixels, and a copy
#   re-saved as JPEG no longer matches exactly.
#
# What it does, per class:
#   1) Makes a 64x64 fingerprint of every image in train/ AND test/
#      (brightness/contrast normalised, so a re-saved copy still matches).
#   2) Groups images whose fingerprints are near-identical.
#   3) Keeps ONE per group and moves the rest to removed_duplicates/.
#      If a group has a copy in test/, the TEST copy is kept and train
#      copies go — so the model is never tested on a picture it trained on.
#
# Safety:
#   * Dry run by default — prints a report, moves nothing.
#   * Run  python remove_duplicates.py --apply  to actually move files.
#   * Files are MOVED (never deleted) and logged in
#     removed_duplicates/removed_log.txt, so it's fully reversible.
# ============================================================

import os
import sys
import shutil
import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEST_DIR = os.path.join(BASE_DIR, "chest_xray")
REMOVED_DIR = os.path.join(BASE_DIR, "removed_duplicates")

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
SPLITS = ["test", "train"]     # test first: when a group spans both, the test copy is kept
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# fingerprints this similar (correlation, 1.0 = identical) are the same picture
SAME_IMAGE = 0.995

APPLY = "--apply" in sys.argv


# ============================================================
# helpers
# ============================================================

def fingerprint(path):
    """64x64 grayscale, normalised to mean 0 / std 1, flattened. None if unreadable."""
    try:
        with Image.open(path) as im:
            im.draft("L", (128, 128))          # fast JPEG decode at reduced size
            a = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR),
                           dtype=np.float32).ravel()
    except Exception:
        return None
    a -= a.mean()
    return a / (a.std() + 1e-6)


# ============================================================
# find duplicate groups, per class
# ============================================================

print("=" * 60)
print("DRY RUN (nothing will be moved)" if not APPLY else "APPLY: moving duplicates ...")
print(f"\n  {'class':<15}{'images':>7}{'to move':>9}{'train=test':>12}   after")

log_lines = []
for cls in CLASSES:
    items, fps = [], []                        # items[i] = (split, filename)
    for split in SPLITS:
        folder = os.path.join(CHEST_DIR, split, cls)
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(VALID_EXT):
                fp = fingerprint(os.path.join(folder, fn))
                if fp is not None:
                    items.append((split, fn))
                    fps.append(fp)

    X = np.stack(fps)
    sim = X @ X.T / X.shape[1]                 # correlation of every pair

    # group near-identical images (union-find)
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(sim >= SAME_IMAGE, 1))):
        parent[root(j)] = root(i)

    groups = {}
    for i in range(len(items)):
        groups.setdefault(root(i), []).append(i)

    to_move = []                               # (index to move, index kept)
    for members in groups.values():
        keep = min(members)                    # test/ copies come first in `items`
        to_move += [(i, keep) for i in members if i != keep]

    n_test = sum(1 for s, _ in items if s == "test")
    moved_test = sum(1 for i, _ in to_move if items[i][0] == "test")
    leaks = sum(1 for i, k in to_move if items[i][0] != items[k][0])
    near_miss = int(np.count_nonzero(np.triu((sim >= 0.98) & (sim < SAME_IMAGE), 1)))
    print(f"  {cls:<15}{len(items):>7}{len(to_move):>9}{leaks:>12}   "
          f"train {len(items) - n_test - (len(to_move) - moved_test)} / test {n_test - moved_test}"
          f"   (similar but kept: {near_miss} pairs)")
    for i, k in to_move[:2]:
        print(f"      e.g. {items[i][0]}/{items[i][1]}  same as  {items[k][0]}/{items[k][1]}")

    for i, k in to_move:
        split, fn = items[i]
        log_lines.append(f"{split}/{cls}/{fn}\tsame as\t{items[k][0]}/{cls}/{items[k][1]}"
                         f"\t{sim[i, k]:.4f}")
        if APPLY:
            dest = os.path.join(REMOVED_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            shutil.move(os.path.join(CHEST_DIR, split, cls, fn), os.path.join(dest, fn))


# ============================================================
# summary
# ============================================================

print("=" * 60)
print(f"'train=test' = a train image that was also in test (leakage).")
if not log_lines:
    print("No duplicates found. Nothing to do.")
elif not APPLY:
    print(f"DRY RUN complete: {len(log_lines)} duplicate files would be moved.")
    print("To move them:  python remove_duplicates.py --apply")
else:
    with open(os.path.join(REMOVED_DIR, "removed_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"✅ Moved {len(log_lines)} duplicates to removed_duplicates/ (see removed_log.txt).")
    print("   Now retrain:  python train_xray.py")
