# ============================================================
# SAFE DATA ADDER for the chest X-ray dataset
#
# In one run it:
#   1) Hashes every image in chest_xray/train + chest_xray/test.
#   2) Hashes every image in the new_data/ staging folder.
#   3) Drops new images that duplicate existing ones (leakage) or
#      duplicate each other.
#   4) Splits the survivors 80/20 into train/test, per class.
#
# Safety:
#   * Dry run by default (APPLY = False) — prints a report, copies nothing.
#   * Set APPLY = True to actually copy.
#   * Copies only, never moves or deletes.
#   * Also warns about pre-existing train/test duplicates.
#
# Note: hashing catches identical / re-saved-identical images, not the
# same photo resized or re-compressed. Strong safeguard, not perfect.
# ============================================================

import os
import shutil
import hashlib
from collections import defaultdict
from PIL import Image
import io

# ============================================================
# CONFIG — edit these
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# real dataset (already has train/ and test/ with the 4 class folders)
CHEST_DIR = os.path.join(BASE_DIR, "chest_xray")

# staging folder with NEW images. Inside it, put one subfolder per class,
# using the same class names as chest_xray:
#
#   new_data/
#     COVID19/
#     NORMAL/
#     PNEUMONIA/
#     TURBERCULOSIS/
#
# Only the classes you're adding need to exist; missing ones are skipped.
NEW_DIR = os.path.join(BASE_DIR, "new_data")

# must match the real folder names exactly (note the TURBERCULOSIS spelling)
CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]

TRAIN_RATIO = 0.8          # 80% train / 20% test
SEED = 42                  # deterministic split
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# False = dry run (report only). True = actually copy.
APPLY = False


# ============================================================
# helpers
# ============================================================

def is_image(fn):
    return fn.lower().endswith(VALID_EXT)

def image_hash(path):
    """Hash the DECODED pixels (not the file bytes), so the same image
    saved under a different name or container still hashes the same.
    Returns None if the file can't be read as an image."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((64, 64))   # normalize for a robust-ish hash
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return hashlib.md5(buf.getvalue()).hexdigest()
    except Exception:
        return None

def list_images_recursive(folder):
    out = []
    if not os.path.isdir(folder):
        return out
    for root, _, files in os.walk(folder):
        for fn in files:
            if is_image(fn):
                out.append(os.path.join(root, fn))
    return out


# ============================================================
# STEP 1: fingerprint existing chest_xray/ (train + test)
# ============================================================

print("=" * 60)
print("STEP 1: Fingerprinting your existing chest_xray/ ...")

existing_hashes = {}          # hash -> "split/CLASS" where it lives
existing_by_split = defaultdict(set)  # (split, cls) -> set of hashes
pre_existing_leak = []        # images already in BOTH train and test

for split in ["train", "test"]:
    for cls in CLASSES:
        folder = os.path.join(CHEST_DIR, split, cls)
        for p in list_images_recursive(folder):
            h = image_hash(p)
            if h is None:
                continue
            # detect a pre-existing train/test leak for this class
            other = "test" if split == "train" else "train"
            if h in existing_by_split[(other, cls)]:
                pre_existing_leak.append((cls, h))
            existing_by_split[(split, cls)].add(h)
            existing_hashes[h] = f"{split}/{cls}"

print(f"   existing images fingerprinted: {len(existing_hashes)}")
if pre_existing_leak:
    print(f"   ⚠️  WARNING: {len(pre_existing_leak)} image(s) already appear in BOTH")
    print(f"       train AND test in your current data (pre-existing leak).")
    by_cls = defaultdict(int)
    for cls, _ in pre_existing_leak:
        by_cls[cls] += 1
    for cls, n in by_cls.items():
        print(f"          {cls}: {n}")
else:
    print("   ✓ no train/test duplicate leak found in existing data")


# ============================================================
# STEP 2: fingerprint the NEW staging folder, drop duplicates
# ============================================================

print("=" * 60)
print("STEP 2: Checking new_data/ for duplicates ...")

if not os.path.isdir(NEW_DIR):
    raise SystemExit(
        f"\nStaging folder not found: {NEW_DIR}\n"
        f"Create a 'new_data' folder next to chest_xray/, with per-class subfolders\n"
        f"(COVID19 / NORMAL / PNEUMONIA / TURBERCULOSIS) holding the new images."
    )

# per class: list of (path, hash) that are genuinely new & unique
to_add = defaultdict(list)
seen_new_hashes = set()

report = {}
for cls in CLASSES:
    folder = os.path.join(NEW_DIR, cls)
    paths = list_images_recursive(folder)
    n_total = len(paths)
    n_dup_existing = 0
    n_dup_within = 0
    n_unreadable = 0

    for p in paths:
        h = image_hash(p)
        if h is None:
            n_unreadable += 1
            continue
        if h in existing_hashes:          # already in chest_xray -> would leak
            n_dup_existing += 1
            continue
        if h in seen_new_hashes:          # duplicate inside new_data itself
            n_dup_within += 1
            continue
        seen_new_hashes.add(h)
        to_add[cls].append((p, h))

    report[cls] = dict(total=n_total, dup_existing=n_dup_existing,
                       dup_within=n_dup_within, unreadable=n_unreadable,
                       kept=len(to_add[cls]))

print("\n  Per-class report (new_data):")
print(f"  {'class':<15}{'found':>7}{'dupVSyours':>12}{'dupInternal':>12}{'unreadable':>11}{'KEPT':>7}")
for cls in CLASSES:
    r = report[cls]
    print(f"  {cls:<15}{r['total']:>7}{r['dup_existing']:>12}{r['dup_within']:>12}"
          f"{r['unreadable']:>11}{r['kept']:>7}")

total_kept = sum(len(v) for v in to_add.values())
print(f"\n  => {total_kept} genuinely-new, unique images ready to add.")
if total_kept == 0:
    print("  Nothing new to add (everything was a duplicate or unreadable). Stopping.")
    raise SystemExit(0)


# ============================================================
# STEP 3: plan the 80/20 split (per class), show it
# ============================================================

print("=" * 60)
print(f"STEP 3: Planning an {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)} split ...")

import random
random.seed(SEED)

plan = {}   # cls -> (train_list, test_list)
for cls in CLASSES:
    items = to_add[cls][:]
    random.shuffle(items)
    cut = int(round(len(items) * TRAIN_RATIO))
    plan[cls] = (items[:cut], items[cut:])
    if items:
        print(f"   {cls}: +{len(items[:cut])} to train, +{len(items[cut:])} to test")


# ============================================================
# STEP 4: APPLY (copy) — only if APPLY = True
# ============================================================

print("=" * 60)
if not APPLY:
    print("DRY RUN complete. Nothing was copied.")
    print("If the report looks right (low 'dupVSyours', sensible KEPT counts),")
    print("set  APPLY = True  at the top of this file and run it again to copy.")
    raise SystemExit(0)

print("APPLY = True  ->  copying files into chest_xray/ ...")

def copy_into(split, cls, items):
    dest = os.path.join(CHEST_DIR, split, cls)
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for src, h in items:
        # unique destination name so nothing gets overwritten
        base = f"added_{h[:10]}_{os.path.basename(src)}"
        base = base.replace(" ", "_")
        dst = os.path.join(dest, base)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"     could not copy {src}: {e}")
    return copied

grand = 0
for cls in CLASSES:
    tr, te = plan[cls]
    c1 = copy_into("train", cls, tr)
    c2 = copy_into("test", cls, te)
    grand += c1 + c2
    if tr or te:
        print(f"   {cls}: copied {c1} -> train, {c2} -> test")

print(f"\n✅ Done. Copied {grand} new images into chest_xray/.")
print("   Now retrain:  python train_xray.py")
print("   (Educational demo only — not a diagnostic tool.)")
