"""
Crop LFW faces before embedding. The raw deep-funneled frames are 250x250 with the face
occupying roughly the central third; feeding them to a 112x112 FR encoder via a plain
resize hands the encoder mostly background.

LFW deep-funneled images are already alignment-normalised (that is what "funneled" means:
faces are congealed to a canonical pose/position), so a fixed central crop is the standard
preprocessing and is deterministic. We take the central CROP_FRAC of the frame and resize
to CROP_SIZE. Pass --detect to instead run insightface detection (slower, rarely needed).

Resumable; mirrors the identity/ folder layout.
Output: data/lfw_cropped/<Person_Name>/<image>.jpg
"""
import os, sys
CACHE = "./.cache"
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(CACHE, "kagglehub"))
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

CROP_SIZE = 112          # matches the FR encoders' input
CROP_FRAC = 0.60         # central 150px of the 250px funneled frame
OUT_ROOT = "./data/lfw_cropped"


def center_crop(args):
    ipath, opath = args
    if os.path.exists(opath):
        return 0
    img = Image.open(ipath).convert("RGB")
    w, h = img.size
    s = min(w, h) * CROP_FRAC
    box = ((w - s) / 2, (h - s) / 2, (w + s) / 2, (h + s) / 2)
    img.crop(tuple(int(round(v)) for v in box)) \
       .resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC).save(opath, quality=95)
    return 1


def main():
    import kagglehub
    base = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    in_root = os.path.join(base, "lfw-deepfunneled", "lfw-deepfunneled")
    os.makedirs(OUT_ROOT, exist_ok=True)

    jobs = []
    for person in sorted(os.listdir(in_root)):
        pdir = os.path.join(in_root, person)
        if not os.path.isdir(pdir):
            continue
        odir = os.path.join(OUT_ROOT, person)
        os.makedirs(odir, exist_ok=True)
        for f in sorted(os.listdir(pdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                jobs.append((os.path.join(pdir, f), os.path.join(odir, f)))
    print(f"{len(jobs)} images, {len(set(os.path.dirname(o) for _, o in jobs))} identities",
          flush=True)

    with ProcessPoolExecutor(max_workers=16) as ex:
        n = sum(ex.map(center_crop, jobs, chunksize=64))
    print(f"cropped {n} new ({len(jobs)-n} already present) -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
