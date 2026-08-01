"""Extract CFP embeddings -> embeddings/cfp/<model>.npy (+ cfp_metadata.npy).

Set HF_TOKEN for the face models. Run: python embed_cfp.py [--models clip arcface ...]
"""
import os
import argparse
import numpy as np

from cfe.config import ALL_MODELS, DATASET_IMG_DIRS, EMB_DIR
from cfe.embedders import extract_and_save

DATASET = "cfp"
IMG_DIR = os.environ.get("CFE_CFP_DIR", DATASET_IMG_DIRS[DATASET])
SAVE_DIR = os.path.join(EMB_DIR, DATASET)


def index_dataset(root_dir, max_identities=None, max_images_per_id=None):
    """CFP layout: <root>/<identity>/frontal/<image>. We use the frontal images."""
    records = []
    identities = sorted(d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d)))
    if max_identities is not None:
        identities = identities[:max_identities]
    for person in identities:
        frontal_dir = os.path.join(root_dir, person, "frontal")
        if not os.path.isdir(frontal_dir):
            continue
        images = sorted(f for f in os.listdir(frontal_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if max_images_per_id is not None:
            images = images[:max_images_per_id]
        for img in images:
            records.append({"identity": person, "image_name": img,
                            "rel_path": os.path.join(person, "frontal", img)})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-identities", type=int, default=None)
    ap.add_argument("--max-images-per-id", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    records = index_dataset(IMG_DIR, args.max_identities, args.max_images_per_id)
    print(f"Indexed {len(records)} images from {IMG_DIR}")
    np.save(os.path.join(SAVE_DIR, f"{DATASET}_metadata.npy"), records, allow_pickle=True)

    extract_and_save(records, IMG_DIR, SAVE_DIR, models=args.models,
                     batch_size=args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
