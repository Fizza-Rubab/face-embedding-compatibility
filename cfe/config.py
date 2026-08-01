"""Model registry, pair matrix, and dataset/output paths."""
import os

FACE_MODELS = ["arcface", "adaface", "magface", "kprpe"]
FOUNDATION_MODELS = [
    "clip", "align", "dinov2", "sam", "blip2",
    "llava", "kosmos", "internvl", "florence", "googlevit",
]
ALL_MODELS = FACE_MODELS + FOUNDATION_MODELS

DISPLAY_NAME = {
    "arcface": "ArcFace", "adaface": "AdaFace", "magface": "MagFace", "kprpe": "KPRPE",
    "clip": "CLIP", "align": "ALIGN", "dinov2": "DINOv2", "sam": "SAM", "blip2": "BLIP-2",
    "llava": "LLaVA", "kosmos": "Kosmos-2", "internvl": "InternVL3", "florence": "Florence-2",
    "googlevit": "ViT",
}


def all_pairs(models, include_self=False):
    return [(a, b) for a in models for b in models if include_self or a != b]


# All-to-all within each family. Main text shows representative pairs; the rest
# go in the supplement.
DEFAULT_PAIRS = all_pairs(FACE_MODELS) + all_pairs(FOUNDATION_MODELS)
MAIN_TEXT_PAIRS = [
    ("arcface", "adaface"), ("adaface", "kprpe"), ("kprpe", "magface"),
    ("clip", "align"), ("blip2", "llava"), ("dinov2", "googlevit"),
    ("sam", "internvl"), ("florence", "kosmos"),
]

# Repo-relative paths; override via the matching env var.
EMB_DIR = os.environ.get("CFE_EMB_DIR", "embeddings")
RESULTS_DIR = os.environ.get("CFE_RESULTS_DIR", "results")
CACHE_DIR = os.environ.get("CFE_CACHE_DIR", ".cache")
CKPT_DIR = os.environ.get("CFE_CKPT_DIR", "checkpoints")

DATASET_IMG_DIRS = {
    "cfp": "data/cfp/Data/Images",
    "lfw": "data/lfw/lfw-deepfunneled_cropped_160all",
    "webface": "data/webface/casia-webface-5",
}


def emb_path(dataset, model):
    return os.path.join(EMB_DIR, dataset, f"{model}.npy")


def meta_path(dataset):
    return os.path.join(EMB_DIR, dataset, f"{dataset}_metadata.npy")
