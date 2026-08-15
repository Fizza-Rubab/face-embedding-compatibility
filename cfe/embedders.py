"""Per-model embedding extractors. EMBEDDERS[name](list_of_PIL) -> (B, D) array.

Models are loaded lazily on first use. The CVLface face models need HF_TOKEN;
MagFace needs its repo on PYTHONPATH; foundation models are public.
"""
import os
# Point the HF cache at the repo (scratch) BEFORE importing transformers, which
# freezes its cache path on import. Otherwise it defaults to ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", os.environ.get("CFE_CACHE_DIR", ".cache"))

import numpy as np
import torch
from torchvision import transforms as T

from cfe.model_loaders import load_model_by_repo_id, load_magface_model

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.environ.get("CFE_CKPT_DIR", "checkpoints")

# Shared 112x112, [-1, 1] preprocessing for the face models.
_FACE_PROC = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_CACHE = {}


def _once(key, builder):
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def _image_features(out):
    # transformers <5 returns a tensor from get_image_features; 5.x returns a
    # ModelOutput whose projected features are in pooler_output / image_embeds.
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"unexpected get_image_features output: {type(out)}")


# Foundation models
def embed_clip(imgs):
    from transformers import CLIPProcessor, CLIPModel
    proc, model = _once("clip", lambda: (
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = _image_features(model.get_image_features(**inputs))
    return emb.float().cpu().numpy()


def embed_align(imgs):
    from transformers import AlignProcessor, AlignModel
    proc, model = _once("align", lambda: (
        AlignProcessor.from_pretrained("kakaobrain/align-base"),
        AlignModel.from_pretrained("kakaobrain/align-base").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = _image_features(model.get_image_features(**inputs))
    return emb.float().cpu().numpy()


def embed_dinov2(imgs):
    from transformers import AutoImageProcessor, AutoModel
    proc, model = _once("dinov2", lambda: (
        AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
        AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).pooler_output
    return emb.float().cpu().numpy()


def embed_sam(imgs):
    from transformers import SamProcessor, SamModel
    proc, model = _once("sam", lambda: (
        SamProcessor.from_pretrained("facebook/sam-vit-base"),
        SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        fmap = model.get_image_embeddings(inputs["pixel_values"])
    emb = fmap.mean(dim=[2, 3])          # global-average-pool the image embedding
    return emb.float().cpu().numpy()


def embed_blip2(imgs):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    proc, model = _once("blip2", lambda: (
        Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b"),
        Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        emb = model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0]
    return emb.float().cpu().numpy()


def embed_llava(imgs):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc, model = _once("llava", lambda: (
        AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf"),
        LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto").eval()))
    vision = model.model.vision_tower
    pixel_values = proc.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(vision.device, torch.float16)
    with torch.no_grad():
        emb = vision(pixel_values).last_hidden_state[:, 0]
    return emb.float().cpu().numpy()


def embed_kosmos(imgs):
    from transformers import AutoProcessor, Kosmos2Model
    proc, model = _once("kosmos", lambda: (
        AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224"),
        Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state.mean(dim=1)
    return emb.float().cpu().numpy()


def embed_internvl(imgs):
    from transformers import AutoModel
    tfm, model = _once("internvl", lambda: (
        T.Compose([T.Resize((448, 448)), T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        AutoModel.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True,
                                  torch_dtype=torch.float16).vision_model.to(DEVICE).eval()))
    xs = torch.stack([tfm(im) for im in imgs]).to(model.device, model.dtype)
    with torch.no_grad():
        emb = model(xs).last_hidden_state[:, 0]
    return emb.float().cpu().numpy()


def embed_florence(imgs):
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    proc, model = _once("florence", lambda: (
        AutoProcessor.from_pretrained("florence-community/Florence-2-base"),
        Florence2ForConditionalGeneration.from_pretrained(
            "florence-community/Florence-2-base", torch_dtype=torch.float16).to(DEVICE).eval()))
    inputs = proc(text=["<OD>"] * len(imgs), images=list(imgs),
                  return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        emb = out.encoder_last_hidden_state.mean(dim=1)
    return emb.float().cpu().numpy()


def embed_googlevit(imgs):
    from transformers import AutoFeatureExtractor, AutoModel
    proc, model = _once("googlevit", lambda: (
        AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224"),
        AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()))
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :]     # CLS token
    return emb.float().cpu().numpy()


# Face-recognition models (CVLface / MagFace)
def _load_cvlface(repo_id):
    save_path = os.path.join(CKPT_DIR, "models--" + repo_id.replace("/", "--"))
    return load_model_by_repo_id(repo_id, save_path=save_path, HF_TOKEN=HF_TOKEN)


def embed_arcface(imgs):
    model = _once("arcface", lambda: _load_cvlface("minchul/cvlface_arcface_ir101_webface4m"))
    batch = torch.stack([_FACE_PROC(im) for im in imgs]).to(model.device)
    with torch.no_grad():
        emb = model(batch)
    return emb.float().cpu().numpy()


def embed_adaface(imgs):
    model = _once("adaface", lambda: _load_cvlface("minchul/cvlface_adaface_ir101_ms1mv2"))
    batch = torch.stack([_FACE_PROC(im) for im in imgs]).to(model.device)
    with torch.no_grad():
        emb = model(batch)
    return emb.float().cpu().numpy()


def embed_kprpe(imgs):
    model, aligner = _once("kprpe", lambda: (
        _load_cvlface("minchul/cvlface_adaface_vit_base_kprpe_webface4m"),
        _load_cvlface("minchul/cvlface_DFA_mobilenet")))
    batch = torch.stack([_FACE_PROC(im) for im in imgs]).to(model.device)
    with torch.no_grad():
        _, keypoints, _, _, _, _ = aligner(batch)     # (B, 5, 2) landmarks
        emb = model(batch, keypoints)
    return emb.float().cpu().numpy()


def embed_magface(imgs):
    ckpt = os.path.join(CKPT_DIR, "MagFace", "magface_epoch_00025.pth")
    model = _once("magface", lambda: load_magface_model(ckpt))
    batch = torch.stack([_FACE_PROC(im) for im in imgs]).to(DEVICE)
    with torch.no_grad():
        emb = model(batch)
    return emb.float().cpu().numpy()


EMBEDDERS = {
    "clip": embed_clip, "align": embed_align, "dinov2": embed_dinov2, "sam": embed_sam,
    "blip2": embed_blip2, "llava": embed_llava, "kosmos": embed_kosmos,
    "internvl": embed_internvl, "florence": embed_florence, "googlevit": embed_googlevit,
    "arcface": embed_arcface, "adaface": embed_adaface, "kprpe": embed_kprpe, "magface": embed_magface,
}


def l2norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def extract_and_save(records, img_dir, save_dir, models=None, batch_size=64):
    """Embed all `records` with each model and save one L2-normalized .npy per model."""
    from PIL import Image
    from tqdm import tqdm

    models = list(models) if models else list(EMBEDDERS)
    os.makedirs(save_dir, exist_ok=True)
    for name in models:
        embed_fn = EMBEDDERS[name]
        outs = []
        for i in tqdm(range(0, len(records), batch_size), desc=name):
            batch = records[i:i + batch_size]
            imgs = [Image.open(os.path.join(img_dir, r["rel_path"])).convert("RGB")
                    for r in batch]
            outs.append(l2norm_rows(embed_fn(imgs)))
        arr = np.vstack(outs)
        np.save(os.path.join(save_dir, f"{name}.npy"), arr)
        print(f"  saved {name}: {arr.shape}")
