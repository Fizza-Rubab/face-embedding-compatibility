# Compatibility of Face Embeddings Across Deep Neural Networks

Official code for the IJCB 2026 paper
**"Compatibility of Face Embeddings Across Deep Neural Networks"**
Fizza Rubab, Yiying Tong, Arun Ross (Michigan State University).

We ask whether independently trained face and foundation models encode facial
identity in geometrically compatible ways. Treating embeddings as point clouds,
we learn **simple linear maps** (Procrustes, Linear least-squares, Ridge) between
one model's embedding space and another's, and show that these low-capacity maps
recover cross-model identification and verification.

---

## What this repository contains

Code to reproduce the experiments and analyses in the paper:

1. **Extract** embeddings for each model on each dataset (`extraction/`).
2. **Align** embedding spaces with Procrustes / Linear / Ridge maps (`cfe/methods.py`).
3. **Evaluate** cross-model identification and verification, intra- and cross-dataset (`experiments/`).
4. **Analyze** the resulting compatibility structure — hierarchy, source/sink
   asymmetry, data-efficiency, RSA/CKA (`analysis/`).

---

## Repository structure

```
cfe/               # importable library (installed with `pip install -e .`)
  config.py        #   model registry, all-to-all pair matrix, dataset paths
  methods.py       #   alignment methods: Procrustes / Linear / Ridge (+ CCA, Neural)
  embedders.py     #   one extractor per model (lazy-loaded), + extract_and_save()
  model_loaders.py #   CVLface / MagFace checkpoint loaders
extraction/        # embed_{cfp,lfw,webface}.py  +  crop_lfw.py
experiments/       # identification_within_{cfp,lfw,webface}.py
                   # verification_within_{cfp,lfw,webface}.py
                   # identification_cross.py, verification_cross.py
analysis/          # hierarchy.py, data_efficiency.py,
                   # rsa_cka.py, quadratic_maps.py, ridge_alpha.py
```

At runtime the code also uses these git-ignored folders (create or symlink them):

```
data/          # raw datasets (see "Datasets")
checkpoints/   # model weights (see "Models")
embeddings/    # generated: embeddings/<dataset>/<model>.npy  (+ <dataset>_metadata.npy)
results/       # generated: per-experiment JSON/CSV outputs
.cache/        # Hugging Face cache (HF_HOME)
```

---

## Installation

```bash
conda create -n cfe python=3.11 -y
conda activate cfe
pip install -e .                 # installs the `cfe` package + core deps
pip install -r requirements.txt  # extra deps for embedding extraction (torch, transformers, …)
```

`pip install -e .` makes `import cfe` work from anywhere. **Run all scripts from the
repository root** so the repo-relative data paths (`embeddings/`, `results/`, …) resolve.

---

## Datasets

Each dataset is expected as `<root>/<identity>/<image>` (CFP additionally nests a
`frontal/` folder). Default locations (override with an env var):

| Dataset | Key | Default path | Env override |
|---|---|---|---|
| CFP (Celebrities in Frontal-Profile) | `cfp` | `data/cfp/Data/Images` | `CFE_CFP_DIR` |
| LFW (Labeled Faces in the Wild) | `lfw` | `data/lfw/lfw-deepfunneled_cropped_160all` | `CFE_LFW_DIR` |
| CASIA-WebFace | `webface` | `data/webface/casia-webface-5` | `CFE_WEBFACE_DIR` |

Paper protocol: CFP uses the 10 frontal images/identity (5,000 images); LFW is
center-cropped to 160×160 (`extraction/crop_lfw.py`); CASIA-WebFace is subsampled to
5 images/identity (`--max-images-per-id 5`, 52,875 images). Paths live in
`cfe/config.py` (`DATASET_IMG_DIRS`).

## Models

Fourteen models across two families (paper Table 1), all defined in `cfe/config.py`
(`ALL_MODELS`). The extractors live in `cfe/embedders.py` (one `embed_<model>` per
model, loaded lazily). Face-specific weights come from the
[CVLface](https://github.com/mk-minchul/CVLface) Hugging Face repos and
[MagFace](https://github.com/IrvingMeng/MagFace); foundation models load directly
from Hugging Face `transformers`.

- **Face-specific:** ArcFace (ir101), AdaFace (ir101), MagFace (ir100), KPRPE (ViT-base)
- **Foundation:** CLIP, ALIGN, DINOv2, SAM, BLIP-2, LLaVA, Kosmos-2, InternVL3, Florence-2, ViT

**Access requirements.** The CVLface face models are gated — set `export HF_TOKEN=...`.
MagFace needs its repo on `PYTHONPATH`
([IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace)) and its checkpoint at
`checkpoints/MagFace/magface_epoch_00025.pth`. Foundation models are public.

---

## Pipeline

All commands are run from the repository root.

### 1. Extract embeddings

```bash
export HF_TOKEN=...                                   # gated CVLface face models
python extraction/embed_cfp.py                        # all 14 models -> embeddings/cfp/<model>.npy
python extraction/embed_lfw.py                        # -> embeddings/lfw/<model>.npy
python extraction/embed_webface.py --max-images-per-id 5
python extraction/embed_cfp.py --models clip dinov2 arcface   # subset only
```

Each script also writes `embeddings/<dataset>/<dataset>_metadata.npy` (per-image
identity/path records, row-aligned with the embeddings).

### 2. Run alignment experiments

```bash
# Intra-dataset identification (paper Table 2, per dataset)
python experiments/identification_within_cfp.py
python experiments/identification_within_lfw.py
python experiments/identification_within_webface.py

# Intra-dataset verification
python experiments/verification_within_cfp.py
python experiments/verification_within_lfw.py
python experiments/verification_within_webface.py

# Cross-dataset generalization, CFP -> LFW (paper Table 3)
python experiments/identification_cross.py
python experiments/verification_cross.py
```

Results (per-pair metrics, aggregated over seeds) are written as JSON/CSV under
`results/<dataset>_analysis`, `results/<dataset>_verification`, and
`results/cross_dataset_*`.

### 3. Analysis

```bash
python analysis/hierarchy.py       # dendrogram + source/sink (incoming/outgoing) asymmetry
python analysis/data_efficiency.py # performance vs. training-data size (Fig 3 trend)
python analysis/rsa_cka.py         # RSA / CKA embedding similarity (supplementary)
python analysis/quadratic_maps.py  # non-linear (quadratic / MLP) baselines (supplementary)
python analysis/ridge_alpha.py     # Ridge alpha sensitivity
```

---

## The alignment API (`cfe/methods.py`)

```python
from cfe.methods import ProcrustesAlignment, LinearAlignment, RidgeAlignment

algo = LinearAlignment()          # or ProcrustesAlignment() / RidgeAlignment(alpha=0.1)
algo.fit(A_train, B_train)        # learn map from model-A space to model-B space
B_hat = algo.transform(A_test)    # align held-out A embeddings into B's space
```

Center embeddings on **training** statistics only (the scripts handle this) to avoid
test-identity leakage. `CCAAlignment` and `NeuralAlignment` are also provided as
baselines; `NeuralAlignment` is the only method that requires PyTorch.

---

## Reproducing paper artifacts (quick map)

| Paper artifact | Script(s) |
|---|---|
| Table 2 — intra-dataset | `experiments/identification_within_*.py`, `experiments/verification_within_*.py` |
| Table 3 — cross-dataset (CFP→LFW) | `experiments/identification_cross.py`, `experiments/verification_cross.py` |
| Performance vs. training data (trend) | `analysis/data_efficiency.py` |
| Dendrogram + source/sink asymmetry | `analysis/hierarchy.py` |
| RSA/CKA, non-linear maps, Ridge-α (suppl.) | `analysis/rsa_cka.py`, `analysis/quadratic_maps.py`, `analysis/ridge_alpha.py` |

Metrics for the tables are the JSON/CSV files each experiment writes under `results/`
(the paper's LaTeX table/figure formatting scripts are not included).

---

## Citation

```bibtex
@inproceedings{rubab2026compatibility,
  title     = {Compatibility of Face Embeddings Across Deep Neural Networks},
  author    = {Rubab, Fizza and Tong, Yiying and Ross, Arun},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
