# Compatibility of Face Embeddings Across Deep Neural Networks

Official code for the IJCB 2026 paper
**"Compatibility of Face Embeddings Across Deep Neural Networks"**
Fizza Rubab, Yiying Tong, Arun Ross (Michigan State University).

We learn simple linear maps (Procrustes, least-squares, Ridge) between the embedding
spaces of independently trained face and foundation models, and show that these
low-capacity maps recover cross-model identification and verification.

## Repository structure

```
cfe/               importable library (pip install -e .): config, alignment methods,
                   embedders, model loaders
extraction/        embed_{cfp,lfw,webface}.py, crop_lfw.py
experiments/       identification / verification, within- and cross-dataset
analysis/          hierarchy, data efficiency, rsa/cka, quadratic maps, ridge alpha
```

## Installation

```bash
conda create -n cfe python=3.11 -y
conda activate cfe
pip install -e .
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`pip install -e .` alone is enough to run the alignment and analysis on existing embeddings.

## Datasets

| Dataset | Key | Default path | Env override |
|---|---|---|---|
| CFP | `cfp` | `data/cfp/Data/Images` | `CFE_CFP_DIR` |
| LFW | `lfw` | `data/lfw_cropped` | `CFE_LFW_DIR` |
| CASIA-WebFace | `webface` | `data/webface/casia-webface-5` | `CFE_WEBFACE_DIR` |

CFP uses the 10 frontal images per identity; LFW is center-cropped by `extraction/crop_lfw.py`;
CASIA-WebFace is subsampled with `--max-images-per-id 5`. Paths live in `cfe/config.py`.

## Models

Fourteen models across two families (paper Table 1), defined in `cfe/config.py`. Face weights
come from CVLface (gated: `export HF_TOKEN=...`) and MagFace; foundation models load from
Hugging Face `transformers`.

Face: ArcFace, AdaFace, MagFace, KPRPE.
Foundation: CLIP, ALIGN, DINOv2, SAM, BLIP-2, LLaVA, Kosmos-2, InternVL3, Florence-2, ViT.

## Pipeline

Run from the repository root.

```bash
export HF_TOKEN=...
python extraction/embed_cfp.py
python extraction/embed_lfw.py
python extraction/embed_webface.py --max-images-per-id 5

python experiments/identification_within_cfp.py
python experiments/verification_within_cfp.py
python experiments/identification_cross.py
python experiments/verification_cross.py

python analysis/hierarchy.py
python analysis/data_efficiency.py
python analysis/rsa_cka.py
python analysis/quadratic_maps.py
python analysis/ridge_alpha.py
```

## Alignment API

```python
from cfe.methods import ProcrustesAlignment, LinearAlignment, RidgeAlignment

algo = LinearAlignment()
algo.fit(A_train, B_train)
B_hat = algo.transform(A_test)
```

## Scripts and analysis

| Paper artifact | Script(s) |
|---|---|
| Table 2: intra-dataset | `experiments/identification_within_*.py`, `experiments/verification_within_*.py` |
| Table 3: cross-dataset (CFP→LFW) | `experiments/identification_cross.py`, `experiments/verification_cross.py` |
| Performance vs. training data | `analysis/data_efficiency.py` |
| Dendrogram + source/sink asymmetry | `analysis/hierarchy.py` |
| RSA/CKA, non-linear maps, Ridge-α | `analysis/rsa_cka.py`, `analysis/quadratic_maps.py`, `analysis/ridge_alpha.py` |

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
