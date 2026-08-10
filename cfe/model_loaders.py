from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import torch
import sys
import numpy as np
from torchvision import transforms
# from MagFace.inference.network_inf import builder_inf

def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)


def load_model_from_local_path(path, HF_TOKEN=None):
    # Patch any relative pretrained_model paths to absolute (no-op if not present)
    wrapper_path = os.path.join(path, 'wrapper.py')
    if os.path.exists(wrapper_path):
        abs_pretrained = os.path.join(path, 'pretrained_model')
        with open(wrapper_path, 'r') as f:
            wrapper_code = f.read()
        patched = wrapper_code.replace(
            "'pretrained_model/model.pt'",
            f"'{abs_pretrained}/model.pt'"
        ).replace(
            '"pretrained_model/model.pt"',
            f'"{abs_pretrained}/model.pt"'
        )
        with open(wrapper_path, 'w') as f:
            f.write(patched)

    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    
    # Remove any cached modules that might conflict
    modules_to_remove = [key for key in sys.modules if key.startswith('models')]
    for key in modules_to_remove:
        del sys.modules[key]
    
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    
    os.chdir(cwd)
    sys.path.pop(0)
    return model


def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def load_magface_model(ckpt_path, arch="iresnet100", embedding_size=512):
    class Args:
        pass
    args = Args()
    args.arch = arch
    args.embedding_size = embedding_size
    args.resume = ckpt_path
    args.cpu_mode = (DEVICE == "cpu")
    try:
        from MagFace.inference.network_inf import builder_inf
    except ImportError as e:
        raise ImportError(
            "MagFace requires the MagFace repo on PYTHONPATH "
            "(https://github.com/IrvingMeng/MagFace)."
        ) from e
    model = builder_inf(args)
    model = model.to(DEVICE).eval()
    return model
