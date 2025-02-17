import argparse
import os
import hashlib
import functools
import json
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm

from imgproc import (
    generate_crop_size_list,
    to_rgb_if_rgba,
    var_center_crop,
)
from data import read_general

# ---- Flux VAE scaling parameters ----
VAE_SCALE = 0.3611
VAE_SHIFT = 0.1159

def handle_image(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB format, converting from RGBA, L, P, etc.
    Raise ValueError if unrecognized mode.
    """
    mode = image.mode.upper()
    if mode == "RGB":
        return image
    elif mode == "RGBA":
        return to_rgb_if_rgba(image)
    elif mode in ("L", "P"):
        return image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

def encode(vae: AutoencoderKL, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encode a normalized image tensor to latents using the Flux VAE, applying SHIFT+SCALE.
    img_tensor shape: (C, H, W) or (1,C,H,W). We'll reshape to (1,C,H,W) if needed.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # (1,C,H,W)
    img_tensor = img_tensor.to(device, non_blocking=True)
    with torch.no_grad():
        # bfloat16 casting for VAE encode
        latent_dist = vae.encode(img_tensor).latent_dist
        # use .mode()[0] or .sample() depending on whether you prefer the mode or random sample
        latents = latent_dist.mode()[0]
        latents = (latents - VAE_SHIFT) * VAE_SCALE
    return latents.float()

def load_image_paths_from_yaml(yaml_path: str) -> list:
    """
    Parse a YAML containing a 'META' key with paths to .jsonl files.
    For each .jsonl (with 'type' == 'image_text'), read lines of JSON
    where we expect an 'image_path' field. Collect these paths in a list.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    image_files = []
    meta_list = data.get("META", [])
    for meta_item in meta_list:
        # Example:  path=/data0/DanbooruWebp/booru1116Webp.jsonl
        #           type=image_text
        ftype = meta_item.get("type", "")
        fpath = meta_item.get("path", "")
        if ftype != "image_text":
            # skip unknown types
            continue
        if not os.path.isfile(fpath):
            print(f"[Warning] JSONL file not found: {fpath}")
            continue

        # Open .jsonl and parse lines
        with open(fpath, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "image_path" in obj:
                        # This is the actual disk path for the image
                        image_files.append(obj["image_path"])
                except Exception as e:
                    print(f"[Warning] JSON parse error in {fpath}: {e}")
                    continue

    return image_files

def main():
    parser = argparse.ArgumentParser(description="Cache image latents using Flux VAE")
    parser.add_argument("--data_yaml", type=str, required=True,
                        help="Path to dataset YAML config (with META -> .jsonl paths)")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Target resolution (e.g., 256, 512, 1024) for center-crop/resize")
    parser.add_argument("--total_split", type=int, default=1,
                        help="Total number of parallel splits/workers")
    parser.add_argument("--current_worker_index", type=int, default=0,
                        help="Index of this worker (0-based)")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size used for generating potential crop sizes")
    parser.add_argument("--random_top_k", type=int, default=1,
                        help="Number of top crop options from var_center_crop to randomly pick")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1) Setup VAE model for encoding:
    # ------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        torch_dtype=torch.float16
    ).eval()

    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )
    vae.to(device)

    # ------------------------------------------------------------------
    # 2) Prepare your transform (crop -> tensor -> normalize).
    #    This must match how images are processed before training.
    # ------------------------------------------------------------------
    max_num_patches = round((args.resolution / (args.patch_size * 1.0)) ** 2)
    crop_size_list = generate_crop_size_list(max_num_patches, args.patch_size)
    image_transform = transforms.Compose([
        transforms.Lambda(functools.partial(var_center_crop,
                                            crop_size_list=crop_size_list,
                                            random_top_k=args.random_top_k)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ------------------------------------------------------------------
    # 3) Load image paths from YAML / JSONL references:
    # ------------------------------------------------------------------
    image_files = load_image_paths_from_yaml(args.data_yaml)
    if not image_files:
        print("[INFO] No image files found. Check your YAML & JSONL contents.")
        return

    # ------------------------------------------------------------------
    # 4) Process each image => transform => encode => save .npz
    # ------------------------------------------------------------------
    worker_idx = args.current_worker_index
    total_split = args.total_split
    res = args.resolution

    for image_path in tqdm(image_files, desc=f"Worker {worker_idx}"):
        # 4.a) Determine if this file belongs to the current worker
        hash_val = int(hashlib.sha1(image_path.encode("utf-8")).hexdigest(), 16)
        if hash_val % total_split != worker_idx:
            continue

        # 4.b) Construct cache path
        base, _ = os.path.splitext(image_path)
        out_path = f"{base}_{res}.npz"
        if os.path.exists(out_path):
            continue

        # 4.c) Read the image from disk & handle mode
        try:
            pil_image = Image.open(read_general(image_path))
            pil_image = handle_image(pil_image)  # ensure RGB
        except Exception as e:
            print(f"[Warning] Could not open image {image_path}: {e}")
            continue

        # Optionally, you can do a simple resize (if your training expects it).
        # Otherwise, rely solely on var_center_crop to pick a final crop size.
        pil_image = pil_image.resize((res, res), Image.Resampling.LANCZOS)

        # 4.d) Apply var_center_crop -> toTensor -> normalize
        try:
            transformed_tensor = image_transform(pil_image)  # shape=(3,H,W)
        except Exception as e:
            print(f"[Warning] Skipping {image_path} due to transform error: {e}")
            continue
        transformed_tensor = transformed_tensor.to(torch.float16)
        # 4.e) Encode with Flux VAE (shift+scale) => latent
        latents = encode(vae, transformed_tensor, device=device)
        latents_np = latents.cpu().numpy()  # shape=(C, H//8, W//8) typically

        # 4.f) Save latents to .npz
        try:
            np.savez_compressed(out_path, latent=latents_np)
        except Exception as e:
            print(f"[Error] Saving .npz for {image_path} failed: {e}")

if __name__ == "__main__":
    main()
