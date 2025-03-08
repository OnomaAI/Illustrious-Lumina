from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings
import pandas as pd
from tqdm import tqdm
import h5py
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml
from datasets import load_dataset
logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class DataNoReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError
def is_huggingface_path(path: str) -> bool:
    # Heuristic: Hugging Face dataset paths are in format "user/dataset"
    # and not an existing local file or directory.
    return "/" in path and not os.path.exists(path) and not "booru" in path

class ImageTextDataset(Dataset):
    """General dataset class that handles both local and Hugging Face image-text data."""
    def __init__(self, path: str, subset: str = None, cache_latents: bool = False, transform=None):
        self.transform = transform
        self.cache_latents = cache_latents
        self.latents = None

        if is_huggingface_path(path):
            # Load from Hugging Face Hub
            dataset_name = path
            # If subset is specified in YAML, use it; otherwise, default to None or a default config
            ds = load_dataset(dataset_name, subset, split="train", streaming=False)
            # Rename fields for consistency
            if "annotation" in ds.column_names:
                ds = ds.rename_column("annotation", "prompt")
            # (Optionally handle other field name mappings if needed)
            self.data = ds    # a datasets.Dataset object
        else:
            # Load from local JSON or folder (existing mechanism)
            # e.g., load json lines with keys "image_path" and "prompt"
            self.data = load_local_imagetext_data(path)  # pseudo-function to load local data

        # If caching latents is requested, pre-compute them
        if cache_latents:
            self._compute_and_cache_latents()
    
    def __len__(self):
        # HuggingFace Dataset acts like a sequence (len available if not streaming)
        return len(self.data)
    
    def process_text_caption(self, caption):
        if "text:" in caption:
            pre_caption, post_caption = caption.split("text:")
            caption = pre_caption + "text: \"" + post_caption + "\""
        return caption
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # If using HuggingFace dataset, sample is a dict; if local, design it to be dict for consistency.
        image = None
        caption = None
        if isinstance(sample, dict):
            # For HuggingFace or JSON data
            if "image" in sample:
                # HuggingFace dataset with decoded image
                image = sample["image"]
            elif "image_path" in sample:
                # Local data case: use read_general to open the image
                image_path = sample["image_path"]
                image = Image.open(read_general(image_path))
            # Get caption text (support different possible key names)
            caption = sample.get("prompt") or sample.get("caption") or sample.get("text")
            caption = self.process_text_caption(caption)
        else:
            # If self.data was a list of tuples or custom format, handle accordingly.
            image_path, caption = sample  # example for tuple format
            caption = self.process_text_caption(caption)
            image = Image.open(read_general(image_path))
        
        if self.transform:
            image = self.transform(image)
        # If latents are cached, return latent instead of image
        if self.latents is not None:
            latent = self.latents[idx]
            return {"image_latent": latent, "prompt": caption}
        else:
            return {"image": image, "prompt": caption}

    def _compute_and_cache_latents(self):
        """Pre-compute latents for all images in the dataset using the VAE, to accelerate training."""
        vae = load_pretrained_vae()  # pseudo-function to load the VAE model
        self.latents = []
        for i in range(len(self.data)):
            # Obtain image (either from memory or by loading from path)
            img = None
            if isinstance(self.data[i], dict) and "image" in self.data[i]:
                img = self.data[i]["image"]
            else:
                path = self.data[i]["image_path"] if isinstance(self.data[i], dict) else self.data[i][0]
                img = Image.open(read_general(path))
            latent = vae.encode(img)  # pseudo-code: get latent from VAE
            self.latents.append(latent)

class MyDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessor, cache_on_disk=False):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.cache_on_disk = cache_on_disk
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            ann, group_indice_range = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            ann, group_indice_range = self._collect_annotations()

        self.ann = ann
        self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}

        logger.info(f"total length: {len(self)}")

        self.item_processor = item_processor

    def __len__(self):
        return len(self.ann)

    def _collect_annotations(self):
        group_ann = {}
        for meta in self.config["META"]:
            meta_path, meta_type = meta["path"], meta.get("type", "default")
            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                # with open(meta_path) as f:
                #     meta_l = json.load(f)
                with open(meta_path, 'r') as json_file:
                    f = json_file.read()
                    meta_l = json.loads(f) 
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                            raise e
            elif meta_ext == ".parquet":
                meta_l = []
                df = pd.read_parquet(meta_path)  # Read the Parquet file into a DataFrame
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {meta_path}"):
                    # Pull the 'index' column (whatever column indicates image index/id)
                    index_val = row["index"]

                    # For each *other* column in the row, if not None/NaN, use it as "prompt"
                    for col in df.columns:
                        if col == "index":
                            continue
                        # Skip if the value is None or NaN
                        if pd.notna(row[col]) and str(row[col]):
                            meta_l.append({
                                "image_path": f"danbooru://{index_val}" if not os.path.exists(index_val) else index_val,
                                "prompt": str(row[col])  # Cast to str in case it's not a string
                            })
            else:
                raise NotImplementedError(
                    f'Unknown meta file extension: "{meta_ext}". '
                    f"Currently, .json, .jsonl, .parquet (with index column + caption columns) are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            logger.info(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            if "ratio" in meta:
                random.seed(0)
                meta_l = random.sample(meta_l, int(len(meta_l) * meta["ratio"]))
                logger.info(f"sample (ratio = {meta['ratio']}) {len(meta_l)} items")
            if "root" in meta:
                for item in meta_l:
                    for path_key in ["path", "image_url", "image", "image_path"]:
                        if path_key in item:
                            item[path_key] = os.path.join(meta["root"], item[path_key])
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            group_ann[meta_type] += meta_l

        ann = sum(list(group_ann.values()), start=[])

        group_indice_range = {}
        start_pos = 0
        for meta_type, meta_l in group_ann.items():
            group_indice_range[meta_type] = [start_pos, start_pos + len(meta_l)]
            start_pos = start_pos + len(meta_l)

        return ann, group_indice_range

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            # off-the-shelf annotation cache exists
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ann, group_indice_range = self._collect_annotations()

        # when cache on disk, rank0 saves items to an h5 file
        serialized_ann = [json.dumps(_) for _ in ann]
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            h5_ann = file.create_dataset("ann", (len(serialized_ann),), dtype=dt)
            h5_ann[:] = serialized_ann
            file.create_dataset("group_indice_range", data=json.dumps(group_indice_range))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info(f"data cache built")

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_dir = f"./accessory_data_cache/{config_identifier}"
        return cache_dir

    @staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            # cache has not yet been completed by rank 0
            assert dist.get_rank() != 0
            sleep(1)
        cache_file = h5py.File(Path(cache_dir) / "data.h5", "r")
        annotations = cache_file["ann"]
        group_indice_range = json.loads(cache_file["group_indice_range"].asstr()[()])
        return annotations, group_indice_range

    def get_item_func(self, index):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        return self.item_processor.process_item(data_item, training_mode=True)

    def __getitem__(self, index):
        try:
            return self.get_item_func(index)
        except Exception as e:
            if isinstance(e, DataNoReportException):
                pass
            elif isinstance(e, DataBriefReportException):
                logger.info(e)
            else:
                logger.info(
                    f"Item {index} errored, annotation:\n"
                    f"{self.ann[index]}\n"
                    f"Error:\n"
                    f"{traceback.format_exc()}"
                )
            for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    if index == indices_this_group[0]:
                        new_index = indices_this_group[-1]
                    else:
                        new_index = index - 1
                    return self[new_index]
            raise RuntimeError

    def groups(self):
        return list(self.group_indices.values())
