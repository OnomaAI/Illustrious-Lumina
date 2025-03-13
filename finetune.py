# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP with wandb logging.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
import warnings
import torch.nn.functional as F

from PIL import Image
# import cairosvg
from diffusers import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import bitsandbytes as bnb
import wandb  # <--- (1) Import wandb
from data import DataNoReportException, ItemProcessor, MyDataset, read_general
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue
#############################################################################
#                           Danbooru prompt processor for safety control                              #
CONVERTABLE_DICT = {
    "masterpiece" : ["amazing quality", "masterpiece", "masterpiece quality", "top quality", "most preferred", "professional"],
    "best quality" : ["best quality", "good", "high quality", "preferred", "best drawing", "best"],
    "bad quality" : ["bad quality", "low quality", "poor quality", "bad drawing", "badly drawn"],
    "worst quality" : ["displeasing", "worst drawing", "amateur", "worst quality"],
    "1girl" : ["one girl", "female", "girl", "female", "1girl", "woman"],
    "1boy" : ["one boy", "one male", "boy", "male", "1boy", "man"],
    "2boys" : ["two boys", "2boys", "2boy", "two male", "two men"],
    "2girls" : ["2girls", "two girls", "two female", "two women"],
    "3boys" : ["three boys", "3boys", "three male"],
    "3girls" : ["three girls", "3girls", "three female"],
    "lineart" : ["line drawing", "lineart", "line art"],
    "no lineart" : ["no lineart", "vector art", "without lines"],
    "lowres" : ["low resolution", "lowres", "low res", "low image quality"],
    "pov" : ["point of view", "pov", "first person view", "first person perspective"],
    "censor" : ["censored", "censor", "censorship"],
    "doctor (arknights)" : ["doctor (arknights)", "arknights doctor", "arknights protagonist"],
    "female doctor (arknights)" : ["female doctor (arknights)", "arknights female doctor"],
    "highres" : ["high resolution", "highres", "high res", "high image quality"],
    "solo" : ["solo", "alone", "single person", "single character", "single view", "protagonist"],
    "solo focus" : ["focus on single character", "solo focus", "single character focus"],
    "long hair" : ["long hair", "long haired", "long haired character"],
    "looking at viewer" : ["looking at camera", "looking at viewer", "eye contact", "looking at you"],
    "blush" : ["blush", "blushing", "embarrassed"],
    "simple background" : ["simple background", "plain background", "simple bg", "focus on character"],
    "full body" : ["full body", "full body shot", "full body view"],
    "upper body" : ["upper body", "upper body shot", "upper body view", "focusing on torso", "without legs focus"],
    "lower body" : ["lower body", "lower body shot", "lower body view", "without head focus"],
    "monochrome" : ["monochrome", "single toned", "gradient with one color"],
    "cowboy shot" : ["cowboy shot", "cowboy angle", "cowboy view", "cropped at thighs"],
    "greyscale" : ["greyscale", "grayscale", "without color", "black and white"],
    "nude" : ["nude", "naked", "nude character"],
    "alternate costume" : ["alternate costume", "alternate outfit", "alternate attire"],
    "day" : ["day", "daytime", "daylight", "sunny"],
    "night" : ["night", "nighttime", "dark", "moonlight"],
    "shadow" : ["shadow", "shadows", "shadowed", "shading"],
    "artist name" : ["artist name", "artist signature"],
    "close-up" : ["close-up", "close up", "closeup", "close shot", "close view"],
    "mugshot" :["mugshot", "mug shot", "mugshot view", "mug shot view", "criminal photo"],
    "lineup" : ["lineup", "line up", "line-up", "group shot", "group photo"],
    "signature" : ["signature", "artist signature"],
    "profile" : ["profile", "side profile", "profile view", "side view", "from side"],
    "multiple views" : ["multiple views", "multiple shots"],
    "from above" : ["from above", "aerial view", "top view", "high angle"],
    "from below" : ["from below", "low angle", "from beneath", "from under"],
    "from behind" : ["from behind", "rear view", "from the back", "back view"],
    "from side" : ["from side", "side view", "lateral view", "profile view"],
    "straight-on" : ["straight-on", "front view", "frontal view", "direct view"],
    "looking back" : ["looking back", "looking behind", "looking over shoulder"],
    "dutch angle" : ["dutch angle", "tilted angle", "slanted angle", "german angle", "oblique angle"],
    "sideways" : ["sideways", "rotated image"],
    "general" : ["general", "", "safe for work", "sfw", "safe"],
    "sensitive" : ["sfw", "casual", "sensitive"],
    "questionable" : ["nsfw", "with partial nudity", "questionable", "questionable content"],
    "explicit" : ["explicit", "nsfw", "with nudity", "adult content", "explicit material"],
}

popular_chars_names = ["momiji", "character", "futo", "inaba", "yor", "seija", "stout", "sakuya", "yazawa", "tamamo", "ellen", "d'arc", "murasa", "misaka", "hearn", "kisaragi", "kaku", "ichinose", "hatate", "suwako", "douji", "aqua", "yoko", "samidare", "kikuchi", "nilou", "yuyuko", "sekibanki", "asashio", "rumia", "megurine", "kotori", "formidable", "frieren", "satori", "shijou", "kyrielight", "kanako", "remilia", "koakuma", "gardevoir", "littner", "princess", "d.va", "saber", "higuchi", "koishi", "bridget", "minami", "inkling", "monster", "kokomi", "miho", "kasodani", "houraisan", "kongou", "artoria", "chen", "pyra", "patchouli", "konpaku", "tojo", "mercury", "shinobu", "tewi", "suika", "izumi", "shiroko", "inazuma", "kurodani", "akemi", "fujiwara", "mononobe", "kokoro", "nagae", "azusa", "youmu", "oma", "kafka", "c.c.", "arisu", "abigail", "mae", "yumemi", "manhattan", "mona", "shirakami", "zhongli", "shibuya", "kawashiro", "kaenbyou", "zero", "nakano", "yuudachi", "tao", "eula", "hoshimachi", "kasen", "raiden", "yuugi", "takane", "murakumo", "hoshii", "watanabe", "rio", "minamoto", "kaname", "minato", "pendragon", "williams", "udongein", "shower", "super", "ryuuko", "himekaidou", "mirko", "cammy", "sayaka", "riamu", "reimu", "yasaka", "komeiji", "nightbug", "tachyon", "kokichi", "lumine", "utsuho", "rem", "tatsumaki", "shimamura", "sonoda", "takagaki", "shenhe", "kagerou", "miki", "houjuu", "lillie", "nagato", "senketsu", "amami", "player", "byakuren", "junko", "asuna", "kashima", "komachi", "kinomoto", "power", "kagamine", "kirisame", "kogasa", "sanae", "souji", "nico", "seiga", "mokou", "aran", "iono", "usami", "nazrin", "akiyama", "kamisato", "joe", "miku", "nozomi", "shooter", "nahida", "luka", "mythra", "claudius", "kyoko", "yagokoro", "iku", "aya", "kaede", "takina", "morrigan", "amiya", "gokou", "yoshika", "suzuya", "dawn", "kamishirasawa", "shuten", "okita", "joseph", "reisalin", "ruri", "haruka", "nitori", "marnie", "plana", "renko", "shameimaru", "samus", "makoto", "holo", "doll", "yuuka", "hinanawi", "hatsune", "shiranui", "daiyousei", "kanzaki", "magician", "rembran", "reiuji", "jougasaki", "tohsaka", "maki", "ibuki", "karin", "kai", "white", "oshino", "koharu", "bowsette", "eiki", "toki", "ayaka", "cafe", "sagiri", "yelan", "zeppeli", "zelda", "wriggle", "hata", "ganaha", "saigyouji", "shimakaze", "mayuzumi", "shogun", "lorelei", "einzbern", "fuyuko", "knowledge", "sonico", "tifa", "rensouhou-chan", "rin", "kyouko", "kaguya", "serval", "nino", "ranko", "madoka", "flandre", "kisaki", "hong", "illyasviel", "koume", "hamakaze", "chun-li", "miko", "oyama", "shanghai", "joestar", "uzuki", "umi", "yui", "kaga", "tomoe", "mika", "mash", "ganyu", "ibaraki", "fubuki", "miorine", "dark", "ayanami", "arona", "2b", "boo", "eirin", "kazusa", "mio", "aensland", "anthonio", "von", "meiling", "parsee", "tachibana", "warrior", "kitagawa", "fumika", "marine", "yamame", "alter", "marisa", "rikka", "megumin", "moriya", "sparkle", "nishizumi", "matoi", "takao", "raikou", "briar", "minamitsu", "rei", "imaizumi", "asuka", "kazami", "hk416", "shiki", "nero", "keine", "amatsukaze", "karyl", "hina", "chino", "mari", "nanami", "izayoi", "yae", "onozuka", "nishikigi", "nishikino", "yamato", "makima", "suigintou", "sagisawa","mizuhashi", "yotsuba", "chiaki", "margatroid", "ushio", "mikoto", "ayase", "mai", "hitori", "venti", "agnes", "scathach", "yoimiya", "gawr", "sagume", "ooyodo", "reisen", "chihaya", "haruhi", "gumi", "akagi", "souryuu", "hirasawa", "homura", "shigure", "hibiki", "yuzuki", "acheron", "link", "sakura", "ryuujou", "atago", "inubashiri", "mami", "nue", "yukari", "eugen", "jeanne", "gura", "firefly", "hestia", "anchovy", "haruna", "aru", "houshou", "gotoh", "akatsuki", "kishin", "alice", "kijin", "hijiri", "kagiyama", "yakumo", "suisei", "ro-500", "keqing", "testarossa", "scarlet", "iowa", "suletta", "tenshi", "langley", "lockhart", "tatara", "mystia", "adachi", "rosa", "hoshiguma", "yuki", "hakurei", "furina", "daiwa", "mahiro", "aris", "suzumiya", "kochiya", "inoue", "fate", "nami", "hunter", "tenryuu", "shirasaka", "astolfo", "caesar", "prinz", "marin", "toyosatomimi", "kafuu", "takarada", "hoshino", "clownpiece", "cynthia", "miyako", "darjeeling", "sangonomiya", "chisato", "rice", "ikazuchi", "cirno", "maribel", "mizumiya", "niko", "kikirara", "riona"]
no_dropout_tokens = [
    # "low ",
    "lineart",
    "l" + "o" + "l" + "i" , # oh no
    "shota", # these are critical tags...
    " art",
    "foreshortening",
    "exaggerat",
    "disembodied",
    "rough",
    "sketch",
    "amateur",
    "displeasing",
    "jaggy",
    #"close up",
    #"close-up",
    "cropped",
    "empty",
    "plain",
    # "from ",
    # " body",
    "multiple",
    "artifact",
    "toon",
    "lowres",
    "koma",
    # "pov",
    "censor",
    "upside" # these are critical tags for image comprehension
    "guro",
    "scat",
    "gore",
    "cover", # now some scan / copyrighted material
    "album",
    "3d",
    "render",
    "name",
    "logo",
    "artist",
    "sign",
    "username",
    "parody",
    "manga",
    "comic",
    "letterbox",
    "watermark",
    "scan",
    "doujin",
    "anatomical nonsense", #for better body part recognition
    "bad hands",
    "bad feet",
    "bad proportions",
    "quality",
    "bad aspect",
    "extra digits",
    "bad reflection",
    "artistic",
    "halo", # blue archive please
    "poorly drawn",
    #"chromatic",
    "chiaroscuro",
    " medium",
    "cropped",
    "tomboy", # these are some case that model might be confused
    "trap",
    "tomgirl",
    "crossdressing",
    "androgynous",
    "futa", 
    "girl", # gender / persons are important
    "boy",
    "men",
    "female",
    "cosplay",
    "male",
    "other",
    "explicit",
    "questionable",
    "simple",
    "underwear",
    "panties",
    "pubic",
    "topless",
#    "background",
    "abstract",
    "monochrome", "single toned", "gradient with one color",
    "greyscale",
    "various",
    "koma",
    "ai-generated" # mark the image is generated by AI
] + [
    "pus"+ "sy",
    "nip"+"ple",
    "pen"+"is",
    "an" +"us" # sexual tokens should not be dropped and always checked
    ]# The tokens that contains this will not be dropped

no_dropout_tokens = set(no_dropout_tokens + popular_chars_names)
strict_no_dropout_tokens = [
    "explicit",
    "questionable",
    "sensitive",
    "nsfw",
    "nudity",
    "adult content",
    "photo",
    "guro",
    "koma",
    "panties",
    "underwear",
    "lo" + "li",
    "sho" + "ta",
    "anime",
    "comic",
    "manga",
    "multiple",
    "chart",
    "collage",
    "diagram",
    "sheet",
    "lineup",
    "panels",
    "graph",
    "turnaround",
    "variation",
    "expression",
    "logo",
    "username",
    "text",
    "copyright",
    "artifact",
    "family tree",
    "bad ",
    "sign",
    "pubic",
    "jaggy",
    "topless",
    "bottomless",
    "twitter",
    "scat",
    "nude",
    "naked",
    "r-18",
    "pus"+ "sy",
    "nip"+"ple",
    "pen"+"is",
    "an" +"us", # sexual tokens should not be dropped and always checked
    "real",
    "figma"
] # the tokens which should never be dropped out

def dropout_tags(tags_string, dropout_p=0.35):
    tags = tags_string.split(",")
    tags = [t.strip() for t in tags]
    tags = [t for t in tags if t]
    random.shuffle(tags)
    filtered = []
    for t in tags:
        if random.random() > dropout_p or t in no_dropout_tokens or check_strict_terms(t):
            if t in CONVERTABLE_DICT:
                tags_available = CONVERTABLE_DICT[t]
                index = random.randint(0, len(tags_available)) # if 0, then it will be the same tag, else it will be a different tag
                if index == 0:
                    filtered.append(t)
                else:
                    filtered.append(tags_available[index-1])
                continue
            filtered.append(t)
    return ", ".join(filtered)

@functools.lru_cache(maxsize=16384)
def check_strict_terms(tag):
    for t in strict_no_dropout_tokens:
        if t in tag:
            return True
    return False


#############################################################################
#                            Data item Processor                            #
#############################################################################

class NonRGBError(DataNoReportException):
    pass

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform, use_cached_latents=False, resolution=1024):
        self.image_transform = transform
        self.special_format_set = set()
        self.use_cached_latents = use_cached_latents
        self.train_res = resolution

    def process_item(self, data_item, training_mode=False):
        if "super_high_quality_caption" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url))
            text = data_item["super_high_quality_caption"]
            system_prompt = (
                "You are an assistant designed to generate high-quality images "
                "with the highest degree of image-text alignment based on textual "
                "prompts. <Prompt Start> "
            )
        elif "path" in data_item:
            url = data_item["path"]
            image = Image.open(read_general(url))
            text = data_item["prompt"]
            system_prompt = (
                "You are an assistant designed to generate images "
                "based on user prompts. <Prompt Start> "
            )
        elif "image_path" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url))
            if "prompt" in data_item:
                text = data_item["prompt"]
                text = dropout_tags(text)
                system_prompt = (
                    "You are an assistant designed to generate images "
                    "based on user prompts. <Prompt Start> "
                )
            elif "sentence" in data_item:
                text = data_item["sentence"]
                system_prompt = (
                    "You are an assistant designed to generate images "
                    "based on user prompts. <Prompt Start> "
                )
            elif "alttext" in data_item:
                text = data_item["alttext"]
                system_prompt = (
                    "You are an assistant designed to generate images "
                    "based on alttext information. <Prompt Start> "
                )
            elif "tags" in data_item:
                text = data_item["tags"]
                system_prompt = (
                    "You are an assistant designed to generate images "
                    "based on danbooru tags. <Prompt Start> "
                )
            else:
                raise ValueError(f"Unrecognized item: {data_item}")
        else:
            raise ValueError(f"Unrecognized item: {data_item}")
        # Check for cached latents if enabled:
        if self.use_cached_latents:
            # Suppose the latent file is <image_path_no_ext>_<resolution>.npz
            # Or define your own naming rule. For example:
            base, ext = os.path.splitext(url)
            # You could store a known resolution or multiple. We'll do something simple:
            # E.g. training might be 1024 by default:
            res = self.train_res
            latent_path = f"{base}_{res}.npz"

            if os.path.exists(latent_path):
                # Load from .npz
                try:
                    arr = np.load(latent_path)["latent"]  # shape = (16, H//8, W//8)
                    # Convert to torch
                    latent_tensor = torch.from_numpy(arr)
                    # Return this latent in place of an image
                    if text is None or text.strip() == "":
                        text = ""
                    text = system_prompt + text
                    return latent_tensor, text
                except Exception as e:
                    print(f"[Warning] Could not load {latent_path}, fallback to normal image: {e}")

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode not in self.special_format_set:
                self.special_format_set.add(mode)
                print(mode, url)
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            elif mode == "P" or mode == "L":
                image = image.convert("RGB")
            else:
                raise NonRGBError()

        image = self.image_transform(image)

        if text is None or text.strip() == "":
            text = ""
        text = system_prompt + text
        return image, text


#############################################################################
#                           Training Helper Functions                       #
#############################################################################

def apply_average_pool(latent, factor):
    """
    Apply average pooling to downsample the latent.
    """
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)

def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    return image, caps

def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()

@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt") if logging_dir else logging.NullHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model

def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        # check strict terms
        if not any([check_strict_terms(caption)]) and random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

#############################################################################
#                                Training Loop                              #
#############################################################################

def main(args):
    """
    Trains a new DiT model with optional wandb logging.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
            )
        )
        
        # ------------------- (2) Initialize wandb (main process) -------------------
        wandb.init(
            project="Lumina-T2I",  # Change to your W&B project name
            name=os.path.basename(args.results_dir),
            config=vars(args),
            dir=args.results_dir,
        )
        
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")

    # create tokenizers
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"

    # create text encoders
    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
    ).cuda()
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    logger.info(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size

    # Create model:
    model = models.__dict__[args.model](
        in_channels=16,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size
    print(f"Model patch size: {model_patch_size}")

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpected keys: {unexpected_keys}")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        logger.info("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"model:\n{model}\n")

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(
        device
    )

    logger.info("AdamW eps 1e-15 betas (0.9, 0.95)")
    if args.use_8bit_adam:
        opt = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=tuple(args.betas),
            eps=args.eps,
            optim_bits=8
        )
        print("[INFO] Using 8-bit AdamW optimizer (bitsandbytes).")
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=tuple(args.betas),
            eps=args.eps,
        )
    if not args.skip_optimizer_load and args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd
        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        if not args.resume:
            resume_step = 0
        else:
            with open(os.path.join(args.resume, "resume_step.txt")) as f:
                resume_step = int(f.read().strip())


    # ---------------------- MODIFIED: Support multiple resolutions ----------------------
    # Add a new argument (see below in the parser) for comma-separated training resolutions.
    # For each resolution, try to get resolution-specific global and micro batch sizes
    # (if not provided, fall back to default values).
    train_resolutions = [int(x.strip()) for x in args.train_resolutions.split(",")]
    data_collection = {}
    for train_res in train_resolutions:
        logger.info(f"Creating data for resolution {train_res}")

        global_bsz = args.__dict__.get(f"global_bsz_{train_res}", args.global_bsz_default)
        local_bsz = global_bsz // dp_world_size
        micro_bsz = args.__dict__.get(f"micro_bsz_{train_res}", args.micro_bsz_default)
        assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."

        patch_size = 8 * model_patch_size
        logger.info(f"patch size: {patch_size}")
        max_num_patches = round((train_res / patch_size) ** 2)
        logger.info(f"Limiting number of patches to {max_num_patches}.")
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
        logger.info("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
        image_transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            item_processor=T2IItemProcessor(
                transform=image_transform,
                use_cached_latents=args.use_cached_latents,
                resolution=train_res
            ),
            cache_on_disk=args.cache_data_on_disk,
        )
        num_samples = global_bsz * args.max_steps
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} "
                    f"({num_samples / len(dataset):.2f} epochs)")
        sampler = get_train_sampler(
            dataset,
            dp_rank,
            dp_world_size,
            global_bsz,
            args.max_steps,
            resume_step,
            args.global_seed + train_res * 100,
        )
        logger.info(f"Sampler ready, loading DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )

        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=args.snr_type,
            do_shift=not args.no_shift,
            seq_len=(train_res // 16) ** 2,
        )

        data_collection[train_res] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
            "transport": transport,
        }
    # ------------------------------------------------------------------------------------

    # Prepare models for training:
    model.train()

    logger.info(f"Training for {args.max_steps:,} steps...")

    for step in range(resume_step, args.max_steps):
        start_time = time()
        for train_res, data_pack in data_collection.items():
            x, caps = next(data_pack["loader_iter"])
            x = [img.to(device, non_blocking=True) for img in x]

            with torch.no_grad():
                vae_scale = {
                    "sdxl": 0.13025,
                    "sd3": 1.5305,
                    "ema": 0.18215,
                    "mse": 0.18215,
                    "cogvideox": 1.15258426,
                    "flux": 0.3611,
                }["flux"]
                vae_shift = {
                    "sdxl": 0.0,
                    "sd3": 0.0609,
                    "ema": 0.0,
                    "mse": 0.0,
                    "cogvideox": 0.0,
                    "flux": 0.1159,
                }["flux"]

                if step == resume_step:
                    warnings.warn(f"vae scale: {vae_scale}    vae shift: {vae_shift}")
                # Map input images to latent space + normalize latents:
                for i, img in enumerate(x):
                    # If x[i] is a 3-channel image => shape=(3,H,W). If x[i] is a 16-channel latent => shape=(16,H',W').
                    if img.shape[0] == 16:
                        # This means it's already a latent, skip VAE
                        pass
                    else:
                        # It's an image => run VAE encoding
                        x[i] = (vae.encode(img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale
                        x[i] = x[i].float()

            with torch.no_grad():
                cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

            loss_item = 0.0
            loss_256_item = 0.0
            loss_1024_item = 0.0

            opt.zero_grad()

            # Number of bins, for loss recording
            n_loss_bins = 20
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            loss_bins_256 = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
            bin_occurrence_256 = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss_256 = torch.zeros(n_loss_bins, device="cuda")

            for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
                mb_st = mb_idx * data_pack["micro_bsz"]
                mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
                last_mb = mb_ed == data_pack["local_bsz"]

                x_mb = x[mb_st:mb_ed]
                x_mb_256 = [apply_average_pool(xx, 4) for xx in x_mb]

                cap_feats_mb = cap_feats[mb_st:mb_ed]
                cap_mask_mb = cap_mask[mb_st:mb_ed]

                model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
                with {
                    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[args.precision]:
                    loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs)
                    loss_dict_256 = data_pack["transport"].training_losses(model, x_mb_256, model_kwargs)

                loss_1024 = loss_dict["loss"].sum() / data_pack["local_bsz"]
                loss_256 = loss_dict_256["loss"].sum() / data_pack["local_bsz"]
                loss = loss_1024 + loss_256
                loss_item += loss.item()
                loss_1024_item += loss_1024.item()
                loss_256_item += loss_256.item()
                with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                    loss.backward()

                # for bin-wise loss recording
                bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                detached_loss = loss_dict["loss"].detach()
                bin_indices_256 = torch.bucketize(loss_dict_256["t"].cuda(), loss_bins_256, right=True) - 1
                detached_loss_256 = loss_dict_256["loss"].detach()

                for i_bin in range(n_loss_bins):
                    mask_1024 = bin_indices == i_bin
                    mask_256 = bin_indices_256 == i_bin
                    bin_occurrence[i_bin] += mask_1024.sum()
                    bin_sum_loss[i_bin] += detached_loss[mask_1024].sum()
                    bin_occurrence_256[i_bin] += mask_256.sum()
                    bin_sum_loss_256[i_bin] += detached_loss_256[mask_256].sum()

            grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)
            dist.all_reduce(bin_occurrence)
            dist.all_reduce(bin_sum_loss)
            dist.all_reduce(bin_occurrence_256)
            dist.all_reduce(bin_sum_loss_256)

            # Log to TensorBoard & W&B
            if tb_logger is not None:
                tb_logger.add_scalar(f"{train_res}/loss", loss_item, step)
                tb_logger.add_scalar(f"{train_res}/loss_256", loss_256_item, step)
                tb_logger.add_scalar(f"{train_res}/loss_1024", loss_1024_item, step)
                tb_logger.add_scalar(f"{train_res}/grad_norm", grad_norm, step)
                tb_logger.add_scalar(f"{train_res}/lr", opt.param_groups[0]["lr"], step)
                for i_bin in range(n_loss_bins):
                    if bin_occurrence[i_bin] > 0:
                        bin_avg_loss = (bin_sum_loss[i_bin] / bin_occurrence[i_bin]).item()
                        tb_logger.add_scalar(
                            f"{train_res}/loss-bin{i_bin+1}-{n_loss_bins}",
                            bin_avg_loss,
                            step,
                        )
                    if bin_occurrence_256[i_bin] > 0:
                        bin_avg_loss_256 = (bin_sum_loss_256[i_bin] / bin_occurrence_256[i_bin]).item()
                        tb_logger.add_scalar(
                            f"{train_res}/loss_256-bin{i_bin+1}-{n_loss_bins}",
                            bin_avg_loss_256,
                            step,
                        )

            # ------------------- (3) Log metrics to wandb (main process only) -------------------
            if rank == 0:
                log_dict = {
                    f"{train_res}/loss": loss_item,
                    f"{train_res}/loss_256": loss_256_item,
                    f"{train_res}/loss_1024": loss_1024_item,
                    f"{train_res}/grad_norm": grad_norm,
                    f"{train_res}/lr": opt.param_groups[0]["lr"],
                }
                for i_bin in range(n_loss_bins):
                    if bin_occurrence[i_bin] > 0:
                        log_dict[f"{train_res}/loss-bin{i_bin+1}-{n_loss_bins}"] = (
                            bin_sum_loss[i_bin] / bin_occurrence[i_bin]
                        ).item()
                    if bin_occurrence_256[i_bin] > 0:
                        log_dict[f"{train_res}/loss_256-bin{i_bin+1}-{n_loss_bins}"] = (
                            bin_sum_loss_256[i_bin] / bin_occurrence_256[i_bin]
                        ).item()
                wandb.log(log_dict, step=step)

            opt.step()
            end_time = time()

            # Update training stats
            metrics = data_pack["metrics"]
            metrics["loss"].update(loss_item)
            metrics["loss_1024"].update(loss_1024_item)
            metrics["loss_256"].update(loss_256_item)
            metrics["grad_norm"].update(grad_norm)
            metrics["Secs/Step"].update(end_time - start_time)
            metrics["Imgs/Sec"].update(data_pack["global_bsz"] / (end_time - start_time))

            for i_bin in range(n_loss_bins):
                if bin_occurrence[i_bin] > 0:
                    bin_avg_loss = (bin_sum_loss[i_bin] / bin_occurrence[i_bin]).item()
                    metrics[f"bin_1024_{i_bin + 1:02}-{n_loss_bins}"].update(
                        bin_avg_loss, int(bin_occurrence[i_bin].item())
                    )
                if bin_occurrence_256[i_bin] > 0:
                    bin_avg_loss_256 = (bin_sum_loss_256[i_bin] / bin_occurrence_256[i_bin]).item()
                    metrics[f"bin_256_{i_bin + 1:02}-{n_loss_bins}"].update(
                        bin_avg_loss_256, int(bin_occurrence_256[i_bin].item())
                    )

            if (step + 1) % args.log_every == 0:
                torch.cuda.synchronize()
                logger.info(
                    f"Res{train_res}_{train_res//4}: (step{step + 1:07d}) "
                    f"lr{opt.param_groups[0]['lr']:.6f} "
                    + " ".join([f"{key}:{str(metrics[key])}" for key in sorted(metrics.keys())])
                )

            start_time = time()

        update_ema(model_ema, model)

        # Save DiT checkpoint:
        if step == 0 or (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()
    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT_Llama2_7B_patch2")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    # Define default batch sizes for resolution 1024 (if you do not provide resolution-specific values, these are used)
    parser.add_argument("--global_bsz_1024", type=int, default=256)
    parser.add_argument("--micro_bsz_1024", type=int, default=1)
    for res in [4096, 2048, 1536, 768, 512, 384, 256]:
        parser.add_argument(f"--global_bsz_{res}", type=int, default=256)
        parser.add_argument(f"--micro_bsz_{res}", type=int, default=1)
    # Add new default batch size arguments (to be used if resolution-specific ones are not provided)
    parser.add_argument("--global_bsz_default", type=int, default=256, help="Default global batch size for training resolutions if not specified")
    parser.add_argument("--micro_bsz_default", type=int, default=1, help="Default micro batch size for training resolutions if not specified")
    # Add an argument for comma-separated list of training resolutions
    parser.add_argument("--train_resolutions", type=str, default="1024", help="Comma separated list of training resolutions to support. E.g.: '384,512,768,1024,1536,2048,4096'")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--use_8bit_adam", action="store_true", 
                        help="Use 8-bit AdamW optimizer from bitsandbytes.")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument(
        "--use_cached_latents",
        action="store_true",
        help="If set, the dataset will try to load .npz latents instead of raw images."
    )
    parser.add_argument(
        "--use_xformers", 
        action="store_true", 
        help="Enable memory-efficient attention via xFormers"
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help=(
            "Initialize the model weights from a checkpoint folder. Compared to --resume, "
            "this loads neither the optimizer states nor the data loader states."
        ),
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument("--skip_optimizer_load", action="store_true")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--no_shift", action="store_true")
    args = parser.parse_args()

    main(args)
