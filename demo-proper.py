import gradio as gr
import json
import math
import os
import random
import socket
import time
import gc


from diffusers.models import AutoencoderKL
import numpy as np
import torch
# import torch.distributed as dist # Removed distributed
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import models # Assuming models.py is in the same directory or accessible
from transport import Sampler, create_transport # Assuming transport.py is accessible
import argparse


# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Arguments for model loading and inference.")
parser.add_argument('--ckpt_path', type=str, default="checkpoint/consolidated.00-of-01.pth", help="Path to checkpoint file")
parser.add_argument('--model_args_path', type=str, default="checkpoint/model_args.pth", help="Path to model args file")

args = parser.parse_args()
# --- Globals for Models and Config ---
# --- Set these paths and configurations correctly ---
CKPT_PATH = args.ckpt_path # IMPORTANT: Set path to your specific checkpoint file
MODEL_ARGS_PATH = args.model_args_path # IMPORTANT: Set path to model args
VAE_TYPE = os.environ.get("VAE_TYPE", "flux") # Or "ema", "mse", "sdxl"
PRECISION = os.environ.get("PRECISION", "bf16") # Or "fp32"
TEXT_ENCODER_MODEL = os.environ.get("TEXT_ENCODER_MODEL", 'google/gemma-2-2B')
# --- End of required path configurations ---

# Will be loaded by load_models() - Initialized to None
tokenizer = None
text_encoder = None
vae = None
model = None
sampler = None
transport = None
device = "cuda" if torch.cuda.is_available() else "cpu"
# Keep dtype definition global as it's used in loading and inference
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[PRECISION]
train_args = None
cap_feat_dim = None

# --- Helper Functions (adapted from original script) ---

def encode_prompt(prompt_batch, text_encoder_model, tokenizer_model, target_device, target_dtype):
    """
    Encodes prompts, moving the text encoder to the target device temporarily.
    """
    captions = []
    for caption in prompt_batch:
         # Simplified: always use the provided caption
        captions.append(caption if isinstance(caption, str) else caption[0])

    # Move text_encoder to the target device for encoding
    text_encoder_model.to(target_device)
    prompt_embeds, prompt_masks = None, None # Initialize
    
    try:
        with torch.no_grad():
            text_inputs = tokenizer_model(
                captions,
                padding=True,
                pad_to_multiple_of=8,
                max_length=256, # Make this configurable if needed
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(target_device)
            prompt_masks = text_inputs.attention_mask.to(target_device)

            # Use autocast on the target device during model inference
            with torch.autocast(device_type=target_device.split(':')[0], dtype=target_dtype):
                 prompt_embeds = text_encoder_model(
                    input_ids=text_input_ids,
                    attention_mask=prompt_masks,
                    output_hidden_states=True,
                 ).hidden_states[-2]

    finally:
         # Move text encoder back to CPU regardless of success/failure
        text_encoder_model.to("cpu")
        if target_device != "cpu":
             torch.cuda.empty_cache() # Clear cache after moving model off GPU
        gc.collect() # Explicitly call garbage collector

    # Return results moved to CPU to minimize GPU memory usage outside inference
    return prompt_embeds.cpu(), prompt_masks.cpu()


def none_or_str(value): # Keep this helper if used by create_transport or sampler
    if value == "None" or value is None:
        return None
    return str(value)

# --- Model Loading Function ---

def load_models():
    global tokenizer, text_encoder, vae, model, sampler, transport, device, dtype, train_args, cap_feat_dim

    print("--- Starting Model Loading (targeting CPU initially) ---")
    torch.set_grad_enabled(False)

    # Load training args
    if not os.path.exists(MODEL_ARGS_PATH):
        raise FileNotFoundError(f"Model args file not found: {MODEL_ARGS_PATH}")
    # Force loading pickled data with weights_only=False if it contains non-tensor data (like args)
    train_args = torch.load(MODEL_ARGS_PATH, map_location='cpu', weights_only=False)
    print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    # --- Load Tokenizer ---
    print(f"Creating Tokenizer: {TEXT_ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEL)
    tokenizer.padding_side = "right"
    print("Tokenizer loaded.")

    # --- Load Text Encoder (to CPU) ---
    print(f"Creating Text Encoder: {TEXT_ENCODER_MODEL}")
    # Load with desired dtype but map to CPU initially
    text_encoder = AutoModel.from_pretrained(
        TEXT_ENCODER_MODEL, torch_dtype=dtype
    ).eval().to("cpu") # Explicitly move to CPU
    cap_feat_dim = text_encoder.config.hidden_size
    print("Text encoder loaded to CPU.")

    # --- Load VAE (to CPU) ---
    print(f"Loading VAE: {VAE_TYPE}")
    vae_path = f"black-forest-labs/FLUX.1-dev" if VAE_TYPE == "flux" else \
               f"stabilityai/sdxl-vae" if VAE_TYPE == "sdxl" else \
               f"stabilityai/sd-vae-ft-{VAE_TYPE}" if VAE_TYPE in ["ema", "mse"] else None
    if vae_path is None:
         raise ValueError(f"Unsupported VAE type: {VAE_TYPE}")

    subfolder = "vae" if VAE_TYPE == "flux" else None
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=dtype).to("cpu") # Explicitly move to CPU
    vae.requires_grad_(False)
    vae.eval()
    print("VAE loaded to CPU.")

    # --- Load DiT Model (to CPU) ---
    print(f"Creating DiT model: {train_args.model}")
    if not hasattr(models, train_args.model):
         raise AttributeError(f"Model class {train_args.model} not found in models module.")

    # Instantiate model on CPU
    model = models.__dict__[train_args.model](
        in_channels=16, # Should match FLUX VAE latent channels
        qk_norm=getattr(train_args, 'qk_norm', True), # Use getattr for safety
        cap_feat_dim=cap_feat_dim,
        # Add other args as needed
    ).eval().to("cpu") # Ensure model is created on CPU and in eval mode

    print(f"Loading DiT checkpoint: {CKPT_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"DiT checkpoint not found: {CKPT_PATH}")

    state_dict = torch.load(CKPT_PATH, map_location='cpu') # Load state dict to CPU

    # Handle potential 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[len('module.'):] if k.startswith('module.') else k] = v
    del state_dict # Free memory

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys: print(f"Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys: print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
    del new_state_dict # Free memory

    # Model remains on CPU here
    print("DiT model loaded and checkpoint applied to CPU.")

    # --- Setup Sampler ---
    path_type = "Linear"
    prediction = "velocity"
    loss_weight = None
    train_eps = None
    sample_eps = None

    transport = create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
    sampler = Sampler(transport) # Sampler itself doesn't hold large weights
    print("Sampler initialized.")
    print("--- Model Loading Complete ---")

# --- Gradio Inference Function ---

def generate_image(
    prompt,
    negative_prompt,
    system_type,
    solver,
    resolution_str,
    guidance_scale,
    num_steps,
    seed,
    time_shifting_factor, # For DPM solver flow shift
    t_shift,             # For ODE solver time shift
    # ODE specific args
    atol=1e-6,
    rtol=1e-3,
):
    """Generates an image, moving models to GPU only when needed."""
    global model, vae, text_encoder, tokenizer, sampler # Access global models

    if model is None or vae is None or text_encoder is None or tokenizer is None or sampler is None:
        return None, "Models not loaded. Please wait or check console.", -1

    print("\n--- Starting Generation ---")
    start_time = time.time()

    # --- Prepare Inputs ---
    if int(seed) == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = int(seed)
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    print(f"Using Seed: {current_seed}")

    # System Prompt Logic (same as before)
    # ... (system prompt selection logic remains unchanged) ...
    if system_type == "align":
        system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "  # noqa
    elif system_type == "base":
        system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  # noqa
    elif system_type == "aesthetics":
        system_prompt = "You are an assistant designed to generate high-quality images with highest degree of aesthetics based on user prompts. <Prompt Start> "  # noqa
    elif system_type == "real":
        system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "  # noqa
    elif system_type == "4grid":
        system_prompt = "You are an assistant designed to generate four high-quality images with highest degree of aesthetics arranged in 2x2 grids based on user prompts. <Prompt Start> "  # noqa
    elif system_type == "tags":
        system_prompt = "You are an assistant designed to generate high-quality images based on user prompts based on danbooru tags. <Prompt Start> "
    elif system_type == "empty":
        system_prompt = ""
    else: # Default or fallback
        system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "


    full_prompt = system_prompt + prompt
    full_negative_prompt = system_prompt + negative_prompt if negative_prompt else ""

    try:
        w_str, h_str = resolution_str.split("x")
        w, h = int(w_str), int(h_str)
        latent_w, latent_h = w // 8, h // 8 # Assume VAE stride 8
    except Exception as e:
        print(f"Error parsing resolution: {e}")
        return None, f"Invalid resolution format: {resolution_str}. Use WxH.", current_seed

    # --- Encode Prompts (using temporary GPU placement) ---
    print("Encoding prompts...")
    # encode_prompt handles moving text_encoder to device and back to CPU
    cap_feats_cpu, cap_mask_cpu = encode_prompt(
        [full_prompt, full_negative_prompt],
        text_encoder,
        tokenizer,
        device, # Target device for encoding
        dtype   # Target dtype for encoding
    )
    print("Prompt encoding complete (Text Encoder back on CPU).")

    # --- Prepare for Sampling ---
    samples = None # Initialize samples variable
    with torch.no_grad():
        try:
            print("Moving DiT model to GPU...")
            model.to(device) # Move main model to GPU

            # Initial noise needs to be on the target device
            z = torch.randn([1, 16, latent_h, latent_w], device=device, dtype=dtype) # Match latent channels (16 for flux)
            z = z.repeat(2, 1, 1, 1) # Repeat for 2 prompts (pos, neg) - Model forward handles CFG splitting

            # Model kwargs need tensors on the target device
            model_kwargs = dict(
                cap_feats=cap_feats_cpu.to(device, dtype=dtype), # Move features to GPU
                cap_mask=cap_mask_cpu.to(device),           # Move mask to GPU
                cfg_scale=guidance_scale,
            )
            del cap_feats_cpu, cap_mask_cpu # Free CPU memory
            gc.collect()

            # --- Perform Sampling (on GPU) ---
            print(f"Starting sampling with solver: {solver}, steps: {num_steps}...")
            with torch.autocast(device_type=device.split(':')[0], dtype=dtype): # Autocast for the diffusion model forward pass
                if solver == "dpm":
                    # DPM typically uses Linear/velocity, create a temporary specific sampler
                    _transport = create_transport("Linear", "velocity")
                    _sampler = Sampler(_transport)
                    sample_fn = _sampler.sample_dpm(
                        model.forward_with_cfg, # Pass the model method directly
                        model_kwargs=model_kwargs,
                    )
                    # DPM sample_fn might need z repeated differently or handle cfg internally
                    # The original code's `forward_with_cfg` likely handles the split, so z shape [2, C, H, W] is correct
                    samples = sample_fn(z, steps=num_steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=time_shifting_factor)

                else: # ODE Solvers
                    # Use the globally loaded sampler, but ensure model call uses the GPU model
                    sample_fn = sampler.sample_ode(
                        sampling_method=solver, num_steps=num_steps,
                        atol=atol, rtol=rtol,
                        time_shifting_factor=t_shift
                    )
                    # sample_fn expects (z, model_call, **model_kwargs)
                    # Pass the GPU model's method directly
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]

            # Sampling finished, samples are on GPU. Keep only the positive sample(s) for decoding.
            samples = samples[:1].detach() # Detach from graph, take positive sample(s) (index 0)

        finally:
            # Move DiT model back to CPU regardless of sampling success/failure
            print("Moving DiT model back to CPU...")
            model.to("cpu")
            del model_kwargs # Remove reference to tensors potentially on GPU
            del z
            if device != "cpu":
                torch.cuda.empty_cache()
            gc.collect()


    # --- Decode and Post-process ---
    pil_image = None
    try:
        if samples is None:
             raise RuntimeError("Sampling failed, 'samples' tensor is None.")
        print("Moving VAE to GPU for decoding...")
        vae.to(device) # Move VAE to GPU

        print("Decoding samples...")
        with torch.no_grad(): # No gradients needed for VAE decode
            # Ensure samples are on the correct device and dtype for VAE
            samples = samples.to(device=device, dtype=vae.dtype)

            # VAE scaling factor and shift (using previously determined values)
            scaling_factor = vae.config.scaling_factor 
            shift_factor = vae.config.shift_factor

            samples = samples / scaling_factor + shift_factor
            # Use autocast for VAE decoding as well if using mixed precision
            with torch.autocast(device_type=device.split(':')[0], dtype=dtype):
                decoded_samples = vae.decode(samples)[0] # Decode expects input matching VAE dtype

        # Move decoded samples to CPU for post-processing
        decoded_samples = decoded_samples.cpu()
        print("Decoding complete.")

        # Normalize [approx -1, 1] -> [0, 1] -> PIL (on CPU)
        decoded_samples = (decoded_samples + 1.0) / 2.0
        decoded_samples.clamp_(0.0, 1.0)

        # Convert to PIL Image (already on CPU)
        pil_image = to_pil_image(decoded_samples[0].float()) # Use first sample, ensure float

    finally:
         # Move VAE back to CPU regardless of decoding success/failure
        print("Moving VAE back to CPU...")
        vae.to("cpu")
        del samples # Remove reference to tensor
        del decoded_samples
        if device != "cpu":
             torch.cuda.empty_cache()
        gc.collect()


    end_time = time.time()
    generation_time = round(end_time - start_time, 2)
    print(f"Generation finished in {generation_time} seconds.")

    if pil_image is None:
        return None, "Error during generation or decoding.", current_seed

    return pil_image, f"Generated in {generation_time}s", current_seed


# --- Load Models Before Launching UI ---
load_models() # Load models to CPU initially

# --- Create Gradio Interface (UI definition remains the same) ---
with gr.Blocks() as demo:
    gr.Markdown("# Lumina 2.0 Generation Demo")
    gr.Markdown("Enter a prompt and adjust settings to generate an image using Lumina 2.0 (Memory Optimized).")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", placeholder="e.g., A photo of an astronaut riding a horse on the moon")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="e.g., low quality, blurry, text, watermark")
            system_type = gr.Dropdown(
                label="System Prompt Type",
                choices=["align", "base", "aesthetics", "real", "4grid", "tags", "empty"],
                value="real"
            )
            resolution = gr.Dropdown(
                label="Resolution (Width x Height)",
                choices=["1024x1024", "1280x768", "768x1280", "1536x1024", "1024x1536"], # Common resolutions
                value="1024x1024"
            )
            solver = gr.Dropdown(
                label="Solver",
                choices=["dpm", "euler", "midpoint", "heun", "rk4"],
                value="dpm"
            )
            run_button = gr.Button("Generate Image", variant="primary")
        with gr.Column(scale=1):
            guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=15.0, step=0.5, value=4.0)
            num_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=200, step=1, value=50)
            seed = gr.Number(label="Seed", value=-1, precision=0) # -1 for random
            time_shifting_factor = gr.Slider(label="Time Shift Factor (DPM)", minimum=0.0, maximum=10.0, step=0.1, value=1.0)
            t_shift = gr.Slider(label="T Shift (ODE)", minimum=0, maximum=10, step=1, value=4)

    with gr.Row():
        output_image = gr.Image(label="Generated Image")

    with gr.Row():
        status_text = gr.Textbox(label="Status", interactive=False)
        final_seed = gr.Number(label="Seed Used", interactive=False)

    run_button.click(
        fn=generate_image,
        inputs=[
            prompt, negative_prompt, system_type, solver, resolution,
            guidance_scale, num_steps, seed, time_shifting_factor, t_shift
        ],
        outputs=[output_image, status_text, final_seed]
    )

    gr.Examples(
        examples=[
            ["A hyperrealistic photo of a cat wearing sunglasses and playing a saxophone on a beach at sunset", "", "real", "dpm", "1024x1024", 4.5, 40, -1, 1.0, 4],
            ["cinematic film still, A vast cyberpunk cityscape at night, raining, neon lights reflecting on wet streets, high detail", "low detail, drawing, illustration, sketch", "aesthetics", "euler", "1280x768", 5.0, 60, -1, 1.0, 4],
            ["Macro shot of a dewdrop on a spider web, intricate details, forest background, bokeh", "blurry, unfocused", "real", "dpm", "1024x1024", 4.0, 50, 12345, 1.0, 4],
        ],
        inputs=[prompt, negative_prompt, system_type, solver, resolution, guidance_scale, num_steps, seed, time_shifting_factor, t_shift],
        outputs=[output_image, status_text, final_seed],
        fn=generate_image
    )


# --- Launch the Gradio App ---
if __name__ == "__main__":
    # Consider adding share=True if you need external access
    demo.launch()
