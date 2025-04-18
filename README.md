# Illustirous-Lumina

Lumina is a powerful image generation model based on Diffusion Transformers (DiT) architecture. Originally developed by [Alpha-VLLM Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0), we've fine-tuned this model for our experimental purposes and are providing this demo to showcase its capabilities. Our fine-tuned model is available on [Hugging Face](https://huggingface.co/OnomaAIResearch/Illustrious-Lumina-v0.03). This documentation provides instructions for setting up and running our tuned version of Lumina.

## Prerequisites

Before setting up the environment, you need to prepare access tokens for the following models:

1. [GoogleGemma-2-2b](https://huggingface.co/google/gemma-2-2b) - You will need a gated access token for this model
2. [BlackForestLabs FLUX.1-dev](https://huggingface.co/google/gemma-2-2b) - You will need a gated acess token for this model

Make sure you have a requested and reveived access to these models on HuggingFace before proceeding with the setup.

## Important Notes
**This is an experimental model:** Illustrious-Lumina-v0.03 is currently in the experimental stage and max exhibit unpredictable behaviors or limitations.

**Prompt**: Due to the characteristics of the DiT architecture, detailed and specific prompts yield significantly better results. Vauge or short prompts may lead to inconsistent or unexpected outputs. We recommend providing comprehensive descriptions including subject, style, composition, and other visual elements you wish to see in the generated image.

## Testing Environment
Our setup has been tested on Ubuntu 20.04.6 LTS. While the model may work on other operating systems, we recommend using the same environment for optimal performance and to avoid compatibility issue.

## Environment Setup

### **1. Clone Git Repository**

```bash
git clone https://github.com/OnomaAI/Illustrious-Lumina.git
```

### **2. Create a Virtual Environment**

Using venv:

```bash
python3.11 -m venv lumina
source lumina/bin/acitvate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### **3. Install dependenices**

```bash
pip install -r requirement.txt
```

### **4. Install flash-attn**

```bash
pip install flash-attn --no-build-isolation
```

**Note for Windows Users:** Flash Attention installation can be challenging on Windows. Please refer to the [official Flash Attention GitHub repository](https://github.com/Dao-AILab/flash-attention) for detailed installation instructions specific to Windows.


## **Running Lumina with Gradio Demo**

- Gradio Demo
```bash
python demo-proper.py \
  --ckpt /path/to/your/ckpt \
  --res 1024 \
  --port 12123 \
```

After execution, access the specified port in your web browser to experience Lumina's image generation capabilities.


![Screenshot 2025-04-18 at 17 43 28](https://github.com/user-attachments/assets/81086f80-d8e4-4dd0-9f9c-dea7dde38abd)
