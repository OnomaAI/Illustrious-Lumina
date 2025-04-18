# Lumina

Lumina is a powerful image generation model based on Diffusion Transformers (DiT) architecture. Originally developed by [Alpha-VLLM Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0), we've fine-tuned this model for our experimental purposes and are providing this demo to showcase its capabilities. This documentation provides instructions for setting up and running our tuned version of Lumina.

## Environment Setup

**1. Clone Git Repository**

```bash
git clone https://github.com/OnomaAI/Illustrious-Lumina.git
```

**1. Create a Virtual Environment**

Using venv:

```bash
python3.11 -m venv lumina
source lumina/bin/acitvate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Using conda:

```bash
conda create -n lumina
conda activate lumina
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**2. Install dependenices**

```bash
pip install -r requirement.txt
```

**3. Install flash-attn**
```bash
pip install flash-attn --no-build-isolation
```

## **Running Lumina with Gradio Demo**

- Gradio Demo
```bash
python demo-porper.py \
  --ckpt /path/to/your/ckpt \
  --res 1024 \
  --port 12123 \
```

After execution, access the specified port in your web browser to experience Lumina's image generation capabilities.


![Screenshot 2025-04-18 at 17 43 28](https://github.com/user-attachments/assets/81086f80-d8e4-4dd0-9f9c-dea7dde38abd)
