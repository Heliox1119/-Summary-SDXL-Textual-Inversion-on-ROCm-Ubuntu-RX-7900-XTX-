# 🧠 Train SDXL Textual Inversion on AMD GPUs (ROCm) with Ubuntu

This guide documents a fully working setup for training **Textual Inversion embeddings** on **Stable Diffusion XL (SDXL 1.0)** using **AMD GPUs** via **ROCm**, tested on **Ubuntu 22.04.5 LTS** with an **RX 7900 XTX**.

---

## 🚀 Objective

Teach a custom token (e.g. `yourtoken`) to represent a visual concept using Textual Inversion on SDXL, **without using NVIDIA GPUs or Docker**.

---

## ⚙️ Setup Overview

- ✅ Ubuntu 22.04.5 LTS
- ✅ ROCm 6.3.2
- ✅ Python 3.10 (venv)
- ✅ AMD RX 7900 XTX
- ✅ Kohya: `bmaltais/kohya_ss` (branch: `master`)
- 🔁 Replaced `sd-scripts/` with version from `kohya-ss/sd-scripts`

⚠️ This breaks the bmaltais GUI, but everything works from CLI via `accelerate`.

---

## 📁 Dataset Structure

```plaintext
kohya_ss/
└── dataset/
    ├── images/
    │   └── 40_yourtoken/
    │       ├── img001.png
    │       ├── img001.txt
    │       └── ...
    ├── model/
    │   ├── config_textual_inversion.toml
    │   └── config_resume.toml
    ├── logs/
    │   └── prompt.txt
    └── outputs/
        └── samples/
```

- `40_` indicates **repeat count** (40x) — this naming works well and is preserved.
- `yourtoken` must match both the folder and token string in the config.
- `train_data_dir` in TOML must point to:  
  `kohya_ss/dataset/images` **(not the subfolder)**

---

## 🖋️ Captions

⚠️ **BLIP doesn’t run on ROCm/AMD**. Captions were manually created using ChatGPT, image-by-image, including NSFW ones, with neutral and clean phrasing optimized for SDXL.

➡️ Results were significantly more accurate and coherent than BLIP or WD14.

### ✏️ Examples:

```text
yourtoken, 1girl, slim, medium breasts, floral bikini, tropical background, bust portrait, looking at viewer, blonde wavy hair, tanned skin, sharp shadows, ultra detailed skin, photorealistic
yourtoken, 1girl, slim, brown sweater, turtleneck, bust portrait, looking at viewer, blonde hair, neutral background, soft studio lighting, ultra detailed skin, photorealistic
yourtoken, 1girl, nude, slim, medium breasts, upper body, facing camera, soft lighting, ultra detailed skin, photorealistic
```

---

## 🧠 Why 1024×1536 images?

- SDXL requires a minimum resolution of **1024×1024**
- 2:3 vertical ratio improves **portrait and full-body detail**
- Lower resolutions reduce latent quality and anatomical accuracy in TI

---

## ⚙️ Training Config (.toml)

TOML is a configuration file used by `sdxl_train_textual_inversion.py` to launch training. Here are two examples:

---

### ✅ Initial training: `config_textual_inversion.toml`

```toml
bucket_no_upscale = true
bucket_reso_steps = 64
cache_latents = true
caption_extension = ".txt"
clip_skip = 1
dynamo_backend = "no"
enable_bucket = true
epoch = 5
gradient_accumulation_steps = 1
gradient_checkpointing = true
huber_c = 0.1
huber_schedule = "snr"
init_word = "woman"
learning_rate = 1e-4
logging_dir = "kohya_ss/dataset/logs"
loss_type = "l2"
lr_scheduler = "constant_with_warmup"
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
lr_warmup_steps = 500
max_bucket_reso = 1536
max_data_loader_n_workers = 0
max_timestep = 1000
max_token_length = 75
max_train_steps = 5200
mem_eff_attn = true
min_bucket_reso = 256
mixed_precision = "bf16"
multires_noise_discount = 0.3
no_half_vae = true
noise_offset_type = "Original"
num_vectors_per_token = 1
optimizer_args = []
optimizer_type = "AdamW"
output_dir = "/home/user/kohya_ss/dataset/outputs"
output_name = "yourtoken_final.safetensors"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
prior_loss_weight = 0
resolution = "1024,1536"
sample_prompts = "/home/user/kohya_ss/dataset/logs/prompt.txt"
sample_dir = "/home/user/kohya_ss/dataset/outputs/samples"
sample_sampler = "k_dpm_2"
sample_every_n_steps = 1000
sample_batch_size = 1
save_every_n_epochs = 1
save_model_as = "safetensors"
save_precision = "bf16"
sdpa = true
token_string = "yourtoken"
train_batch_size = 1
train_data_dir = "kohya_ss/dataset/images"
```

---

### 🔁 Resume training: `config_resume.toml`

ℹ️ Works only if `save_state = true` was set in the initial run or passed via `--save_state`.

❗ `init_word` must be removed or training will restart from scratch.

```toml
bucket_no_upscale = true
bucket_reso_steps = 64
cache_latents = true
caption_extension = ".txt"
clip_skip = 1
dynamo_backend = "no"
enable_bucket = true
epoch = 2
gradient_accumulation_steps = 1
gradient_checkpointing = true
huber_c = 0.1
huber_schedule = "snr"
learning_rate = 1e-5
logging_dir = "kohya_ss/dataset/logs"
loss_type = "l2"
lr_scheduler = "constant_with_warmup"
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
lr_warmup_steps = 500
max_bucket_reso = 1536
max_data_loader_n_workers = 0
max_timestep = 1000
max_token_length = 75
max_train_steps = 3200
mem_eff_attn = true
min_bucket_reso = 256
mixed_precision = "bf16"
multires_noise_discount = 0.3
no_half_vae = true
noise_offset_type = "Original"
num_vectors_per_token = 1
optimizer_args = []
optimizer_type = "AdamW"
output_dir = "/home/user/kohya_ss/dataset/outputs"
output_name = "yourtoken_final_refined"
resume_from = "/home/user/kohya_ss/dataset/outputs/yourtoken_final-000002.safetensors"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
prior_loss_weight = 0
resolution = "1024,1536"
sample_prompts = "/home/user/kohya_ss/dataset/logs/prompt.txt"
sample_dir = "kohya_ss/dataset/outputs/samples"
sample_sampler = "k_dpm_2"
sample_every_n_steps = 500
sample_batch_size = 1
save_every_n_steps = 500
save_model_as = "safetensors"
save_precision = "bf16"
sdpa = true
token_string = "yourtoken"
train_batch_size = 1
train_data_dir = "kohya_ss/dataset/images"
```

---

## 🧪 Monitor Your Training

Run monitoring with:

```bash
tensorboard --logdir kohya_ss/dataset/logs
```

Intermediate samples appear in:

```
kohya_ss/dataset/outputs/samples/
```

💡 These samples are the **only reliable way** to check:
- prompt conditioning
- visual consistency
- overfitting (e.g. skin smearing, repeated faces)

---

## 🧪 Test the Final Embedding

```python
from safetensors.torch import load_file
x = load_file("yourtoken_final.safetensors")
print(x.keys())
```

If the result is `{}`, training failed silently.  
Check `save_state`, `resume_from`, and always monitor with samples.

---

## 📚 Final Advice (For Beginners Too)

- ✅ Use images ≥ 1024x1024 — ideally **1024x1536**
- ✅ Monitor all intermediate steps
- ✅ Use `1e-5` or `3e-5` LR for long training (20k+ steps)
- ❌ Remove `init_word` in resumes
- ⚠️ Don't point `train_data_dir` directly to the subfolder

---

## 📌 In Summary

- `bmaltais/kohya_ss` is only used as a dependency base
- `sd-scripts/` must be replaced by the one from `kohya-ss` for SDXL TI
- The GUI won’t work — but CLI training is fully functional

---

## 🖊️ Author

Maintained and tested by **Heliox**.  
Feel free to share, fork, or improve this setup 🙌
