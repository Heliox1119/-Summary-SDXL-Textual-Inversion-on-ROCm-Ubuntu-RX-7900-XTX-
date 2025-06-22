# README - SDXL Embedding Training on Ubuntu + ROCm + RX 7900 XTX

This document is a personal technical logbook designed to serve as both a reference and a guide for anyone wishing to understand or replicate a Textual Inversion training process using the SDXL model. Textual Inversion is a technique that allows training a custom keyword (or "token") to represent a specific visual concept or identity. SDXL is one of the latest and most powerful versions of Stable Diffusion, optimized for high-resolution image generation. This guide details the full process on a system equipped with an AMD GPU using ROCm, including all successful setups, failed attempts, workarounds, and validated configurations.

---

## ✅ PART 1: WORKING SETUP (START TO FINISH)

### System Configuration

- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: AMD RX 7900 XTX
- **Drivers**: ROCm 6.3.2
- **Python**: 3.10 via virtual environment (venv)
- **Kohya**: `master` branch of [bmaltais/kohya\_ss](https://github.com/bmaltais/kohya_ss), with a full replacement of the `sd-scripts/` directory by the latest version from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

⚠️ This method likely disables the graphical user interface (UI) of `bmaltais/kohya_ss`. In this setup, the GUI is not used: the repository is used solely as a base to install all required dependencies for training and monitoring.

ℹ️ **Important note**: The `sdxl_train_textual_inversion.py` script from `bmaltais/kohya_ss` is currently broken — it **does not save the learned vectors**, either intermediate or final. Although training appears to run normally, the resulting embedding file will be empty. For this reason, the working version from `kohya-ss/sd-scripts` is used instead.

### Environment Setup

1. Install Ubuntu 22.04.5
2. Add ROCm 6.3.2 repositories and install GPU drivers
3. Create a virtual Python environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip wheel
   ```
4. Clone the base repository:
   ```bash
   git clone https://github.com/bmaltais/kohya_ss
   cd kohya_ss
   ```
5. Replace the `sd-scripts` folder with the latest from `kohya-ss/sd-scripts`

### 📁 Project Structure

ℹ️ **Folder name = token name**: In this setup, the subfolder name `40_yourtoken` under `kohya_ss/dataset/images/` matches the token string defined in the TOML config file (`token_string = "yourtoken"`). This isn’t required, but it helps keep the dataset structure clear and avoids confusion during training.

```
kohya_ss/dataset/
├── images/
│   └── 40_yourtoken/
│       ├── img001.png
│       ├── img001.txt
│       ├── ...
├── logs/
│   └── prompt.txt
├── model/
│   └── config_textual_inversion.toml
├── outputs/
│   ├── yourtoken_final_000001.safetensors
│   ├── ...
│   ├── yourtoken_final.safetensors
│   └── samples/
│       ├── 00001.png
│       ├── 00002.png
│       └── ...
```

---

## 🎨 Dataset and Captions

- **Images** resized to 1024x1536 → this resolution respects SDXL’s minimum of 1024x1024, while offering a more vertical 2:3 aspect ratio, which works better for portraits or full-body renders. Below this threshold, the quality of latent noise deteriorates and SDXL may struggle to capture fine morphological details—especially during Textual Inversion.
- **⚠️ Important**: In the TOML configuration files, set `train_data_dir = "kohya_ss/dataset/images"` (and **not** `kohya_ss/dataset/images/40_yourdataset`) to ensure that `sdxl_train_textual_inversion.py` correctly scans for subfolders containing your images and captions.
- **Folder name**: `kohya_ss/dataset/images/40_yourdataset` → the prefix `40` here indicates the repeat count. It’s not mandatory but follows a working convention that Kohya supports reliably.
- **Captions** are written manually: each caption includes the core token, figure type, style, outfit, background, and lighting cues.

ℹ️ **About Captioning**: BLIP (Bootstrapped Language-Image Pretraining) and WD14 are automated taggers that extract image descriptions using pretrained models. While helpful, they often miss context or produce inconsistent tags — and BLIP doesn’t run on AMD ROCm. To get better results, I asked ChatGPT to manually caption each image (even NSFW ones, strictly for TI training). The result: clean, consistent, and much more precise than what BLIP or WD14 would generate.

#### 📄 Caption Examples:

```text
=== kohya_ss/dataset/images/40_yourdataset/img001.txt ===
yourtoken, 1girl, slim, medium breasts, floral bikini, tropical background, bust portrait, looking at viewer, loose blonde hair, sunny outdoor scene, ultra detailed skin, photorealistic

=== kohya_ss/dataset/images/40_yourdataset/img002.txt ===
yourtoken, 1girl, brown sweater, turtleneck, bust portrait, looking at viewer, blonde hair, neutral background, soft studio lighting, ultra detailed skin, photorealistic

=== kohya_ss/dataset/images/40_yourdataset/img003.txt ===
yourtoken, 1girl, red ribbon top, close-up portrait, looking at viewer, melancholic expression, blonde hair tied back, elegant style, ultra detailed skin, photorealistic
```

## 🛠️ Key Configuration (TOML)

ℹ️ A TOML file in Kohya allows you to gather all training parameters in a clean, reusable format. It simplifies launching runs without having to type out every argument manually.

### ✅ Initial Training Config Example

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

### 🔁 Resume Training Config Example

ℹ️ This configuration only works if the initial training was launched with `save_state = true` or `--save_state` in the command line. Also make sure to **remove **`` in the resume TOML or it may restart from scratch and overwrite progress.

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

## 🧪 Sample Verification

You can use Python to check that your `.safetensors` file is valid and not empty:

```python
from safetensors.torch import load_file
embedding = load_file("yourtoken_final.safetensors")
print(embedding.keys())  # should show clip_l and clip_g keys
```

If you see an empty dictionary `{}`, then the embedding failed. Restart with fixed script or from latest checkpoint.

## 📊 Monitoring & TensorBoard

Enable monitoring during training with:

```bash
tensorboard --logdir kohya_ss/dataset/logs
```

- View live loss curves and learning rate
- Samples saved to `outputs/samples/` can be opened manually to verify morph consistency, style adaptation, and convergence

## 📌 Reference Paths Summary

- `train_data_dir` = `kohya_ss/dataset/images`
- `output_dir` = `/home/user/kohya_ss/dataset/outputs`
- `sample_dir` = `/home/user/kohya_ss/dataset/outputs/samples`
- `prompt.txt` = `/home/user/kohya_ss/dataset/logs/prompt.txt`
- `resume_from` = `/outputs/yourtoken_final-000002.safetensors`

## 🎓 Final Tips

This section is aimed at beginners too: it summarizes the key practices to ensure stable and effective training.

- Always **monitor your training** with TensorBoard and review samples
- The generated samples are the **only reliable way** to track learning progress and detect overfitting
- Use small `learning_rate` (like 3e-5) for long runs to reduce risk of style overfit
- Prefer `save_state = true` to enable recovery in case of crash
- Watch out for resolution mismatches or caption errors — they affect output quality immediately

✅ That's it — this setup produces functional SDXL embeddings on AMD ROCm!

---

🖊️ Written and maintained by **Heliox**

