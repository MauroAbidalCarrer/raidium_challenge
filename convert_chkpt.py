import os
import torch
from pathlib import Path
from transformers import SwinConfig, SwinForMaskedImageModeling

import src.configs as cfg


def convert_checkpoint(
    input_ckpt_path: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Load trainer checkpoint
    # --------------------------------------------------
    ckpt = torch.load(input_ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise ValueError("Checkpoint does not contain a 'model' key")

    state_dict = ckpt["model"]
    model_cfg = cfg.ModelConfig(**ckpt["model_cfg"])

    # --------------------------------------------------
    # 2. Strip wrapper prefixes
    #
    # Expected original prefix:
    #   model.swin.*
    #   model.encoder_norm.*
    # --------------------------------------------------
    cleaned_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "")
            cleaned_state_dict[new_key] = v

    if not cleaned_state_dict:
        raise RuntimeError("No Swin weights found after prefix stripping")

    # --------------------------------------------------
    # 3. Remove non-HF / non-backbone keys
    # --------------------------------------------------
    filtered_state_dict = {}
    for k, v in cleaned_state_dict.items():
        if k.startswith(("decoder", "mask_generator", "mask_token")):
            continue
        filtered_state_dict[k] = v

    # --------------------------------------------------
    # 4. Rebuild SwinConfig
    #
    # IMPORTANT: must exactly match pretraining
    # --------------------------------------------------
    print(model_cfg)
    img_size = 256 // (model_cfg.downscaling or 1)

    swin_cfg = SwinConfig(
        image_size=img_size,
        patch_size=model_cfg.constructor_kwargs["patch_size"],
        embed_dim=model_cfg.constructor_kwargs["embed_dim"],
        depths=[model_cfg.constructor_kwargs["per_layer_depth"]] * model_cfg.constructor_kwargs["n_layers"],
        num_heads=[2 ** (2 + i) for i in range(model_cfg.constructor_kwargs["n_layers"])],
        num_channels=1,
        window_size=6 // (model_cfg.downscaling or 1),
        use_absolute_embeddings=False,
    )

    # --------------------------------------------------
    # 5. Instantiate HF model and load weights
    # --------------------------------------------------
    model = SwinForMaskedImageModeling(swin_cfg)

    missing, unexpected = model.load_state_dict(
        filtered_state_dict,
        strict=False,
    )

    print("Missing keys (expected):")
    for k in missing:
        print("  ", k)

    print("\nUnexpected keys (ignored):")
    for k in unexpected:
        print("  ", k)

    # --------------------------------------------------
    # 6. Save as Hugging Face checkpoint
    # --------------------------------------------------
    model.save_pretrained(output_dir)

    print(f"\nHugging Face checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output HF checkpoint directory")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)
    m = SwinForMaskedImageModeling.from_pretrained(args.output)
    print(m.swin.embeddings.patch_embeddings.projection.weight.mean())

