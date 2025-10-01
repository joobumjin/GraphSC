import json

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def main():
    # Download the model and config files

    data_dir = "/users/bjoo2/data/bjoo2/qbam/data"

    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_pytorch_model.bin",
        local_dir=f"{data_dir}/checkpoints"
    )
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_config.json",
        local_dir=f"{data_dir}/checkpoints"
    )

    # Load the model and config files
    model_name = "biomedclip_local"

    with open(f"{data_dir}/checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]


    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg


    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=f"{data_dir}/checkpoints/open_clip_pytorch_model.bin",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        encoding = model.encode_image(preprocess(torch.zeros((1, 256, 256, 1))))

    print(encoding.shape)
    
##############################################################################

if __name__ == '__main__':
    main()