import json

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from preprocessing.img_preprocessing import get_image_loaders, HealthyData
import pickle

class Data():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
       self.x = self.x.to(device)
       self.y = self.y.to(device)

       return self
    
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

    data_base_dir = f"/users/bjoo2/data/bjoo2/qbam/data/full_imgs"
    data_dirs = {"train": [f"{data_base_dir}/train_TER_imgs_0.pkl", 
                           f"{data_base_dir}/train_TER_imgs_1.pkl", 
                           f"{data_base_dir}/train_TER_imgs_2.pkl"], 
                 "valid": f"{data_base_dir}/valid_TER_imgs_0.pkl", 
                 "test":  f"{data_base_dir}/test_TER_imgs_0.pkl"}
 
    print(f"Loading Data")

    batch_size = 1

    train_loaders, val_loaders, test_loaders, out_dim = get_image_loaders(data_dirs, "TER", batch_size)

    encoding_out = f"/users/bjoo2/data/bjoo2/qbam/data/biomedclip_embeds"
    print(f"Data loaded")

    for loaders, split in zip([train_loaders, val_loaders, test_loaders], ["Train", "Val", "Test"]):
        data = []
        for loader in loaders:
            for batch in loader:
                images = torch.stack([preprocess(to_pil_image(batch.x[ind])) for ind in range(batch_size)]).to(device)
        
                with torch.no_grad():
                    encoding = model.encode_image(images)

                data.append(Data(encoding, batch.y))

        with open(f"{encoding_out}/{split}.pkl", 'wb') as f:
            pickle.dump(data, f)
        

    # linear_probe = torch.nn.Linear(512, out_dim)
    # print(linear_probe(encoding).shape)
    
##############################################################################

if __name__ == '__main__':
    main()