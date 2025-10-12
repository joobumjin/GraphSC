import argparse
import json
import os
from tqdm import tqdm

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import hf_hub_download
from conch.open_clip_custom import create_model_from_pretrained
from preprocessing.img_preprocessing import get_image_loaders, ImageData
import pickle

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                          help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,  choices=['Donor'],                      help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')
    parser.add_argument('--multi_opt',      action="store_true",                                    help='Whether or not to optimize against mutliple objectives')
    parser.add_argument('--model_path',     type=str,                                               help='Optional path to saved model weights')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

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

    # Load the model and config files
    print(f"Loading Model")
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    data_base_dir = f"{data_dir}/full_imgs/Healthy"
    data_dirs = {"train": [f"{data_base_dir}/train_TER_imgs_0.pkl", 
                           f"{data_base_dir}/train_TER_imgs_1.pkl", 
                           f"{data_base_dir}/train_TER_imgs_2.pkl"], 
                 "valid": [f"{data_base_dir}/valid_TER_imgs_0.pkl"], 
                 "test":  [f"{data_base_dir}/test_TER_imgs_0.pkl"]}
 
    print(f"Loading Data")

    batch_size = 1

    train_loaders, val_loaders, test_loaders, out_dim = get_image_loaders(data_dirs, "TER", batch_size)

    encoding_out = f"{data_dir}/conch_embeds"
    os.makedirs(encoding_out, exist_ok=True)
    print(f"Data loaded")

    for loaders, split in zip([train_loaders, val_loaders, test_loaders], ["Train", "Val", "Test"]):
        data = []
        print(f"Embedding {split}")
        for ind, loader in enumerate(loaders):
            for batch in tqdm(loader, desc=f"   Processing Loader {ind + 1}/{len(loaders)}"):
                images = torch.stack([preprocess(to_pil_image(batch.x[ind])) for ind in range(batch_size)]).to(device)
        
                with torch.inference_mode():
                    encoding = model.encode_image(images, proj_contrast=False, normalize=False)

                data.append(Data(encoding, batch.y))

        with open(f"{encoding_out}/{split}.pkl", 'wb') as f:
            pickle.dump(data, f)

    print(f"Batch Image Embs Shape: {encoding.shape}")    

    # linear_probe = torch.nn.Linear(512, out_dim)
    # print(linear_probe(encoding).shape)
    
##############################################################################

if __name__ == '__main__':
    main()