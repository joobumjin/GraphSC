from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import numpy as np
import glob
import pandas as pd
import itertools
from PIL import Image


class ImageData():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edge_index, self.batch = None, None #mimic geometric's Data class

    def to(self, device):
       self.x = self.x.to(device)
       self.y = self.y.to(device)

       return self
    
def get_image_loaders(data_dirs, target, batch_size, dataset="Healthy", crop=True):
    image_shape = [3, 1024, 1024] if dataset == "Healthy" else [3, 256, 256]
    def collate(data, crop_fn):
        """
        In our cases, we want to collate a list of Data instances
        """
        #is it better to prealloc numpy?
        images = torch.zeros((batch_size, *image_shape)) #prealloc
        #painful but needs to be done this way because each image has diff shape
        for ind, sample in enumerate(data): images[ind] = crop_fn(torch.Tensor(sample.x).permute(2, 0, 1)) if crop_fn is not None else torch.Tensor(sample.x).permute(2, 0, 1) #gather
        # images = torch.Tensor(np.transpose(np.array([sample.x for sample in data]), axes=(0,3,1,2)))
        if crop_fn is not None: images = crop_fn(images)

        labels = torch.Tensor(np.array([sample.y for sample in data]))[:, None]

        return ImageData(images, labels)

    train_pkls = data_dirs["train"]
    valid_pkls = data_dirs["valid"]
    test_pkls = data_dirs["test"]

    crop_fn = torchvision.transforms.RandomCrop(image_shape[1]) if crop else None
    # train_datasets = [Healthy2Dataset(base_dir, train_csv, target) for train_csv in train_csvs]
    # valid_dataset = Healthy2Dataset(base_dir, valid_csv, target)
    # test_dataset = Healthy2Dataset(base_dir, test_csv, target)
    print(f"Constructing Datasets")
    train_dset = list(itertools.chain(*[pd.read_pickle(f"{train_pkl}") for train_pkl in train_pkls]))
    val_dset = list(itertools.chain(*[pd.read_pickle(f"{valid_pkl}")for valid_pkl in valid_pkls]))
    test_dset = list(itertools.chain(*[pd.read_pickle(f"{test_pkl}") for test_pkl in test_pkls]))
    
    # num_targets = 2 if target == "Both" else 1
    sample = test_dset[0]
    num_targets = sample.y.shape[1] if len(np.array(sample.y).shape) == 2 else np.array(sample.y).size

    print(f"Constructing Dataloaders")
    train_loaders = [DataLoader(train_dset, batch_size = batch_size, collate_fn=lambda data: collate(data, crop_fn=crop_fn))]
    valid_loaders = [DataLoader(val_dset, batch_size = batch_size, collate_fn=lambda data: collate(data, crop_fn=crop_fn))]
    test_loaders = [DataLoader(test_dset, batch_size = batch_size, collate_fn=lambda data: collate(data, crop_fn=crop_fn))]
    

    return train_loaders, valid_loaders, test_loaders, num_targets



# class Healthy2Dataset(Dataset):
#   def __init__(self, base_dir, all_data_df, target, image_transform = None, target_transform=None):
#     self.target_list = ["TER", "VEGF"] if target == "Both" else [f"{target}"]

#     self.base_dir = base_dir
#     self.df = pd.read_csv(f"{base_dir}/{all_data_df}")
#     for dtype in self.target_list: 
#       self.df = self.df[self.df[dtype].notnull()]

#     self.valid_files = glob.glob(f"RGB/*/*", root_dir=base_dir)

#     self.df = self.df[self.df["file_path"].isin(self.valid_files)]

#     self.image_transform = image_transform
#     self.target_transform = target_transform

#     self.date_dict = {
#       "d3": "16-Feb-17",
#       "d4": "23-Feb-17",
#       "d5": "2-Mar-17",
#       "d6": "9-Mar-17",
#       "d7": "16-Mar-17",
#       "d8": "23-Mar-17"
#   }

#   def __len__(self):
#     return len(self.df)

#   def __getitem__(self, idx):
#     row = self.df.iloc[idx]
#     img_path = row["file_path"]
#     image = tifffile.imread(f"{self.base_dir}/{img_path}")
#     labels = [row[dtype] for dtype in self.target_list]
#     if self.image_transform: image = self.image_transform(image)
#     if self.target_transform: labels = self.target_transform(labels)
#     return HealthyData(image, labels)
#     # return image, labels
