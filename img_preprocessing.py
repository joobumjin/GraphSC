##################################################################################################################################

def get_image_loaders(base_dir, data_dirs, target, batch_size):
    def collate(data, crop):
        """
        In our cases, we want to collate a list of Data instances
        """
        images = torch.Tensor(np.transpose(np.array([sample.x for sample in data]), axes=(0,3,1,2)))
        labels = torch.Tensor(np.array([sample.y for sample in data]))

        return HealthyData(crop(images), labels)

    train_csv = data_dirs["train"]
    test_csv = data_dirs["test"]

    crop = torchvision.transforms.CenterCrop((1024, 1024))
    train_dataset = Healthy2Dataset(base_dir, train_csv, target)
    test_dataset = Healthy2Dataset(base_dir, test_csv, target)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn=lambda data: collate(data, crop=crop))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, collate_fn=lambda data: collate(data, crop=crop))

    return train_loader, test_loader

from torch.utils.data import Dataset, DataLoader
import tifffile
import torch
import torchvision
import numpy as np
import glob

class HealthyData():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edge_index, self.batch = None, None #mimic geometric's Data class

    def to(self, device):
       self.x = self.x.to(device)
       self.y = self.y.to(device)

       return self


class Healthy2Dataset(Dataset):
  def __init__(self, base_dir, all_data_df, target, image_transform = None, target_transform=None):
    self.target_list = ["TER", "VEGF"] if target == "Both" else [f"{target}"]

    self.base_dir = base_dir
    self.df = pd.read_csv(f"{base_dir}/{all_data_df}")
    for dtype in self.target_list: 
      self.df = self.df[self.df[dtype].notnull()]

    self.valid_files = glob.glob(f"RGB/*/*", root_dir=base_dir)

    self.df = self.df[self.df["file_path"].isin(self.valid_files)]

    self.image_transform = image_transform
    self.target_transform = target_transform

    self.date_dict = {
      "d3": "16-Feb-17",
      "d4": "23-Feb-17",
      "d5": "2-Mar-17",
      "d6": "9-Mar-17",
      "d7": "16-Mar-17",
      "d8": "23-Mar-17"
  }

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_path = row["file_path"]
    image = tifffile.imread(f"{self.base_dir}/{img_path}")
    labels = [row[dtype] for dtype in self.target_list]
    if self.image_transform: image = self.image_transform(image)
    if self.target_transform: labels = self.target_transform(labels)
    return HealthyData(image, labels)
    # return image, labels
