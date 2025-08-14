import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, denoise
from glob import glob
from tqdm import tqdm
from tifffile import imwrite
from pathlib import Path
from time import time

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

model = models.CellposeModel(pretrained_model = "cpsam", gpu=True)
# batch_size = 32

for folder in tqdm(range(2,3)):
    folder_name = f"LORD-{folder}"
    print(folder_name)
    data_dir = f"/users/bjoo2/scratch/{folder_name}_upsample"
    files = io.get_image_files(data_dir, '_outlines')
    print(len(files))
    for file in tqdm(files):
        images = [io.imread(f) for f in [file]]
        masks, flows, styles, = model.eval(images, batch_size=12, diameter=None, normalize={"tile_norm_blocksize": 0})    
    
        io.masks_flows_to_seg(images,
                              masks,
                              flows,
                              [file],
                              channels=None,
                              # channels=[3, 0],
                              )

        io.save_masks(images,
                      masks,
                      flows,
                      [file],
                      channels=None, # channels=[3,0],
                      png=False, # save masks as PNGs and save example image
                      tif=False, # save masks as TIFFs
                      save_txt=False, # save txt outlines for ImageJ
                      save_flows=False, # save flows as TIFFs
                      save_outlines=True, # save outlines as TIFFs
                      save_mpl=False # make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
                      )