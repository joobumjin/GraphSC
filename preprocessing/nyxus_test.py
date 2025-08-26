import os
import numpy as np
import tifffile
import PIL

from nyxus import Nyxus
nyx = Nyxus(["*ALL*"])

upsample_dir = f"/users/bjoo2/scratch/LORD-2_upsample/tifs"
seg_dir = f"/users/bjoo2/scratch/LORD-2_upsample/npys"
outline_dir = f"/users/bjoo2/scratch/LORD-2_upsample/outlines"
mask_dir = f"/users/bjoo2/scratch/LORD-2_upsample/png_masks"

image = tifffile.imread(f"{upsample_dir}/{os.listdir(upsample_dir)[0]}")
mask = np.array(PIL.Image.open(f"{mask_dir}/{os.listdir(mask_dir)[0]}"))
features = nyx.featurize(image, mask)