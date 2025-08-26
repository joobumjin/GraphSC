import os
import numpy as np
from tifffile import imread
from PIL import Image

import faulthandler
faulthandler.enable()

from nyxus import Nyxus
nyx = Nyxus(["*ALL*"])

upsample_dir = f"/users/bjoo2/scratch/LORD-2_upsample/tifs"
seg_dir = f"/users/bjoo2/scratch/LORD-2_upsample/npys"
outline_dir = f"/users/bjoo2/scratch/LORD-2_upsample/outlines"
mask_dir = f"/users/bjoo2/scratch/LORD-2_upsample/png_masks"

image = imread(f"{upsample_dir}/{os.listdir(upsample_dir)[0]}")
mask = np.array(Image.open(f"{mask_dir}/{os.listdir(mask_dir)[0]}"))
features = nyx.featurize(image, mask)