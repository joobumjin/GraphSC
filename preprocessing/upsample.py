import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, denoise
from glob import glob
from tqdm.notebook import tqdm
from tifffile import imwrite
from pathlib import Path
from time import time

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

from cellpose import denoise
dn = denoise.DenoiseModel(model_type="upsample_cyto3", gpu=True, chan2=True)

for folder in tqdm(range(2,3)):
    folder_name = f"LORD-{folder}"
    start = time()
    data_dir = f"/users/bjoo2/data/bjoo2/qbam/data/full_imgs/RGB/{folder_name}"
    up_dir = f"/users/bjoo2/scratch/{folder_name}_upsample"
    files = io.get_image_files(data_dir, '_outlines')
    print(len(files))
    print(f"Time to gather files: {time() - start}")
    start = time()
    for file in files:
        fps = [file]
        image = [io.imread(fp) for fp in fps]
        print(f"Time to read: {time() - start}")
        start = time()
        imgs_up = dn.eval(image, channels=None, diameter=10., batch_size=12)
        print(f"Time to upscale: {time() - start}")
        start = time()
            
        Path(up_dir).mkdir(parents=False, exist_ok=True)
        for fp, img_up in zip(fps, imgs_up):
            imwrite(f"{up_dir}/{os.path.basename(fp)}", img_up)
        print(f"Time to write: {time() - start}")