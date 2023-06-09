import sys
import torch as th
import numpy as np
from guided_diffusion.image_datasets import load_data

data_dir = sys.argv[1] # just one argument

G = load_data(data_dir=data_dir, batch_size=50000, image_size=32, random_flip=False)
ref_batch, _ = next(iter(G))
ref_batch = ((ref_batch + 1.) / 2.)

ref_batch = ref_batch.permute(0, 2, 3, 1) * 255
ref_batch = ref_batch.to(th.uint8)

with open('./cifar10_reference_50000x32x32x3.npz', 'wb') as f:
    np.savez(f, ref_batch)