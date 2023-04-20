# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import os
import torch
import utils.binvox_rw
import numpy as np

th = 0.2

def save_as_binvox(volume_dir, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # from IPython import embed;embed()
    volume = volume.squeeze().__ge__(th)

    #save voxel
    # volume2 = torch.ge(torch.from_numpy(volume), th).numpy() #volume numpy.ndarray (32, 32, 32)
    # (volume1==volume2).all() == True
    # from IPython import embed;embed() 
    dims = [32,32,32]
    scale = 0.7
    translate = [0.0, 0.0, 0.0]
    binvox_volume = utils.binvox_rw.Voxels(volume,dims,translate,scale,'xyz')

    volume_save_path = os.path.join(save_dir, 'GT-voxels-%06d.binvox' )
    with open(volume_save_path, 'wb') as f:
        binvox_volume.write(f)


    return cv2.imread(save_path)
