import argparse
import numpy as np
import os
import utils.binvox_rw
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path')
parser.add_argument('thres', type=float)
parser.add_argument('--cam')
args = parser.parse_args()

data = np.load(args.path)
alpha = data['alpha']
rgb = data['rgb']
if rgb.shape[0] < rgb.shape[-1]:
    alpha = np.transpose(alpha, (1,2,0))
    rgb = np.transpose(rgb, (1,2,3,0))
print('Shape', alpha.shape, rgb.shape)
print('Active rate', (alpha > args.thres).mean())
print('Active nums', (alpha > args.thres).sum())
xyz_min = np.array([0,0,0])
xyz_max = np.array(alpha.shape)




# volume = alpha.__ge__(args.thres)
# volume = torch.ge(torch.from_numpy(alpha), args.thres).numpy()
# volume = np.zeros((18, 20, 36))
# volume[10:22, 10:22, 10:22] = 1
volume = alpha

#ax = 0
# m_ax0 = np.zeros((volume.shape[2] - volume.shape[0] , volume.shape[1], volume.shape[2]))
# volume = np.concatenate((volume,m_ax0),axis=0)

#ax = 1
# m_ax1 = np.zeros((volume.shape[0] ,volume.shape[2] - volume.shape[1], volume.shape[2]))
# volume = np.concatenate((volume,m_ax1),axis=1)

# from IPython import embed;embed()

dims =  volume.shape
print('dims', dims)
scale = 0.7
translate = [0.0, 0.0, 0.0]
binvox_volume = utils.binvox_rw.Voxels(volume > args.thres,dims,translate,scale,'xyz')

# volume_save_path = os.path.join(args.path, 'GT-voxels-%06d.binvox' )
volume_save_path = os.path.basename(args.path) + '.binvox'
with open(volume_save_path, 'wb') as f:
    binvox_volume.write(f)

