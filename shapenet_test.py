import sys
import os
import numpy as np
import binvox_rw


def iou_test(gt_dir,voxel_dir):

    # load gt voxel
    with open(gt_dir, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    volum_gt = model.data
    
    # load voxel
    with open(voxel_dir, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    volum = model.data

    # alignment
    for i in range(volum.shape[0]):
        volum[i] = np.rot90(volum[i],1)
    for i in range(volum.shape[2]):
        volum[:,:,i] = np.rot90(volum[:,:,i],1)

    # from IPython import embed;embed()
    
    intersection = np.sum(np.multiply( volum, volum_gt))
    union = np.sum((volum + volum_gt) > 0.9)
    iou = intersection / union

    print(iou, voxel_dir.split('/')[-2])
    

    return iou
        

def test_batch():
    # gt_class_name = '04379243'
    # voxel_class_name = '04379243_1'
    # gt_class_name = '02691156'
    # voxel_class_name = '02691156_1'
    class_list=['02691156','02828884','02933112','03001627','03211117','03636649','04256520','04379243']
    #class_list = ['03691459',]
    all_iou = []

    for class_name in class_list:
        gt_class_name = class_name
        voxel_class_name = class_name + '_1'
        gt_basedir = '../Pix2Vox_datasets/ShapeNetVox32/' + gt_class_name
        voxel_basedir = '../myShapeNet_render/' + voxel_class_name

        #iou
        class_iou = []
        expname_list = os.listdir(voxel_basedir)
        for i, expname in enumerate(expname_list):
            gt_dir = os.path.join(gt_basedir,expname,'model.binvox')
            voxel_dir = os.path.join(voxel_basedir,expname,'model.binvox')
            try:
                iou_test_result = iou_test(gt_dir, voxel_dir)
            except ValueError :
                continue
            class_iou.append(iou_test_result)
            
        
        #Output testing results
        class_iou = np.sort(class_iou)
        mean_iou = []
        for i in range(0,len(class_iou)):
            mean_iou.append(class_iou[i])
            all_iou.append(class_iou[i])
        mean = np.mean(mean_iou, axis=0)
        # mean_iou = np.mean(class_iou, axis=0)

        # Print header
        print('============================ '+ gt_class_name +' ============================')
        print('number:',len(mean_iou))
        print('mean_iou:',mean)
        print('max_iou:',np.max(mean_iou))
        print('min_iou:',np.min(mean_iou))
    
    print('============================ TEST RESULT ============================')
    print('number:',len(all_iou))
    print('mean_iou:',np.mean(all_iou, axis=0))





if __name__ == "__main__":

    test_batch()
