# 3RVRF
This repo is implementation for 3D Object Reconstruction from Volumetric Radiance Fields in pytorch.
![image](/fig/method.png)

## Datasets
-  ShapeNet
-  ShapeNet unseen
-  Pix3D

[Google Drive](https://drive.google.com/drive/folders/1Q2Kos5r9WSSh-N8QO1FNaDf8sZCNB3kR?usp=share_link "here")

## Installation
```bash
cd 3RARF
conda env create -f environment.yml
```
## Run and Test
**Training**
```python
python run.py
```
**Testing**
```python
python shapenet_test.py  #Testing on the ShapeNet dataset
```
For testing on other dataset can use `shapenet_unseen_test.py` and `pix3d_test.py`
## Acknowledgement
The code base is origined from [DVGO](https://github.com/sunset1995/DirectVoxGO "DVGO").
