# Face Aging with Identity-Preserved Conditional Generative Adversarial Networks
This repo is the official open source of [Face Aging with Identity-Preserved Conditional Generative Adversarial Networks, CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf) by Zongwei Wang, Xu Tang, Weixin Luo and Shenghua Gao. 
It is implemented in tensorflow. Please follow the instructions to run the code.
![scalars_tensorboard](images/framework.JPG)

## 1. Installation
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
  tensorflow-gpu==1.4.1
  scipy==1.0.0
  opencv-python==3.3.0.10
  numpy==1.11.0
  Pillow==5.1.0
  joblib==0.11
  ops==0.4.7
```

```shell
pip install -r requirements.txt
```
* Other libraries
```code
CUDA 8.0
Cudnn 6.0
```
## 2. Download datasets
We use the Cross-Age Celebrity Dataset for training and Evaluation. More details about this dataset, please refer to (http://bcsiriuschen.github.io/CARC/). After face detection, aligning and center cropping, we split 
images into 5 age groups: 11-20, 21-30, 31-40, 41-50 and 50+.

## 3. Testing on saved models
* Download the trained models(https://1drv.ms/u/s!AlUWwwOcwDWobCqmuFyKGIt4qaA)
```shell
unzip checkpoints.zip
```

* Running the sript
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    models/pretrains/ped2
```


## 4. Training from scratch (here we use ped2 and avenue datasets for examples)
```shell
python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```



## Citation
If you find this useful, please cite our work as follows:
```code
@INPROCEEDINGS{wang2018face_aging, 
	author={Z. Wang and X. Tang, W. Luo and S. Gao}, 
	booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	title={Face Aging with Identity-Preserved Conditional Generative Adversarial Networks}, 
	year={2018}
}
```
