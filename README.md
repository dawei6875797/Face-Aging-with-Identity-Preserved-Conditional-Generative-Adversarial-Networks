# Face Aging with Identity-Preserved Conditional Generative Adversarial Networks
This repo is the official open source of [Face Aging with Identity-Preserved Conditional Generative Adversarial Networks, CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf) by Zongwei Wang, Xu Tang, Weixin Luo and Shenghua Gao. 
It is implemented in tensorflow. Please follow the instructions to run the code.

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

* Running the sript (as ped2 and avenue datasets for examples) and cd into **Codes** folder at first.
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    models/pretrains/ped2
```

```shell
python inference.py  --dataset  avenue    \
                    --test_folder  ../Data/avenue/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    models/pretrains/avenue
```


## 4. Training from scratch (here we use ped2 and avenue datasets for examples)
* Set hyper-parameters
The default hyper-parameters, such as $\lambda_{init}$, $\lambda_{gd}$, $\lambda_{op}$, $\lambda_{adv}$ and the learning rate of G, as well as D, are all initialized in **training_hyper_params/hyper_params.ini**. 
* Running script (as ped2 or avenue for instances) and cd into **Codes** folder at first.
```shell
python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```
* Model selection while training
In order to do model selection, a popular way is to testing the saved models after a number of iterations or epochs (Since there are no validation set provided on above all datasets, and in order to compare the performance with other methods, we just choose the best model on testing set). Here, we can use another GPU to listen the **snapshot_dir** folder. When a new model.cpkt.xxx has arrived, then load the model and test. Finnaly, we choose the best model. Following is the script.
```shell
python inference.py  --dataset  ped2    \
                     --test_folder  ../Data/ped2/testing/frames       \
                     --gpu  1
```
Run **python train.py -h** to know more about the flag options or see the detials in **constant.py**.
```shell
Options to run the network.

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU    the device id of gpu.
  -i ITERS, --iters ITERS
                        set the number of iterations, default is 1
  -b BATCH, --batch BATCH
                        set the batch size, default is 4.
  --num_his NUM_HIS    set the time steps, default is 4.
  -d DATASET, --dataset DATASET
                        the name of dataset.
  --train_folder TRAIN_FOLDER
                        set the training folder path.
  --test_folder TEST_FOLDER
                        set the testing folder path.
  --config CONFIG      the path of training_hyper_params, default is
                        training_hyper_params/hyper_params.ini
  --snapshot_dir SNAPSHOT_DIR
                        if it is folder, then it is the directory to save
                        models, if it is a specific model.ckpt-xxx, then the
                        system will load it for testing.
  --summary_dir SUMMARY_DIR
                        the directory to save summaries.
  --psnr_dir PSNR_DIR  the directory to save psnrs results in testing.
  --evaluate EVALUATE  the evaluation metric, default is compute_auc
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
