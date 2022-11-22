# Neural Reflectance for Shape Recovery with Shadow Handling
> [Junxuan Li](https://junxuan-li.github.io/), and Hongdong Li. 
> CVPR 2022 (Oral Presentation).
## [Paper](https://arxiv.org/abs/2203.12909) 
We proposed a method for Photometric Stereo that
* Formulated the **shape** estimation and **material** estimation in a **self-supervised** framework which explicitly predicted **shadows** to mitigate the errors.
* Achieved the **state-of-the-art** performance in surface normal estimation and been an order of magnitude **faster** than previous methods. 
* Suitable for applications in AR/VR such as **object relighting** and **material editing**.

**Keywords**: Shape estimation,  BRDF estimation, inverse rendering, unsupervised learning, shadow estimation.

### Our object intrinsic decomposition
<p align="center">
    <img src='assets/overall_simple.jpg' width="600">
</p>

### Our object relighting
<p align="center">
    <img src='assets/bear.gif' height="180">
    <img src='assets/buddha.gif' height="180">
    <img src='assets/goblet.gif' height="180">
    <img src='assets/reading.gif' height="180">
</p>
<p align="center">
    <img src='assets/cat.gif' height="180">
    <img src='assets/cow.gif' height="180">
    <img src='assets/harvest.gif' height="180">
</p>

### Our material relighting
<p align="center">
    <img src='assets/material_editing.jpg' width="600">
</p>

If you find our code or paper useful, please cite as

    @inproceedings{li2022neural,
      title={Neural Reflectance for Shape Recovery with Shadow Handling},
      author={Li, Junxuan and Li, Hongdong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={16221--16230},
      year={2022}
    }

## Dependencies

First, make sure that all dependencies are in place. We use [anaconda](https://www.anaconda.com/) to install the dependencies.

To create an anaconda environment called `neural_reflectance`, run
```
conda env create -f environment.yml
conda activate neural_reflectance
```

## Quick Test on DiLiGenT main dataset
Our method is tested on the [DiLiGenT main dataset](https://sites.google.com/site/photometricstereodata/single?authuser=0).

To reproduce the results in the paper, we have provided [pre-computed models](https://www.dropbox.com/s/74nbauzt1h8rkb3/diligent_precomputed.zip) for quick testing. Simply run
```
bash configs/download_precomputed_models.sh
bash configs/test_precomputed_models.sh
```
The above scripts should create output folders in `runs/paper_config/diligent/`. The results are then available in `runs/paper_config/diligent/*/est_normal.png` for visualization.

## Train from Scratch 

### DiLiGenT Datasets
<p align="center">
    <img src='assets/diligent_results.png' width="800">
</p>

First, you need to download the [DiLiGenT main dataset](https://sites.google.com/site/photometricstereodata/single?authuser=0) and unzip the data to this folder `data/DiLiGenT/`.

After you have downloaded the data, run
```
python train.py --config configs/diligent/reading.yml
```
to test on each object. You can replace `configs/diligent/reading.yml ` with to other `yml` files for testing on other objects.

Alternatively, you can run
```
bash configs/train_from_scratch.sh
```
This script should run and test all the 10 objects in `data/DiLiGenT/pmsData/*` folder. And the output is stored in `runs/paper_config/diligent/*`.


### Gourd&amp;Apple dataset
<p align="center">
    <img src='assets/apple_results.jpg' width="800">
</p>

The [Gourd&amp;Apple dataset](http://vision.ucsd.edu/~nalldrin/research/cvpr08/datasets/) dataset can be downloaded in [here](http://vision.ucsd.edu/~nalldrin/research/cvpr08/datasets/). Then, unzip the data to this folder `data/Apple_Dataset/`.

After you have downloaded the data, please run

```
python train.py --config configs/apple/apple.yml 
```
to test on each object. You can replace `configs/apple/apple.yml ` with to other `yml` files for testing on other objects.

### Using Your Own Dataset

If you want to train a model on a new dataset, you can follow the python file `load_diligent.py` to write your own dataloader.

## Acknowledgement
Part of the code is based on [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) and [UPS-GCNet
](https://github.com/guanyingc/UPS-GCNet) repository.

## Citation
If you find our code or paper useful, please cite as

    @inproceedings{li2022neural,
      title={Neural Reflectance for Shape Recovery with Shadow Handling},
      author={Li, Junxuan and Li, Hongdong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={16221--16230},
      year={2022}
    }
