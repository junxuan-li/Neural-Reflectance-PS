# Neural Reflectance for Shape Recovery with Shadow Handling
> [Junxuan Li](https://junxuan-li.github.io/), and Hongdong Li. 
> CVPR 2022 (Oral Presentation).
## [Paper](https://arxiv.org/abs/2203.12909) | [Video](https://www.youtube.com/watch?v=-5httWqzvNI) 
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

    @article{li2022neural,
      title={Neural Reflectance for Shape Recovery with Shadow Handling},
      author={Li, Junxuan and Li, Hongdong},
      journal={arXiv preprint arXiv:2203.12909},
      year={2022}
    }
## Codes will be coming soon!
