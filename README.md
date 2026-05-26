# Low-Rank Residual Diffusion Models (LRDM)

This repository contains the code used to create the results presented in the paper: [Low-Rank Residual Diffusion Models
](https://openaccess.thecvf.com/content/)


![LRDM](./poster/lrdm-poster.png)

## Installation

```
conda create -n lrdm python=3.11 -y
conda activate lrdm

bash env/install.sh
```

## Datasets
Download the [GoPro](https://github.com/swz30/MPRNet/blob/main/Deblurring/Datasets/README.md), [ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN), [Raindrop](https://github.com/rui1996/DeRaindrop), [Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html), [RealBlur](https://github.com/rimchang/RealBlur) dataset to `./data`

## Train
```
python ./train.py configs/rain1400.yml
```

## Evaluation
```
python ./eval.py configs/rain1400.yml
```

### Pre-trained Models Checking
LRDM on Rain1400 checkpoints: [here](https://huggingface.co/jun0519/lrdm)

After 
```
python ./eval.py configs/rain1400.yml
```
The results should be:
```
Average PSNR:  34.40480469099684
Average SSIM:  0.9542205011891243
test end
```

## Citation
If you are using our code, model, data, or evaluation pipeline, please consider citing our work:

```
@InProceedings{Tan_2026_CVPR,
    author    = {Tan, Junfu and Yuan, Jiang},
    title     = {Low-Rank Residual Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
}
```

## Credits
This repository is based on code form:

[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch),
[RDDM](https://github.com/nachifur/RDDM/tree/main)