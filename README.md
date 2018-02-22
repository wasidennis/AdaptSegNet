# Learning to Adapt Structured Output Space for Semantic Segmentation

Pytorch implementation of our method for adapting semantic segmentation from the synthetic dataset (source domain) to the real dataset (target domain).

Contact: Yi-Hsuan Tsai (wasidennis at gmail dot com) and Wei-Chih Hung (whung8 at ucmerced dot edu)

## Paper
Learning to Adapt Structured Output Space for Semantic Segmentation <br />
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home)\*, [Wei-Chih Hung](https://hfslyc.github.io/)\*, [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html) and [Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/) <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 (\* indicates equal contribution).

This is the authors' code described in the above paper. Please cite our paper if you find it useful for your research.

```
@inproceedings{Tsai_CVPR_2018,
  author = {Y.-H. Tsai and W.-C. Hung and S. Schulter and K. Sohn and M.-H. Yang and M. Chandraker},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Learning to Adapt Structured Output Space for Semantic Segmentation},
  year = {2018}
}
```

## Example Results

![Alt Text](https://github.com/wasidennis/AdaptSegNet/blob/master/figure/result_git.png)

## Quantitative Reuslts


## Installation
* Install PyTorch from http://pytorch.org

* Clone this repo
```
git clone https://github.com/wasidennis/AdaptSegNet
cd AdaptSegNet
```
## Dataset
* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as the source domain, and put it in the `dataset/gta5` folder

* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as the target domain, and put it in the `dataset/cityscapes` folder

## Testing
* Download the pre-trained model and put it in the `model` folder

* 

## Training
* 

## Acknowledgment
This code borrows a lot from [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab).

## Note
The model and code are available for non-commercial research purposes only.
* 02/2018: code released




