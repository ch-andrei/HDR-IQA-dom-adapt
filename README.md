# Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range Displays

Readme is currently WIP

## Contents

This repository contains a Python/Pytorch implementation for training models with domain adaptation (DA) 
between standard dynamic range (SDR) and high dynamic range (HDR) data, as  described in:

<i> Andrei Chubarau, Hyunjin Yoo, Tara Akhavan, James Clark. 
Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range Displays, 2024.
<a href="https://arxiv.org/abs/2405.00670">Arxiv link</a>. (in print at HVEI 2024; doi TBD)
</i>

We evaluate our proposed training recipe with DA using two Full-Reference IQA models, namely PieAPP [1] and VTAMIQ [2].

## Paper Abstract

Conventional image quality metrics (IQMs), such as PSNR and SSIM, are designed for perceptually uniform gamma-encoded 
pixel values and cannot be directly applied to perceptually non-uniform linear high-dynamic-range (HDR) colors. 
Similarly, most of the available datasets consist of standard-dynamic-range (SDR) images collected in standard and 
possibly uncontrolled viewing conditions. Popular pre-trained neural networks are likewise intended for SDR inputs, 
restricting their direct application to HDR content. On the other hand, training HDR models from scratch is challenging 
due to limited available HDR data. In this work, we explore more effective approaches for training deep learning-based 
models for image quality assessment (IQA) on HDR data. We leverage networks pre-trained on SDR data (source domain) and 
re-target these models to HDR (target domain) with additional fine-tuning and domain adaptation. We validate our 
methods on the available HDR IQA datasets, demonstrating that models trained with our combined recipe outperform 
previous baselines, converge much quicker, and reliably generalize to HDR inputs.

## Repository Structure

This repo is based on the codebase from <a href="https://github.com/ch-andrei/VTAMIQ">VTAMIQ</a>.
We also include a re-implementation of PieAPP (from <a href="https://github.com/gfxdisp/pu_pieapp">here</a>).

### Training Details and Code

Current setup uses up to 12GB VRAM.
Batch sizes can be modified in ./training/train_config.py in dataloader_config_vtamiq and dataloader_config_pieapp for 
VTAMIQ and PieAPP, respectively.

## Method

### Deep correlation alignment (DeepCORAL)

With CORrelation ALignment (CORAL) [3], we align the correlations of deep-layer activations between source and target 
domain data, which improves generalization on the target domain. DeepCORAL thus exploits similarities between the two 
domains and allows neural networks to learn domain-invariant but task-specific features. CORAL loss is defined as:

$`L_{CORAL} = \frac{1}{4d^2} \|C_S - C_T\|^2_F`$

where $`C_S`$ and $`C_T`$ are the covariance matrices of the source and the target $`d`$-dimensional feature activations, 
respectively, and $`\|\ \|^2_F`$ is the Frobenius norm.

### Domain adaptation between SDR and HDR

During training, we draw a batch of SDR and a batch of HDR data, compute conventional IQA loss (e.g., MAE) between 
the expected and the predicted IQA values, and further compute deep CORAL loss to align  

### Normalization schemes

When training neural networks with PU-encoded data, we must normalize PU-encoded values to some known floating 
point range. For models pre-trained on sRGB data, the expected range of input values follows from the range of values 
used in pre-training.

For example, for VTAMIQ, we use ViT pre-trained on values in range [-1, 1]. For PieAPP, we use pre-trained weights from
VGG16 which was trained on inputs normalized by Imagenet mean and variance (roughly ranging in [-2.6, 2.6]) 

A regular sRGB image is thus expected to be first transformed from its range in [0, 1] into what a network expects 
as input.

PU21 encoding [4] (banding + glare variant) maps 0.005-10000 cd/m2 luminance inputs to PU units in range [Pmin, Pmax] 
(e.g., Pmin ~= 0 and Pmax ~= 600 PU units).

We test two normalization schemes, termed _Pmax_ and _255_.

With _Pmax_ normalization, we subtract Pmin from the input and then divide by (Pmax - Pmin).

With _255_ normalization, we subtract Pmin and divide by (255 - Pmin). 

Note that _255_ normalization aligns SDR luminance level of 100 cd/m2 to roughly 1.0 after normalization, 
while 100 < L < 10000 cd/m2 maps to roughly 1.0 < x < 2.3.

These values are then further transformed to the expected range for a given pre-trained network.

With _Pmax_, normalized PU units are mapped precisely to the range of values used in pre-training. 
However, these values physically represent different quantities: 
for instance, 10000 cd/m2 is mapped to the same value as the maximum value of an sRGB image.

With _255_ normalization, PU units max EXCEED the expected min/max range, but luminance of 100 cd/m2 is mapped to
the same value as the maximum value of an sRGB image.

The benefit of _Pmax_ normalization is that ensures that a model operates on values necessarily seen during 
pre-training: it brute forces all possible PU-encoded values (and consequently luminance) to the range used 
in pre-training. 

The benefit of _255_ normalization is that it aligns the range of values used in pre-training with SDR luminance level,
but HDR values may exceed this range: this emphasizes the domain expansion from SDR to HDR. 
The major downside, however, is that the model may struggle for the range of values it has not seen during training.

Empirically, we find that _Pmax_ normalization has better final model performance and _255_ does not generalize as well.
Additionally, _255_ normalization is notably subpar with domain adaptation.

## References

[1] E. Prashnani, H. Cai, Y. Mostofi, and P. Sen. PieAPP: Perceptual image-error assessment through pairwise 
preference. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition 
(CVPR), pages 1808–1817, Los Alamitos, CA, USA, jun 2018. IEEE Computer Society.

[2] A. Chubarau and J. Clark. VTAMIQ: Transformers for attention modulated image quality assessment. 
CoRR, abs/2110.01655, 2021.

[3] B. Sun and K. Saenko. Deep CORAL: Correlation alignment for deep domain adaptation. 
In Computer Vision – ECCV 2016 Workshops, pages 443–450, Cham, 2016. Springer International Publishing.

[4] R. K. Mantiuk and M. Azimi. PU21: A novel perceptually uniform encoding for adapting existing quality metrics for 
hdr. In 2021 Picture Coding Symposium (PCS), pages 1–5, 2021.
=======
# HDR-IQA-dom-adapt

More details and source code will be shared after HVEI2024 conference (late February 2024).
