(WIP)

# Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range Displays

<p align="center">
    <img src='https://github.com/ch-andrei/HDR-IQA-dom-adapt/blob/main/figures/HVEI2024_123_AndreiChubarau_thumbnail.png' width=400>
</p>

## Contents

This repository contains a Python/Pytorch implementation for training models with domain adaptation (DA)
between standard dynamic range (SDR) and high dynamic range (HDR) data, as  described in:

<i> Andrei Chubarau, Hyunjin Yoo, Tara Akhavan, James Clark.
Adapting Pretrained Networks for Image Quality Assessment on High Dynamic Range Displays, 2024.
<a href="https://arxiv.org/abs/2405.00670">Arxiv link</a>.
</i>
(in print at HVEI 2024; doi TBD)

We evaluate our proposed training recipe with DA for two Full-Reference IQA models, namely PieAPP [1] and VTAMIQ [2].

Full citation TBD

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

This repo follows the codebase for <a href="https://github.com/ch-andrei/VTAMIQ">VTAMIQ</a>.
We include a re-implementation of PieAPP (from <a href="https://github.com/gfxdisp/pu_pieapp">here</a>) and extend the
training procedure with domain adaptation between SDR and HDR data.

See ./run_pretrain_model.py and ./run_da.py for more details on training.

Models should be pre-trained on sRGB data (optionally, on PU-encoded SDR data) prior to running training
with domain adaptation between SDR and HDR.

## Method

We re-train PieAPP and VTAMIQ using our proposed training procedure including pre-training on sRGB data and fine-tuning
on PU-encoded data. We explore domain adaptation between SDR and HDR data to further improve performance on the
target HDR domain. We also evaluate two normalization schemes for PU-encoded inputs.

### Deep correlation alignment (DeepCORAL)

With CORrelation ALignment (CORAL) [3], we align the correlations of deep-layer activations between source and target
domain data, which improves generalization on the target domain. DeepCORAL thus exploits similarities between the two
domains and allows neural networks to learn domain-invariant but task-specific features.

CORAL loss is defined as:

$`L_{CORAL} = \frac{1}{4d^2} \|C_S - C_T\|^2_F,`$

where $`C_S`$ and $`C_T`$ are the covariance matrices of the source and the target $`d`$-dimensional feature activations,
respectively, and $`\|\ \|^2_F`$ is the Frobenius norm.

### Domain adaptation between SDR and HDR

During training, we draw a batch of SDR and a batch of HDR data, compute conventional IQA losses (e.g., MAE) between
the expected and the predicted IQA values, and further compute deep CORAL loss between the SDR and HDR batches.
The final loss function is defined as

$`L = \alpha L_{SDR} + \beta L_{HDR} + \lambda L_{CORAL}.`$

CORAL loss weight $`\lambda`$ must be adjusted based on the use-case. We recommend $`0.01 < \lambda < 0.1`$ to roughly
match the magnitude of $`L_{CORAL}`$ to the other loss terms at the end of the training.

### Normalization schemes

When training neural networks with PU-encoded data, we must normalize PU-encoded values to some known floating
point range. For models pre-trained on sRGB data, the expected range of input values follows from the range of values
used in pre-training.

For example, for VTAMIQ, we use ViT pre-trained on values in range [-1, 1]. For PieAPP, we use pre-trained weights from
VGG16 which was trained on inputs normalized by Imagenet mean and variance (roughly ranging in [-2.6, 2.6])

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

With _255_ normalization, PU units max exceed the expected min/max range, but luminance of 100 cd/m2 is mapped to
the same value as the maximum value of an sRGB image.

The benefit of _Pmax_ normalization is that it ensures that a model operates on values necessarily seen during
pre-training: it brute forces all possible PU-encoded values (and consequently luminance) to the range used
in pre-training.

The benefit of _255_ normalization is that it aligns the range of sRGB values used in pre-training with PU-encoded
SDR luminance levels, thus ensuring that similar predictions are produced for non-PU-encoded sRGB data and
for PU-encoded SDR data. This follows the intent of PU encoding, which maps 100 cd/m2 to 255 PU units for similar
reasons. PU-encoded HDR values, however, exceed the range of values from pre-training; a model may then produce
ambiguous predictions.

As described in our paper, we empirically find that _Pmax_ normalization consistently outperforms _255_, both with and
without domain adaptation. Our initial motivation was that PU-encoded values with _255_ normalization more faithfully
reproduces physical luminance levels relative to sRGB data used in pre-training, and that the degradation in performance
on HDR data can be addressed with domain adaptation, but our final results do not support this hypothesis. Simply
aligning the full range of PU values with pre-training data leads to better prediction accuracy.

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
