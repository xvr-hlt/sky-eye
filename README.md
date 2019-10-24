# sky-eye

## Thoughts

- The problem can be decomposed into
  1. Segmenting the 'before' image, and;
  2. Classifying each pixel of identified buildings based on the 'after' image.
  
- Strategy: get MVP, break problem into sub-components (building classification etc.), see which approach works best.
- Experiments that can (probably) be run on the reduced-resolution component: broad architecture, best losses, polygonisation experiments.
  
## Set-up

### Questions

- Does polygonisation help? Specifically, at the 'before' layer, we can polygonalise to smooth the predictions and them use each polygon to predict the majority-damage. This should help address some of the pixel drift also. The polygonisation isn't differentiable, so we probably can't do this end-to-end.
    - Current theory: best approach is to polygonalise the 'pre' image and then take the majority class of all pixels – solution to satellite drift.
   
   
- What's the best set up for the loss of the building damage? There are four ordinal damage classes (not damages/slightly damaged/major damage/destroyed), as well as an implicit no-building class in the "post" heatmap.
    - How do we handle the 'no-building' case? Do we explicitly model it as an option, or just model the damage classes and mask out the loss.

- How do we handle images with no polygons? These are penalised heavily in the loss.


## Architecture

- Stuff to try: [class-context concatenation](https://github.com/PkuRainBow/OCNet.pytorch), or explicit edge categorisation in a separate CNN module (https://paperswithcode.com/paper/gated-scnn-gated-shape-cnns-for-semantic)

### Joint models vs. dual models

- Is a single model for the before/after images better, or two separate specialised models. 
  - Pros: we can concatenate/combine the filters of before/after (maybe adding deformable conv/attentional mechanisms to account for pixel drift). Intuitively, seeing the 'before' picture helps you evaluate the extent of the damage moreso than just seeing the 'after' photo.
  - Cons: harder to tune the combined model.
- What's the best way of combining?

### Experiments

- [UNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/rpu0bhol/overview) vs. [LinkNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/n1sjxbai/overview) – Linknet looks like it performs well.
    - Using efficientnet-b7: similar performance, but UNet seems to consume slightly more memory: [UNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/37g5ozbp?workspace=user-xvr-hlt) uses 90% GPU memory vs. [LinkNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/i1op16sa/system) at 82%, although UNet trains 30% faster.
    - Performance seems pretty similar.
- [FPN](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/hsogqx3z?workspace=user-xvr-hlt) uses 83% memory.
- [PSPNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/9lvz3chz/system) uses 78% memory.

Conclusion: LinkNet has best memory/performance profile, with UNet close behind (faster, more memory).

### Pretraining

- Is it better to use models pretrained for building segmentation, or roll my own using a (potentially) nicer/more specialised architecture.
  - [Pretrained models](https://solaris.readthedocs.io/en/latest/pretrained_models.html).
  - Most of the data that the models were pretrained on is also publically available – so the only other advantage is that the architectures demonstrably work.
- Not a lot of difference between [initialised from scratch](https://app.wandb.ai/xvr-hlt/sky-eye/runs/e41vlr5w) and [pretrained](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd).
- xdxd pretrained model has stability issues, `selimsef_spacenet4_densenet121unet` trains [okay](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd) (with removing first encoder layer and head due to n_channels mismatch).
- Biggest densenet works best – `selimsef_spacenet4_densenet121unet` and `selimsef_spacenet4_resnet...` don't seem to work as well.
- [EfficientNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/rpu0bhol/overview) outperforms pretrained model.

Conclusion: train from scratch.

## Loss and training

### Questions

- What combo of Dice/Focal/BCE/Jaccard is best?
    - Experiments with 4x Dice, 1x Focal inconclusive (https://app.wandb.ai/xvr-hlt/sky-eye/runs/mxknx2wr?workspace=user-xvr-hlt) vs (https://app.wandb.ai/xvr-hlt/sky-eye/runs/vppciq3g?workspace=user-xvr-hlt).
- What level of half-precision should we use?

### Half precision training

- Half precision training using AMP in the default mode works best. See: [full precision](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/nf4axyr0?workspace=user-xvr-hlt) lower batch size, [half-precision (default)](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/i1op16sa?workspace=user-xvr-hlt) 82% mem@batch8, [half-precision (alternative)](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/ycet76vn?workspace=user-xvr-hlt) 86% mem@batch8, more variance.


## Eval hacking
- Evaluation is 30% building localisation + 70% classification. 
- However, in order to score a pixel correctly for classification, we first need to have localised it correctly.
- This potentially implies we should be more recall oriented for localisation?
- Also, in around 18% of images there are no buildings at all. In these cases, the eval metric is 1. if we predict no pixels, or 0. otherwise. 
    - To evaluate: add a threshold on images to predict 
- # sky-eye

## Thoughts

- The problem can be decomposed into
  1. Segmenting the 'before' image, and;
  2. Classifying each pixel of identified buildings based on the 'after' image.
  
- Strategy: get MVP, break problem into sub-components (building classification etc.), see which approach works best.
- Experiments that can (probably) be run on the reduced-resolution component: broad architecture, best losses, polygonisation experiments.
  
## Set-up

### Questions

- Does polygonisation help? Specifically, at the 'before' layer, we can polygonalise to smooth the predictions and them use each polygon to predict the majority-damage. This should help address some of the pixel drift also. The polygonisation isn't differentiable, so we probably can't do this end-to-end.
    - Current theory: best approach is to polygonalise the 'pre' image and then take the majority class of all pixels – solution to satellite drift.
   
   
- What's the best set up for the loss of the building damage? There are four ordinal damage classes (not damages/slightly damaged/major damage/destroyed), as well as an implicit no-building class in the "post" heatmap.
    - How do we handle the 'no-building' case? Do we explicitly model it as an option, or just model the damage classes and mask out the loss.

- How do we handle images with no polygons? These are penalised heavily in the loss.


## Architecture

- Stuff to try: [class-context concatenation](https://github.com/PkuRainBow/OCNet.pytorch), or explicit edge categorisation in a separate CNN module (https://paperswithcode.com/paper/gated-scnn-gated-shape-cnns-for-semantic)

### Joint models vs. dual models

- Is a single model for the before/after images better, or two separate specialised models. 
  - Pros: we can concatenate/combine the filters of before/after (maybe adding deformable conv/attentional mechanisms to account for pixel drift). Intuitively, seeing the 'before' picture helps you evaluate the extent of the damage moreso than just seeing the 'after' photo.
  - Cons: harder to tune the combined model.
- What's the best way of combining?

### Experiments

- [UNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/rpu0bhol/overview) vs. [LinkNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/n1sjxbai/overview) – Linknet looks like it performs well.
    - Using efficientnet-b7: similar performance, but UNet seems to consume slightly more memory: [UNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/37g5ozbp?workspace=user-xvr-hlt) uses 90% GPU memory vs. [LinkNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/i1op16sa/system) at 82%, although UNet trains 30% faster.
    - Performance seems pretty similar.
- [FPN](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/hsogqx3z?workspace=user-xvr-hlt) uses 83% memory.
- [PSPNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/9lvz3chz/system) uses 78% memory.

### Pretraining

- Is it better to use models pretrained for building segmentation, or roll my own using a (potentially) nicer/more specialised architecture.
  - [Pretrained models](https://solaris.readthedocs.io/en/latest/pretrained_models.html).
  - Most of the data that the models were pretrained on is also publically available – so the only other advantage is that the architectures demonstrably work.
- Not a lot of difference between [initialised from scratch](https://app.wandb.ai/xvr-hlt/sky-eye/runs/e41vlr5w) and [pretrained](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd).
- xdxd pretrained model has stability issues, `selimsef_spacenet4_densenet121unet` trains [okay](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd) (with removing first encoder layer and head due to n_channels mismatch).
- Biggest densenet works best – `selimsef_spacenet4_densenet121unet` and `selimsef_spacenet4_resnet...` don't seem to work as well.
- [EfficientNet](https://app.wandb.ai/xvr-hlt/sky-eye-full/runs/rpu0bhol/overview) outperforms pretrained model.

Conclusion: train from scratch.

## Loss and training

### Questions

- What combo of Dice/Focal/BCE/Jaccard is best?
    - Experiments with 4x Dice, 1x Focal inconclusive (https://app.wandb.ai/xvr-hlt/sky-eye/runs/mxknx2wr?workspace=user-xvr-hlt) vs (https://app.wandb.ai/xvr-hlt/sky-eye/runs/vppciq3g?workspace=user-xvr-hlt).
- What level of half-precision should we use?


## Eval hacking
- Evaluation is 30% building localisation + 70% classification. 
- However, in order to score a pixel correctly for classification, we first need to have localised it correctly.
- This potentially implies we should be more recall oriented for localisation?
- Also, in around 18% of images there are no buildings at all. In these cases, the eval metric is 1. if we predict no pixels, or 0. otherwise. 
    - To evaluate: add a threshold on images to predict 
- Damage maps are calculated via harmonic mean. After defining a polygon, it might be beneficial to colour polygons proportional to the damage class rather than as a solid colour. This would have the same expected value of correct pixels, but decrease variance which is important if we're calculating harmonic mean.

## Post-predict optimisation

- Image -> masks:
    - Ensembling
    - Manually finding class thresholds for damage heatmap
    - [Test time augmentation](https://github.com/qubvel/ttach)
- Masks -> output
    - Polygonalisation
    
    
    
To try: existing [instance segmentation](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) approaches.

## Post-predict optimisation

- Image -> masks:
    - Ensembling
    - Manually finding class thresholds for damage heatmap
    - [Test time augmentation](https://github.com/qubvel/ttach)
- Masks -> output
    - Polygonalisation
    
    
    
To try: existing [instance segmentation](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) approaches.
