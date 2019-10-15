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


## Architecture

### Questions

- Is it better to use models pretrained for building segmentation, or roll my own using a (potentially) nicer/more specialised architecture.
  - [Pretrained models](https://solaris.readthedocs.io/en/latest/pretrained_models.html).
  - Most of the data that the models were pretrained on is also publically available – so the only other advantage is that the architectures demonstrably work.
- Is a single model for the before/after images better, or two separate specialised models. 
  - Pros: we can concatenate/combine the filters of before/after (maybe adding deformable conv/attentional mechanisms to account for pixel drift). Intuitively, seeing the 'before' picture helps you evaluate the extent of the damage moreso than just seeing the 'after' photo.
- What's the best way of combining? 

### Experiments

- Not a lot of difference between [initialised from scratch](https://app.wandb.ai/xvr-hlt/sky-eye/runs/e41vlr5w) and [pretrained](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd).
- xdxd pretrained model has stability issues, `selimsef_spacenet4_densenet121unet` trains [okay](https://app.wandb.ai/xvr-hlt/sky-eye/runs/h0v80nxd) (with removing first encoder layer and head due to n_channels mismatch).

  
  
  

## Loss and training

### Questions

- What combo of Dice/Focal/BCE/Jaccard is best?
- What level of half-precision should we use?

