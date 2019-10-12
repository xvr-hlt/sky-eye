# sky-eye

## Thoughts

- The problem can be decomposed into
  1. Segmenting the 'before' image, and;
  2. Classifying each pixel of identified buildings based on the 'after' image.
  
## Set-up

### Questions 



## Architecture

### Questions

- Is it better to use models pretrained for building segmentation, or roll my own using a (potentially) nicer/more specialised architecture.
  - [Pretrained models](https://solaris.readthedocs.io/en/latest/pretrained_models.html).
  - Most of the data that the models were pretrained on is also publically available – so the only other advantage is that the architectures demonstrably work.
- Is a single model for the before/after images better, or two separate specialised models. 
  - Pros: we can concatenate/combine the filters of before/after (maybe adding deformable conv/attentional mechanisms to account for pixel drift). Intuitively, seeing the 'before' picture helps you evaluate the extent of the damage moreso than just seeing the 'after' photo.

## Loss and training

### Questions

- What combo of Dice/Focal/BCE/Jaccard is best?
- What level of half-precision should we use?
