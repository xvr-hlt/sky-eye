from PIL import Image, ImageDraw
import numpy as np

def vis_im_mask(image, mask=None, size=(1024,1024), opacity=.5, colours=('blue', 'yellow', 'orange', 'red')):
    im = Image.fromarray(image).resize(size)
    if mask is None:
        return im
    if len(mask.shape) == 2:
        mask = mask.reshape(1, *mask.shape)
    for msk, colour in zip(mask, colours):
        ma = Image.fromarray(((1-msk*opacity)*255).astype(np.uint8)).resize(size)
        highlight = Image.new('RGB', size, colour)
        im = Image.composite(im, highlight, ma).convert('RGB')
    return im