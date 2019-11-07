from PIL import Image, ImageDraw
import numpy as np

def vis_im_mask(image, mask=None, size=(1024,1024), opacity=.5, colours=('blue', 'yellow', 'orange', 'red')):
    im = Image.fromarray(np.array(image)).resize(size)
    if mask is None:
        return im
    if len(mask.shape) == 2:
        mask = mask.reshape(1, *mask.shape)
    for msk, colour in zip(mask, colours):
        ma = Image.fromarray(((1-msk*opacity)*255).astype(np.uint8)).resize(size)
        highlight = Image.new('RGB', size, colour)
        im = Image.composite(im, highlight, ma).convert('RGB')
    return im


def vis_boxes(image, boxes=[], labels=None, colours=('green', 'yellow', 'red', 'purple')):
    im = Image.fromarray(np.array(image))
    labels = labels if labels is not None else [0 for _ in range(boxes)]
    draw = ImageDraw.Draw(im)
    h,w = im.size
    for box, cl_ix in zip(boxes, labels):
        draw.rectangle(box, outline=colours[cl_ix], width=3)
    return im