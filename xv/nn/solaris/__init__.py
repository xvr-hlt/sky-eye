import os

weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')

if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)

from . import metrics, model_io, zoo

