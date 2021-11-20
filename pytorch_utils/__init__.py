import matplotlib

'''Allows maplolib to work headless, may need to remove this if you want to
run it interactively'''
matplotlib.use('Agg')

from .batcher import batcher
