import math
import numpy as np
import operator
from tree_visual import *


dataSet = [['youth', 'no', 'no', 1, 'refuse'],
           ['youth', 'no', 'no', '2', 'refuse'],
           ['youth', 'yes', 'no', '2', 'agree'],
           ['youth', 'yes', 'yes', 1, 'agree'],
           ['youth', 'no', 'no', 1, 'refuse'],
           ['mid', 'no', 'no', 1, 'refuse'],
           ['mid', 'no', 'no', '2', 'refuse'],
           ['mid', 'yes', 'yes', '2', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '2', 'agree'],
           ['elder', 'yes', 'no', '2', 'agree'],
           ['elder', 'yes', 'no', '3', 'agree'],
           ['elder', 'no', 'no', 1, 'refuse'],
           ]
labels = ["age", "working", "house", "credit_situation"]


def classUnique(lst):
    return dict(zip(*np.unique(lst, return_counts=True)))
