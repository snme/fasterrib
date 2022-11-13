import sys
import os
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import queue
import random

# torch + numerical imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from collections import defaultdict

# image preprocessing
import cv2 as cv
from skimage.filters import unsharp_mask

# clustering
from sklearn.cluster import KMeans

# linear regression
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '../')
from src.rfc_dataset import RFCDataset

from time import sleep

def in_bounds(row, col, H, W):
    return row >= 0 and col >= 0 and row < H and col < W

def safe_access(img, row, col):
    if in_bounds(row, col, img.shape[0], img.shape[1]):
        return True, img[row][col]
    else:
        return False, 0

def explore_neighbors(img, row, col, queue, tracking_bg, bg_pixel_value=-1):
    # increments to a pixel's row and col values to find the neighboring pixels
    incr = [-1, 0, 1]

    # shape (for checking bounds)
    H, W = img.shape

    # add in-bounds pixels to the queue to explore.
    for row_incr in incr:
        for col_incr in incr:
            # don't add self
            if row_incr == 0 and col_incr == 0:
                continue

            # don't add if out of bounds
            if not in_bounds(row + row_incr, col + col_incr, H, W):
                continue

            # "corner" rule—don't let a background region bleed through diagonal non-boundary region.
            is_diag = row_incr != 0 or col_incr != 0
            if tracking_bg and is_diag:
                if img[row, col + col_incr] != bg_pixel_value and  \
                   img[row + row_incr, col] != bg_pixel_value:
                    continue

            queue.put((row + row_incr, col + col_incr))

def find_groupings(img, bg_pixel_value=-1): # O(img.shape[0] * img.shape[1])

    # indices of possible pixels to search. also serves as a "visited" array for BFS.
    remaining_indices = set([(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])])

    # final groupings of indices to form each enclosed boundary.
    groupings = []
    edges = []

    is_bg_pred = lambda val : val == bg_pixel_value
    is_not_bg_pred = lambda val : val != bg_pixel_value

    while len(remaining_indices) > 0: # will repeat O(1) times

        # starting index of BFS search (start with finding overall background)
        if len(groupings) == 0 and len(edges) == 0:
            s_ind = (1, 1)
        else:
            s_ind = random.choice(tuple(remaining_indices)) # conversion is inefficient, but rare.

        # set up a predicate function so that:
        # if s_ind is the index of a non-background pixel, explore all connected pixels to find those that
        #    are also not background pixels
        # if s_ind is the index of a background pixel, explore all connected background pixels.
        s_row, s_col = s_ind
        tracking_bg = img[s_row, s_col] == bg_pixel_value
        if tracking_bg:
            pred = is_bg_pred
        else:
            pred = is_not_bg_pred

        # BFS around s_ind #
        grouping = []
        to_explore = queue.SimpleQueue()
        to_explore.put(s_ind)
        while len(remaining_indices) > 0 and not to_explore.empty():
            ind = to_explore.get_nowait()
            row, col = ind
            # only continue search and store current pixel if:
            # pixel hasn't been visited, and pixel is part of the group we're exploring.
            if ind in remaining_indices and pred(img[row, col]):
                # add to visited
                remaining_indices.remove(ind)
                # add to grouping
                grouping.append(ind)

                # add neighbors to explore (all pixels directly touching current pixel, including diagonals)
                explore_neighbors(img, row, col, to_explore, tracking_bg, bg_pixel_value)

        # if we're currently finding a group of connected pixels in a hole,
        # store in groupings. else, store in edges.
        if tracking_bg:
            groupings.append(grouping)
        else:
            edges.append(grouping)

    return groupings, edges
