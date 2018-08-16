"""
Get cityscape list for training
"""
import os

import sys

def get_pair(img_dir, label_dir):
    if os.path.exists(img_dir) and os.path.exists(label_dir):
        all_images = [i for i in os.listdir(img_dir) if i.endswith('png') or i.endswith('jpg') or i.endswith('jpeg')]
        all_labels = [i for i in os.listdir(label_dir) if i.endswith('png') or i.endswith('jpg') or i.endswith('jpeg')]

        # find the pair
        # leftImg8bit/train/cologne/cologne_000000_000019_leftImg8bit.png gtFine/train/cologne/cologne_000000_000019_gtFine_labelIds.png
        pass
