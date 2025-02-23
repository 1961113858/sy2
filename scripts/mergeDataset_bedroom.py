#!/usr/bin/env python
# coding=utf-8
import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

NUM_EACH_CLASS = 4

if __name__ == '__main__':
    logger.add("./log/log_{time}.log")
    np.random.seed(0)

    root = 'D:/cxcy2/LEGO-Net-main/sy2/outputs'
    scene_type = 'Bedroom'

    dataset1 = np.load(os.path.join(root, 'Bedroom.npy'))
    dataset2 = np.load(os.path.join(root, 'MasterBedroom.npy'))
    dataset3 = np.load(os.path.join(root, 'SecondBedroom.npy'))

    dataset = np.concatenate([dataset1, dataset2, dataset3], axis=0)
    # normalize the data in 2D
    dataset_normalize_list = []
    num_obj_list = []
    for scene_idx, scene in enumerate(dataset):
        logger.info(scene_idx)
        valid_abs_index = np.where(scene[:, -1])[0]

        # Filter out scenes with less than 6 objects
        if valid_abs_index.shape[0] < 6:
            logger.info('too few, dep')
            continue
        num_obj_list.append(valid_abs_index.shape[0])

        # Filter out scenes with size larger than 8m
        sx = np.max(scene[valid_abs_index, 3]) - np.min(scene[valid_abs_index, 3])
        sy = np.max(scene[valid_abs_index, 4]) - np.min(scene[valid_abs_index, 4])
        sobjx = np.max(scene[valid_abs_index, 6])
        sobjy = np.max(scene[valid_abs_index, 7])
        if sx > 8 or sy > 8 or sobjx > 6 or sobjy > 6:
            logger.info(f'too large: {sx}, {sy}, dep')
            continue

        # make scene center 0 in xy plane
        center = np.mean(scene[valid_abs_index, 3:5], axis=0)
        scene_normalize = copy.deepcopy(scene)
        scene_normalize[valid_abs_index, 3:5] -= center

        dataset_normalize_list.append(scene_normalize)

    dataset_normalize = np.stack(dataset_normalize_list, axis=0)
    index_permute = np.random.permutation(dataset_normalize.shape[0])
    dataset_normalize = dataset_normalize[index_permute]
    np.save(os.path.join(root, 'Bedroom_train_val.npy'), dataset_normalize)

    # Compute distribution
    distribution = np.zeros((30))
    for v in num_obj_list:
        distribution[v] += 1
    print('-' * 30)
    for i, v in enumerate(distribution):
        print('#objects: ', i, '#scenes: ', v) 