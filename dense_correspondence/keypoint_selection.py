"""
Compute keypoint confidence scores in order to select 'good' keypoints.
Also see visualize_keypoint_confidence_scores.ipynb
"""

import os

# numpy
import numpy as np

# torch
import torch

# key_dynam
from key_dynam.utils.utils import save_pickle


def create_scoring_function(gamma=3):
    """
    A low score is good. 0 would be perfect
    :param gamma:
    :type gamma:
    :return:
    :rtype:
    """

    def func(x):
        return np.exp(gamma * np.power((1 - x), 2))

    return func


def score_heatmap_values(heatmap_values,
                         # [N, K] where K is num keypoints values coming from 'unnormalized' heatmap, should be in [0,1] range
                         scoring_func):
    scores_vec = scoring_func(heatmap_values)
    scores_mean = np.mean(scores_vec, axis=0)  # [K,]
    sorted_idx = np.argsort(scores_mean)  # sort in low-to-high order

    return {'scores_vec': scores_vec,
            'scores_mean': scores_mean,
            'sorted_idx': sorted_idx,
            }


def select_spatially_separated_keypoints(sorted_idx,  # [N,]
                                         position_vec,  # #[N, D] where D = 2 (pixel space) or D = 3 (3D space)
                                         position_diff_threshold,  # float
                                         K,  # num descriptors to select
                                         verbose=False,
                                         ):
    position_vec = torch.Tensor(position_vec)
    N = len(sorted_idx)
    descriptor_idx = [sorted_idx[0]]
    counter = 1

    while (len(descriptor_idx) < K) and (counter < N):

        # add if distance to existing keypoints is sufficiently large
        idx = sorted_idx[counter]
        pos = position_vec[idx]
        p = position_vec[descriptor_idx]  # [counter, D]

        # take care of case where counter = 1
        if len(p.shape) == 1:
            p.unsqueeze(0)

        # now p.shape = [counter, D]
        pos_delta = p - pos
        min_delta_norm = torch.norm(pos_delta, dim=1).min()

        if verbose:
            print("\n\n")
            print("counter", counter)
            print("sorted_idx[counter]", idx)
            print("descriptor_idx", descriptor_idx)
            print("p", p)
            print("pos", pos)
            print('delta_norm', torch.norm(pos_delta, dim=1))
            print("min_delta_norm: %.2f" % (min_delta_norm))
            print("p.shape", p.shape)

        if min_delta_norm >= position_diff_threshold:
            descriptor_idx.append(idx)
            if verbose:
                print("adding ref descriptor to set")
        else:
            if verbose:
                print("skipping")

        counter += 1

    if len(descriptor_idx) < K:
        raise RuntimeError(
            "unable to find the requested number (%d) of spatially separated keypoints. Was only able to find (%d)" % (
                K, len(descriptor_idx)))
    return descriptor_idx


def score_and_select_spatially_separated_keypoints(metadata,  # metadata.p,
                                                   confidence_score_data,  # data.p file
                                                   K,  # number of reference descriptors
                                                   position_diff_threshold,  # threshold in pixels
                                                   output_dir,

                                                   visualize=False,
                                                   multi_episode_dict=None,  # needed if you want to visualize
                                                   ):
    """
    Scores keypoints according to their confidence.
    Selects the top keypoints that are "spatially separated"
    Saves out a file 'spatial_descriptors.p' that records this data
    """
    data = confidence_score_data
    heatmap_values = data['heatmap_values']
    scoring_func = create_scoring_function(gamma=3)
    score_data = score_heatmap_values(heatmap_values,
                                      scoring_func=scoring_func)
    sorted_idx = score_data['sorted_idx']

    keypoint_idx = select_spatially_separated_keypoints(sorted_idx,
                                                        metadata['indices'],
                                                        position_diff_threshold=position_diff_threshold,
                                                        K=K,
                                                        verbose=False)

    ref_descriptors = metadata['ref_descriptors'][keypoint_idx]  # [K, D]
    spatial_descriptors_data = score_data
    spatial_descriptors_data['spatial_descriptors'] = ref_descriptors
    spatial_descriptors_data['spatial_descriptors_idx'] = keypoint_idx
    save_pickle(spatial_descriptors_data, os.path.join(output_dir, 'spatial_descriptors.p'))
