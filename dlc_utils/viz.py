import cv2
import os
import numpy as np
import pandas as pd
from moseq2_viz.model.util import _get_transitions
from moseq2_viz.viz import usage_plot
from moseq2_viz.model.util import get_syllable_statistics
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from functools import partial
from moseq2_viz.io.video import write_frames_preview
from .analysis import get_syllable_slices
import ruamel.yaml as yaml

def plot_changepoints(cps):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax = sns.kdeplot(cps, gridsize=1000)
    plt.xlim(0, 2)
    plt.xlabel('Block duration (s)')
    plt.ylabel('P(duration)')
    fig.savefig('changepoints.png')

    return fig, ax

def plot_usages(index_filepath, labels, max_syllables):
    
    with open(index_filepath, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()
    
    df_dict = {
        'usage': [],
        'group': [],
        'syllable': []
    }
    
    for i, sess in enumerate(index_data['files']):
        
        tmp_usages, tmp_durations = get_syllable_statistics(labels[i], count='usage', max_syllable=max_syllables)
        total_usage = np.sum(list(tmp_usages.values()))
        if total_usage <= 0:
            total_usage = 1.0
        for k, v in tmp_usages.items():
            df_dict['usage'].append(v / total_usage)
            df_dict['syllable'].append(k)
            df_dict['group'].append(sess['group'])

    df = pd.DataFrame.from_dict(data=df_dict)

    fig, _ = usage_plot(df, groups=list(set(df_dict['group'])), headless=True)

def plot_training_lls(dct):

    fig, ax = plt.subplots()

    ax.plot(dct['ll'])
    plt.ylabel('Training likelihood')
    plt.xlabel('Iteration')
    fig.savefig('train_lls.png')

    return fig, ax

def plot_model_cp_diff(dct, cps):
    model_cps = []
    for _labels in dct["labels"]:
        locs = _get_transitions(_labels)[1] / 30.
        model_cps += list(np.diff(locs))
    model_cps = np.array(model_cps)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax = sns.kdeplot(cps, gridsize=1000, label="Model-free")
    ax = sns.kdeplot(model_cps, gridsize=1000, label="Model")
    plt.xlim(0, 3)
    plt.xlabel('Block duration (s)')
    plt.ylabel('P(duration)')
    fig.savefig('model_vs_cps.png')

    return fig, ax


def make_grid_movies(labels, cropped_frames, output_dir='./crowd_movies/'):
    max_syllables, nexamples = 30, 30
    filename_format = 'syllable_{:d}.mp4'
    ordering = list(range(max_syllables))

    print('getting slices')
    slice_dct = get_syllable_slices(labels, max_syllables)

    print('getting matrices')
    crowd_matrices = make_grid_matrix(labels, cropped_frames, slice_dct, nexamples)

    print(crowd_matrices.shape)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with mp.Pool() as pool:
        write_fun = partial(write_frames_preview, fps=30, depth_min=0,
                            depth_max=255, cmap='jet')
        pool.starmap(write_fun,
                     [(os.path.join(output_dir, filename_format.format(i, 'usage', ordering[i])),
                       crowd_matrix)
                      for i, crowd_matrix in enumerate(crowd_matrices) if crowd_matrix is not None])

def make_grid_matrix(labels, cropped_frames, dct, nexamples):

    crowd_matrices = []
    for example in range(0, nexamples):

        durs = np.array([i[-1] - i[0] for i in dct[example]['ts']])

        idx = np.where(durs > 0)[0]
        use_slices = [_ for i, _ in enumerate(dct[example]['ts']) if i in idx]

        max_dur = durs.max()

        rows = int(nexamples / 4)
        cols = int(nexamples / rows) + 1
        pad = 30
        crop_size=(144,144)

        crowd_matrix = np.zeros((max_dur + pad * 2, crop_size[1]*rows, crop_size[0]*cols), dtype='uint8')

        count = 0
        xpad, ypad = 0, 0

        xc0 = crop_size[0] // 2
        yc0 = crop_size[1] // 2

        for idx in use_slices:
            nframes = len(labels)
            cur_len = idx[1] - idx[0]
            use_idx = (idx[0] - pad, idx[1] + pad + (max_dur - cur_len))

            if use_idx[0] >= 0:
                frames = cropped_frames[use_idx[0]:use_idx[1]]

                for i in range(len(frames)):

                    new_frame_clip = frames[i]

                    if i >= pad and i <= pad + cur_len:
                        cv2.circle(new_frame_clip, (xc0, yc0), 3, (255, 255, 255), -1)

                    crowd_matrix[i][xpad:xpad+crop_size[0], ypad:ypad+crop_size[1]] = new_frame_clip

                ypad = ypad + crop_size[1]
                if count % cols == 0:
                    ypad = 0
                    xpad = xpad + crop_size[0]

                count += 1

                if count >= nexamples:
                    break

        crowd_matrices.append(crowd_matrix)

    return np.asarray(crowd_matrices)