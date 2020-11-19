'''

General utility functions.

'''

import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.model.util import get_syllable_usages, compute_behavioral_statistics

def merge_labels_with_scalars(sorted_index, model_path):
    '''
    Computes all the syllable statistics to plot, including syllable scalars.

    Parameters
    ----------
    sorted_index (dict): Sorted dict of modeled sessions
    model_fit (dict): Trained ARHMM results dict
    model_path (str): Respective path to the ARHMM model in use.
    max_sylls (int): Maximum number of syllables to include

    Returns
    -------
    df (pd.DataFrame): Dataframe containing all of the mean syllable statistics
    scalar_df (pd.DataFrame): Dataframe containing the frame-by-frame scalar and label data
    '''

    # Load scalar Dataframe to compute syllable speeds
    scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)

    df = compute_behavioral_statistics(scalar_df, count='usage',
                                       groupby=['group', 'uuid', 'SessionName', 'SubjectName'])

    return df, scalar_df

def compute_syllable_explained_variance(model, n_explained=99):
    '''
    Computes the maximum number of syllables to include that explain the given
     percentage of explained variance.

    Parameters
    ----------
    model (dict): ARHMM results dict
    n_explained (int): Percentage of explained variance

    Returns
    -------
    max_sylls (int): Number of syllables that explain the given percentage of the variance
    '''

    syllable_usages = list(get_syllable_usages(model['labels'], count='usage').values())
    cumulative_explanation = 100 * np.cumsum(syllable_usages / sum(syllable_usages))

    max_sylls = np.argwhere(cumulative_explanation >= n_explained)[0][0]
    print(f'Number of syllables explaining {n_explained}% variance: {max_sylls}')

    return max_sylls

def index_to_dataframe(index_path):
    '''
    Reads the index file into a dictionary and converts it into an editable DataFrame.

    Parameters
    ----------
    index_path (str): Path to index file

    Returns
    -------
    index_data (dict): Dict object containing all parsed index file contents
    df (pd.DataFrame): Formatted dict in DataFrame form including each session's metadata
    '''

    with open(index_path, 'r') as f:
        index_data = yaml.safe_load(f)

    files = index_data['files']
    meta = [f['metadata'] for f in files]

    meta_df = pd.DataFrame(meta)
    tmp_df = pd.DataFrame(files)

    df = pd.concat([meta_df, tmp_df], axis=1)

    return index_data, df