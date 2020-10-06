'''

General utility functions.

'''

import pandas as pd
import ruamel.yaml as yaml

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