'''

General utility functions.

'''
import pandas as pd
from pprint import pprint
from os.path import basename
from moseq2_viz.util import read_yaml
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.model.util import compute_behavioral_statistics

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

    index_data = read_yaml(index_path)

    files = index_data['files']
    meta = [f['metadata'] for f in files]

    meta_df = pd.DataFrame(meta)
    tmp_df = pd.DataFrame(files)

    df = pd.concat([meta_df, tmp_df], axis=1)

    def apply_filename(n):
        return basename(n[0])

    df['filename'] = df['path'].apply(apply_filename)

    return index_data, df

class bcolors:
    '''
    Class containing color UNICODE values used to color printed output.
    '''

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def uuid_lookup(target_uuid, uuid_dict):
    """
    Look up session infomtion with full/partial uuid. Helper function for users to look up uuid after running interactive_scalar_summary

    Parameters
    ----------
    target_uuid (str): full or partial uuid the user wants to look up.
    uuid_dict (dict): dictionary from interactive_scalar_summary widget that has all the session information 
    """    
    # prepare the dictionary for printing out the lookup result
    for uuid, info in uuid_dict.items():
         # remove unneeded field in dictionary
        for key in ['ColorDataType', 'ColorResolution', 'DepthDataType', 'DepthResolution', 'IsLittleEndian', 'NidaqChannels', 'NidaqSamplingRate']:
            del info['metadata'][key]
        # move the file path information from info['path'] to info['metadata'] for printing
        info['metadata']['h5 path'], info['metadata']['yaml path'] = info['path']
    
    for uuid, info in uuid_dict.items():
        if target_uuid in uuid:
            print('UUID:', uuid)
            pprint(info['metadata'])