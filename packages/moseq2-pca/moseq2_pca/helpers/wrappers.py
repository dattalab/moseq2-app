"""
Wrapper functions for PCA.
"""

import os
import h5py
import click
import logging
import datetime
import warnings
import numpy as np
import dask.array as da
from moseq2_pca._yaml import write_yaml
from tqdm.auto import tqdm
from moseq2_pca.viz import plot_pca_results, changepoint_dist
from os.path import abspath, join, exists, splitext, basename, dirname
from moseq2_pca.helpers.data import get_pca_paths, get_pca_yaml_data, load_pcs_for_cp
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local, train_pca_dask, get_changepoints_dask
from moseq2_pca.util import recursive_find_h5s, select_strel, initialize_dask, set_dask_config, close_dask, \
            h5_to_dict, check_timestamps

def load_and_check_data(input_dir, output_dir, config_data):
    """
    Load relevant h5 and yaml files found in given input directory, then check for timestamps and warn the user if they are missing.

    Args:
    input_dir (str): input directory containing extracted h5 files to find
    output_dir (str): directory name to save pca results

    Returns:
    output_dir (str): output directory path
    h5s (list): list of found h5 files
    yamls (list): list of corresponding yaml files
    dicts (list): list of corresponding metadata.json files
    """
    # dynamically change the set_dask_config memory setting
    if config_data['cluster_type']=="local":
        set_dask_config(memory={'target': 0.85, 'spill': True, 'pause': False, 'terminate': False})
    else:
        set_dask_config()

    # Set up output directory
    output_dir = abspath(output_dir)
    if not exists(output_dir):
        os.makedirs(output_dir)

    # find directories with .dat files tchat either have incomplete or no extractions
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    check_timestamps(h5s)  # function to check whether timestamp files are found

    return output_dir, h5s, dicts, yamls

def train_pca_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to train PCA.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    """

    if config_data['missing_data'] and config_data['use_fft']:
        raise NotImplementedError("FFT and missing data not implemented yet")

    # Get training data
    output_dir, h5s, dicts, yamls = load_and_check_data(input_dir, output_dir, config_data)

    # Setting path to PCA config file
    save_file = join(output_dir, output_file)

    # Edge Case: Handling pre-existing PCA file
    if not config_data.get('overwrite_pca_train', False):
        if exists(f'{save_file}.h5'):
            click.echo(f'The file {save_file}.h5 already exists.\nWould you like to overwrite it? [y -> yes, n -> no]\n')
            ow = input()
            if ow.lower() != 'y':
                return config_data

    # Hold all frame filtering parameters in a single dict
    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'gaussfilter_time': config_data['gaussfilter_time'],
        'tailfilter': select_strel(config_data['tailfilter_shape'], config_data['tailfilter_size']),
        'medfilter_time': config_data['medfilter_time'],
        'medfilter_space': config_data['medfilter_space']
    }

    logging.basicConfig(filename=f'{output_dir}/train.log', level=logging.ERROR)

    # Load all open h5 file references
    h5ps = [h5py.File(h5, mode='r') for h5 in h5s]

    # Subset extracted frames, then read them into chunked Dask arrays
    arrays = []
    for fp in tqdm(h5ps):
        temp_extracted = fp[config_data['h5_path']]
        num_frames = int(len(temp_extracted) * config_data.get('train_on_subset', 1))
        arrays.append(
            da.from_array(
                temp_extracted, chunks=config_data['chunk_size']
            )[np.sort(np.random.choice(len(temp_extracted), num_frames, replace=False))]
        )

    # To extracted frames, then read them into chunked Dask arrays
    stacked_array = da.concatenate(arrays, axis=0)

    # Filter out depth value extreme values; Generally same values used during extraction
    stacked_array[stacked_array < config_data['min_height']] = 0
    stacked_array[stacked_array > config_data['max_height']] = 0

    config_data['data_size'] = stacked_array.nbytes

    # Initialize Dask client
    client, cluster, workers = \
        initialize_dask(cluster_type=config_data['cluster_type'],
                        nworkers=config_data['nworkers'],
                        cores=config_data['cores'],
                        processes=config_data['processes'],
                        memory=config_data['memory'],
                        wall_time=config_data['wall_time'],
                        queue=config_data['queue'],
                        timeout=config_data['timeout'],
                        cache_path=config_data['dask_cache_path'],
                        local_processes=config_data['local_processes'],
                        dashboard_port=config_data['dask_port'],
                        data_size=config_data['data_size'])

    click.echo(f'Processing {len(stacked_array)} total frames')

    # Optionally read corresponding frame masks if for recording sessions that contain inscopix,
    # photometry, or ephys cables. These sessions in particular include frame-by-frame masks
    # to explicitly tell PCA where the mouse is, removing any noise or obstructions.
    # Note: timestamps for all files are required in order for this operation to work.
    if config_data['missing_data'] or config_data.get('cable_filter_iters', 0) > 1:
        config_data['missing_data'] = True # in case cable filter iterations > 1
        mask_dsets = [h5py.File(h5, mode='r')[config_data['h5_mask_path']] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=config_data['chunk_size']) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < config_data['mask_threshold'],
                                            stacked_array > config_data['mask_height_threshold'])
        click.echo('Loaded mask for missing data')

    else:
        stacked_array_mask = None

    params = config_data
    params['start_time'] = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
    params['inputs'] = h5s

    # Update PCA config yaml file
    config_store = f'{save_file}.yaml'
    write_yaml(params, config_store)

    # Compute Principal Components
    try:
        output_dict = \
            train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                           clean_params=clean_params, use_fft=config_data['use_fft'],
                           rank=config_data['rank'], cluster_type=config_data['cluster_type'],
                           min_height=config_data['min_height'],
                           max_height=config_data['max_height'], client=client,
                           iters=config_data['missing_data_iters'],
                           recon_pcs=config_data['recon_pcs'])
    except Exception as e:
        # Clearing all data from Dask client in case of interrupted PCA
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('Training interrupted. Closing Dask Client. You may find logs of the error here:')
        click.echo('---- ', join(output_dir, 'train.log'))
    finally:
        # After Success or failure: Shutting down Dask client and clearing any residual data
        close_dask(client, cluster, config_data['timeout'])

        # close all open h5 files
        [fp.close() for fp in h5ps]

    try:
        # Plotting training results
        plot_pca_results(output_dict, save_file, output_dir)

        # Saving PCA to h5 file
        with h5py.File(f'{save_file}.h5', 'w') as f:
            for k, v in output_dict.items():
                f.create_dataset(k, data=v, compression='gzip', dtype='float32')

        config_data['pca_file'] = f'{save_file}.h5'
    except:
        click.echo('Could not save PCA since the training was interrupted.')
        pass

    return config_data

def apply_pca_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to obtain PCA Scores.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    success (bool): flag to indicate whether the PCA scores were computed successfully
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set up data
    output_dir, h5s, dicts, yamls = load_and_check_data(input_dir, output_dir, config_data)

    # Set path to PCA Scores file
    save_file = join(output_dir, output_file)

    # Handling pre-existing PCA file
    # no intended pca overwrite
    if not config_data.get('overwrite_pca_apply', False):
        if exists(f'{save_file}.h5'):
            click.echo(
                f'The file {save_file}.h5 already exists.\nWould you like to overwrite it? [y -> yes, n -> no]\n')
            ow = input()
            if ow.lower() != 'y':
                return config_data, False

    # Get path to trained PCA file to load PCs from
    config_data, pca_file, pca_file_scores = get_pca_paths(config_data, output_dir)

    print('Loading PCs from', pca_file)
    with h5py.File(config_data['pca_file'], 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # Get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = splitext(pca_file)[0] + '.yaml'

    # Get filtering parameters and optional PCA reconstruction parameters (if missing_data == True)
    use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(pca_yaml)

    with warnings.catch_warnings():
        # Compute PCA Scores locally (without dask)
        if config_data['cluster_type'] == 'nodask':
            apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                            use_fft=use_fft, clean_params=clean_params,
                            save_file=save_file, chunk_size=config_data['chunk_size'],
                            mask_params=mask_params, fps=config_data['fps'],
                            missing_data=missing_data, h5_path=config_data['h5_path'],
                            h5_mask_path=config_data['h5_mask_path'], verbose=config_data['verbose'])

        else:
            # Initialize Dask client
            client, cluster, workers = \
                initialize_dask(cluster_type=config_data['cluster_type'],
                                nworkers=config_data['nworkers'],
                                cores=config_data['cores'],
                                processes=config_data['processes'],
                                memory=config_data['memory'],
                                wall_time=config_data['wall_time'],
                                queue=config_data['queue'],
                                timeout=config_data['timeout'],
                                cache_path=config_data['dask_cache_path'],
                                dashboard_port=config_data['dask_port'],
                                data_size=config_data.get('data_size', None))

            logging.basicConfig(filename=f'{output_dir}/scores.log', level=logging.ERROR)

            # Compute PCA Scores
            try:
                apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                               use_fft=use_fft, clean_params=clean_params,
                               save_file=save_file, chunk_size=config_data['chunk_size'],
                               fps=config_data['fps'], client=client, missing_data=missing_data,
                               mask_params=mask_params, h5_path=config_data['h5_path'],
                               h5_mask_path=config_data['h5_mask_path'], verbose=config_data['verbose'])
            except:
                # Clearing all data from Dask client in case of interrupted PCA
                click.echo('Operation interrupted. Closing Dask Client.')
            finally:
                # After Success or failure: Shutting down Dask client and clearing any residual data
                close_dask(client, cluster, config_data['timeout'])

    config_data['pca_file_scores'] = save_file + '.h5'
    return config_data, True

def compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to compute model-free Changepoints.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Get loaded h5s and yamls
    output_dir, h5s, dicts, yamls = load_and_check_data(input_dir, output_dir, config_data)

    # Set path to changepoints
    save_file = join(output_dir, output_file)

    # Get paths to PCA, PCA Scores file
    config_data, pca_file, pca_file_scores = get_pca_paths(config_data, output_dir)

    # Load Principal components, set up changepoint parameter dict, and optionally load reconstructed PCs.
    pca_components, changepoint_params, missing_data, mask_params = load_pcs_for_cp(pca_file, config_data)

    # Initialize Dask client
    client, cluster, workers = \
        initialize_dask(cluster_type=config_data['cluster_type'],
                        nworkers=config_data['nworkers'],
                        cores=config_data['cores'],
                        processes=config_data['processes'],
                        memory=config_data['memory'],
                        wall_time=config_data['wall_time'],
                        queue=config_data['queue'],
                        timeout=config_data['timeout'],
                        cache_path=config_data['dask_cache_path'],
                        dashboard_port=config_data['dask_port'],
                        data_size=config_data.get('data_size', None))

    logging.basicConfig(filename=f'{output_dir}/changepoints.log', level=logging.ERROR)

    # Compute Changepoints
    try:
        get_changepoints_dask(pca_components=pca_components, pca_scores=pca_file_scores,
                              h5s=h5s, yamls=yamls, changepoint_params=changepoint_params,
                              save_file=save_file, chunk_size=config_data['chunk_size'],
                              fps=config_data['fps'], client=client, missing_data=missing_data,
                              mask_params=mask_params, h5_path=config_data['h5_path'],
                              h5_mask_path=config_data['h5_mask_path'], verbose=config_data['verbose'])
    except:
        click.echo('Operation interrupted. Closing Dask Client.')
        close_dask(client, cluster, config_data['timeout'])

    # After Success: Shutting down Dask client and clearing any residual data
    close_dask(client, cluster, config_data['timeout'])

    # Read Changepoints from saved file
    import numpy as np
    with h5py.File(f'{save_file}.h5', 'r') as f:
        cps = h5_to_dict(f, 'cps')

    # add change point path to config file
    config_data['changepoint_file'] = save_file + '.h5'
    # Plot and save Changepoint PDF histogram
    block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
    out = changepoint_dist(block_durs, headless=True)
    if out:
        fig, _ = out
        fig.savefig(f'{save_file}_dist.png')
        fig.savefig(f'{save_file}_dist.pdf')
        fig.close('all')

    return config_data

def clip_scores_wrapper(pca_file, clip_samples, from_end=False):
    """
    Clip PCA scores from the beginning or end overwriting the original results.

    Args:
    pca_file (str): Path to PCA scores.
    clip_samples (int): number of samples to clip from beginning or end
    from_end (bool): if true clip from end rather than beginning
    """

    with h5py.File(pca_file, 'r') as f:
        store_dir = dirname(pca_file)
        base_filename = splitext(basename(pca_file))[0]
        new_filename = join(store_dir, f'{base_filename}_clip.h5')

        with h5py.File(new_filename, 'w') as f2:
            f.copy('/metadata', f2)
            for key in tqdm(f['/scores'].keys(), desc='Copying data'):
                if from_end:
                    f2[f'/scores/{key}'] = f[f'/scores/{key}'][:-clip_samples]
                    f2[f'/scores_idx/{key}'] = f[f'/scores_idx/{key}'][:-clip_samples]
                else:
                    f2[f'/scores/{key}'] = f[f'/scores/{key}'][clip_samples:]
                    f2[f'/scores_idx/{key}'] = f[f'/scores_idx/{key}'][clip_samples:]
