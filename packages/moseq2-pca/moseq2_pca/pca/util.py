"""
Utility functions for PCA.
"""

import dask
import h5py
import click
import warnings
import numpy as np
import dask.array as da
from tqdm.auto import tqdm
import dask.array.linalg as lng
from dask.distributed import as_completed, progress
from moseq2_pca.util import (clean_frames, insert_nans, read_yaml, get_changepoints, get_rps)


def mask_data(original_data, mask, new_data):
    """
    Create a mask subregion given a boolean mask if missing data flag is used.

    Args:
    original_data (numpy.ndarray): input frames
    mask (numpy.ndarray): mask array
    new_data (numpy.ndarray): frames to use

    Returns:
    output (numpy.ndarray): masked data array
    """

    # need to make a copy otherwise distributed scheduler barfs
    output = original_data.copy()
    output[mask] = new_data[mask]

    return output

def compute_svd(dask_array, mean, rank, iters, missing_data, mask, recon_pcs, min_height, max_height, client):
    """
    Runs Singular Vector Decomposition on the inputted frames. If missing_data == True, use missing data PCA.

    Args:
    dask_array (dask 2d-array): Reshaped input data array of shape (nframes x nfeatures)
    mean (numpy.array): Means of each row in dask_array.
    rank (int): Rank of the desired thin SVD decomposition.
    iters (int): Number of SVD iterations
    missing_data (bool): Indicates whether to compute SVD with a masked array
    mask (dask 2d-array): None if missing_data == False, else mask array of shape dask_array
    recon_pcs (int): Number of PCs to reconstruct for missing data.
    min_height (int): Minimum height of mouse above the ground, used to filter reconstructed PCs.
    max_height (int): Maximum height of mouse above the ground, used to filter reconstructed PCs.
    client (dask Client): Dask client to process batches.

    Returns:
    s (numpy.array): computed singular values (eigen-values).
    v (numpy.ndarray): computed principal components (eigen-vectors).
    mean (numpy.array): updated mean of dask array if missing_data == True.
    total_var (float): total variance captured by principal components.
    """

    if not missing_data:
        # Compute PCs
        _, s, v = lng.svd_compressed((dask_array - mean).astype('float64'), rank, n_power_iter=0, compute=True)
    else:
        for iter in tqdm(range(iters), total=iters, desc='Computing Iterative PCA'):
            u, s, v = lng.svd_compressed((dask_array - mean).astype('float64'), rank, n_power_iter=0, compute=True)
            if iter < iters - 1:
                recon = u[:, :recon_pcs].dot(da.diag(s[:recon_pcs]).dot(v[:recon_pcs, :])) + mean
                recon[recon < min_height] = 0
                recon[recon > max_height] = 0
                dask_array = da.map_blocks(mask_data, dask_array, mask, recon, dtype=dask_array.dtype)
                mean = dask_array.mean(axis=0)

    # Compute total variance
    total_var = dask_array.var(ddof=1, axis=0).sum()
    futures = client.compute([s, v, mean, total_var])

    # set notebook=False because progress bar doesn't show up otherwise
    progress(futures, notebook=False)

    s, v, mean, total_var = client.gather(futures)
    return s, v, mean, total_var

def compute_explained_variance(s, nsamples, total_var):
    """
    Compute the explained variance and explained variance ratio contributed by each computed Principal Component.

    Args:
    s (numpy.array): computed singular values.
    nsamples (int): number of included samples.
    total_var (float): total variance captured by principal components.

    Returns:
    explained_variance (numpy.array): list of floats denoting the explained variance per PC.
    explained_variance_ratio (numpy.array): list of floats denoting the explained variance ratios per PC.
    """

    explained_variance = s ** 2 / (nsamples - 1)
    explained_variance_ratio = explained_variance / total_var

    return explained_variance, explained_variance_ratio

def get_timestamps(f, frames, fps=30):
    """
    Read the timestamps from a given h5 file.

    Args:
    f (read-open h5py File): open "results_00.h5" h5py.File object in read-mode
    frames (numpy.ndarray): list of 2d frames contained in opened h5 File.
    fps (int): frames per second.

    Returns:
    timestamps (numpy.array): array of timestamps for inputted frames variable
    """

    if '/timestamps' in f:
        # h5 format post v0.1.3
        timestamps = f['/timestamps'][()] / 1000.0
    elif '/metadata/timestamps' in f:
        # h5 format pre v0.1.3
        timestamps = f['/metadata/timestamps'][()] / 1000.0
    else:
        print('WARNING: timestamps were not found. Using default frame series-ordering.')
        timestamps = np.arange(frames.shape[0]) / fps

    return timestamps

def copy_metadatas_to_scores(f, f_scores, uuid):
    """
    Copy metadata from individual session extract h5 files to the PCA scores h5 file.

    Args:
    f (read-open h5py File): open "results_00.h5" h5py.File object in read-mode
    f_scores (read-open h5py File): open "pca_scores.h5" h5py.File object in read-mode
    uuid (str): uuid of inputted session h5 "f".
    """

    if '/metadata/acquisition' in f:
        # h5 format post v0.1.3
        metadata_name = f'metadata/{uuid}'
        f.copy('/metadata/acquisition', f_scores, name=metadata_name)
    elif '/metadata/extraction' in f:
        # h5 format pre v0.1.3
        metadata_name = f'metadata/{uuid}'
        f.copy('/metadata/extraction', f_scores, name=metadata_name)

def train_pca_dask(dask_array, clean_params, use_fft, rank, cluster_type, client,
                   mask=None, iters=10, recon_pcs=10, min_height=10, max_height=100):
    """
    Train PCA using dask arrays.

    Args:
    dask_array (dask array): chunked frames to train PCA
    clean_params (dict): dictionary containing filtering parameters
    use_fft (bool): indicates whether to use 2d-FFT on images.
    rank (int): Matrix rank to use
    cluster_type (str): indicates which cluster to use.
    client (Dask.Client): client object to execute dask operations
    mask (dask array): dask array of masked data if missing_data parameter==True
    iters (int): number of SVD iterations
    recon_pcs (int): number of PCs to reconstruct. (if missing_data = True)
    min_height (int): minimum mouse height from floor in (mm)
    max_height (int): maximum mouse height from floor in (mm)

    Returns:
    output_dict (dict): dictionary containing PCA training results.
    """

    missing_data = False

    # Get smallest chunk
    smallest_chunk = np.min(dask_array.chunks[0])

    # Apply the mask if it was provided via missing_data == True
    if mask is not None:
        click.echo('Found mask, applying to training data')
        missing_data = True
        dask_array[mask] = 0
        mask = mask.reshape(len(mask), -1)

    # Apply filters
    if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
        dask_array = dask_array.map_overlap(
            clean_frames, depth=(np.minimum(smallest_chunk, 20), 0, 0), boundary='reflect',
            dtype='float32', **clean_params)
    else:
        dask_array = dask_array.map_blocks(clean_frames, dtype='float32', **clean_params)

    # Optionally apply FFT to training data
    if use_fft:
        print('Using FFT...')
        dask_array = dask_array.map_blocks(
            lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
            dtype='float32')

    # Reshape the data to 2D matrix
    dask_array = dask_array.reshape(len(dask_array), -1).astype('float32')

    if cluster_type == 'slurm':
        print('Cleaning frames...')
        dask_array = client.persist(dask_array)
        if mask is not None:
            mask = client.persist(mask)

    # Compute mean to subtract from data later
    mean = dask_array.mean(axis=0)

    if cluster_type == 'slurm':
        mean = client.persist(mean)

    # todo compute reconstruction error

    print('\nComputing SVD...')

    # Pack the PCA training parameters
    svd_training_parameters = {
        'dask_array': dask_array,
        'mask': mask,
        'mean': mean,
        'rank': rank,
        'iters': iters,
        'missing_data': missing_data,
        'recon_pcs': recon_pcs,
        'min_height': min_height,
        'max_height': max_height
    }

    # Train the PCA
    s, v, mean, total_var = compute_svd(**svd_training_parameters, client=client)

    print('\nCalculation complete...')

    # correct the sign of the singular vectors
    tmp = np.argmax(np.abs(v), axis=1)
    correction = np.sign(v[np.arange(len(v)), tmp])
    v *= correction[:, None]

    # Get explained variances
    explained_variance, explained_variance_ratio = compute_explained_variance(s, len(dask_array), total_var)

    # Pack computed values into output dictionary
    output_dict = {
        'components': v,
        'singular_values': s,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'mean': mean
    }

    return output_dict


def apply_pca_local(pca_components, h5s, yamls, use_fft, clean_params,
                    save_file, chunk_size, mask_params, missing_data, fps=30,
                    h5_path='/frames', h5_mask_path='/frames_mask', verbose=False):
    """
    Project the input frame data by the transpose of the given PCs to obtain PCA Scores
    using local cluster/platform.

    Args:
    pca_components (numpy.array): array of computed Principal Components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    use_fft (bool): indicate whether to use 2D-FFT
    clean_params (dict): dictionary containing filtering options
    save_file (str): path to pca_scores filename to save
    chunk_size (int): size of chunks to process
    mask_params (dict): dictionary of masking parameters (if missing data)
    missing_data (bool): indicates whether to use mask arrays.
    fps (int): frames per second
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')
    verbose (bool): print session names as they are being loaded.

    Returns:
    """

    with h5py.File(f'{save_file}.h5', 'w') as f_scores:
        for h5, yml in tqdm(zip(h5s, yamls), total=len(h5s), desc='Computing scores'):
            # Load the file's metadata
            data = read_yaml(yml)
            uuid = data['uuid']

            if verbose:
                print('Loading', h5)

            with h5py.File(h5, 'r') as f:
                # Load frames
                frames = f[h5_path][()].astype('float32')

                if missing_data:
                    # Load masked frames
                    mask = f[h5_mask_path][()]
                    mask = np.logical_and(mask < mask_params['mask_threshold'],
                                          frames > mask_params['mask_height_threshold'])
                    frames[mask] = 0
                    mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

                # Filter the data
                frames = clean_frames(frames, **clean_params)

                # Apply FFT
                if use_fft:
                    frames = np.fft.fftshift(np.abs(np.fft.fft2(frames)), axes=(1, 2))

                # Reshape the data to 2D matrix
                frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

                timestamps = get_timestamps(f, f[h5_path], fps)
                copy_metadatas_to_scores(f, f_scores, uuid)

            # Compute scores
            scores = frames.dot(pca_components.T)

            # if we have missing data, simply fill in, repeat the score calculation,
            # then move on
            if missing_data:
                # Compute reconstructed PCs
                recon = scores.dot(pca_components)
                recon[recon < mask_params['min_height']] = 0
                recon[recon > mask_params['max_height']] = 0
                frames[mask] = recon[mask]
                scores = frames.dot(pca_components.T)

            # Insert NaNs into scores array
            scores, score_idx, _ = insert_nans(data=scores, timestamps=timestamps,
                                               fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

            # Write scores
            f_scores.create_dataset(f'scores/{uuid}', data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset(f'scores_idx/{uuid}', data=score_idx,
                                    dtype='float32', compression='gzip')


def apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                   save_file, chunk_size, mask_params, missing_data,
                   client, fps=30, h5_path='/frames', h5_mask_path='/frames_mask', verbose=False):
    """
    Project input frame data by the transpose of the given PCs to obtain PCA Scores using distributed Dask cluster.

    Args:
    pca_components (numpy.array): array of computed Principal Components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    use_fft (bool): indicate whether to use 2D-FFT
    clean_params (dict): dictionary containing filtering options
    save_file (str): path to pca_scores filename to save
    chunk_size (int): size of chunks to process
    mask_params (dict): dictionary of masking parameters (if missing data)
    missing_data (bool): indicates whether to use mask arrays.
    fps (int): frames per second
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')
    verbose (bool): print session names as they are being loaded.

    Returns:
    """

    futures = []
    uuids = []
    h5_file_pointers = []

    for h5, yml in tqdm(zip(h5s, yamls), total=len(h5s), desc='Loading Data'):
        # Load metadata
        data = read_yaml(yml)
        uuid = data['uuid']

        h5p = h5py.File(h5, mode='r')

        if verbose:
            print('Loading', h5)

        # Load data
        frames = da.from_array(h5p[h5_path], chunks=chunk_size).astype('float32')

        if missing_data:
            # Load masked data
            mask = da.from_array(h5p[h5_mask_path], chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

        # Apply filters
        if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
            frames = frames.map_overlap(
                clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
        else:
            frames = frames.map_blocks(clean_frames, dtype='float32', **clean_params)

        # Apply FFT
        if use_fft:
            frames = frames.map_blocks(
                lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
                dtype='float32')

        # Reshape data to 2D and compute scores
        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])
        scores = frames.dot(pca_components.T)

        if missing_data:
            # Reconstruct missing scores data
            recon = scores.dot(pca_components)
            recon[recon < mask_params['min_height']] = 0
            recon[recon > mask_params['max_height']] = 0
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)
            # Compute reconstructed scores
            scores = frames.dot(pca_components.T)

        futures.append(scores)
        uuids.append(uuid)
        h5_file_pointers.append(h5p)

    # pin the batch size to the number of workers (assume each worker has enough RAM for one session)
    batch_size = len(client.scheduler_info()['workers'])

    with h5py.File(f'{save_file}.h5', 'w') as f_scores:

        batch_count = 0
        batches = range(0, len(futures), batch_size)
        for i in tqdm(batches, total=len(batches), desc='Computing scores in batches'):
            # Setting up dask operations
            futures_batch = client.compute(futures[i:i+batch_size])
            uuids_batch = uuids[i:i+batch_size]
            h5s_batch = h5s[i:i+batch_size]

            keys = [tmp.key for tmp in futures_batch]
            batch_count += 1

            for future, result in as_completed(futures_batch, with_results=True):
                file_idx = keys.index(future.key)

                with h5py.File(h5s_batch[file_idx], mode='r') as f:
                    # Load timestamps
                    timestamps = get_timestamps(f, f[h5_path], fps)
                    copy_metadatas_to_scores(f, f_scores, uuids_batch[file_idx])

                # Insert NaNs in missing frames in scores array
                scores, score_idx, _ = insert_nans(data=result, timestamps=timestamps,
                                                   fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

                # Write scores
                f_scores.create_dataset(f'scores/{uuids_batch[file_idx]}', data=scores,
                                        dtype='float32', compression='gzip')
                f_scores.create_dataset(f'scores_idx/{uuids_batch[file_idx]}', data=score_idx,
                                        dtype='float32', compression='gzip')

    # once complete, close open h5 file pointers
    [h5p.close() for h5p in h5_file_pointers]

def get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                          save_file, chunk_size, mask_params, missing_data,
                          client, fps=30, pca_scores=None, progress_bar=False,
                          h5_path='/frames', h5_mask_path='/frames_mask', verbose=False):
    """
    Compute model-free changepoint block durations using random projections.

    Args:
    changepoint_params (dict): dict of changepoint parameters
    pca_components (numpy.array): computed principal components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    save_file (str): path to save changepoint files
    chunk_size (int): size of chunks to process in dask.
    mask_params (dict): dict of missing_data mask parameters.
    missing_data (bool): indicate whether to use mask_params
    client (dask Client): initialized Dask Client object
    fps (int): frames per second
    pca_scores (numpy.array): computed principal component scores
    progress_bar (bool): display progress bar
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')
    verbose (bool): print session names as they are being loaded.

    Returns:
    """

    futures = []
    uuids = []
    h5_file_pointers = []
    nrps = changepoint_params.pop('rps')

    for h5, yml in tqdm(zip(h5s, yamls), disable=progress_bar, desc='Setting up calculation', total=len(h5s)):
        # Load session metadata
        data = read_yaml(yml)
        uuid = data['uuid']

        if verbose:
            print('Loading', h5)

        h5p = h5py.File(h5, 'r')

        # Load frames
        frames = da.from_array(h5p[h5_path], chunks=chunk_size).astype('float32')

        # Load timestamps
        timestamps = get_timestamps(h5p, frames, fps)

        if missing_data and pca_scores is None:
            raise RuntimeError("Need to compute PC scores to impute missing data")
        elif missing_data:
            # Load masked data
            mask = da.from_array(h5p[h5_mask_path], chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0

            # Reshape mask
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

            # Load scores
            with h5py.File(pca_scores, 'r') as f:
                scores = f[f'scores/{uuid}']
                scores_idx = f[f'scores_idx/{uuid}']
                scores = scores[~np.isnan(scores_idx), :]

            if np.sum(frames.chunks[0]) != scores.shape[0]:
                warnings.warn(f'Chunks do not add up to scores shape in file {h5}')
                continue

            # Load scores into dask
            scores = da.from_array(scores, chunks=(frames.chunks[0], scores.shape[1]))

        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

        if missing_data:
            # Compute reconstructed PC scores
            recon = scores.dot(pca_components)
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)

        # Compute random projections
        rps = dask.delayed(get_rps, pure=False)(frames, rps=nrps, normalize=True)

        # Compute changepoints delayed job
        cps = dask.delayed(get_changepoints, pure=True)(rps, timestamps=timestamps, **changepoint_params)

        futures.append(cps)
        uuids.append(uuid)
        h5_file_pointers.append(h5p)

    # pin the batch size to the number of workers (assume each worker has enough RAM for one session)
    batch_size = len(client.scheduler_info()['workers'])

    with h5py.File(f'{save_file}.h5', 'w') as f_cps:
        f_cps.create_dataset('metadata/fps', data=fps, dtype='float32')

        batch_count = 0

        batches = range(0, len(futures), batch_size)
        for i in tqdm(batches, total=len(batches), desc='Computing changepoints in batches'):
            # Running dask job
            futures_batch = client.compute(futures[i:i+batch_size])
            uuids_batch = uuids[i:i+batch_size]
            keys = [tmp.key for tmp in futures_batch]

            batch_count += 1

            for future, result in as_completed(futures_batch, with_results=True):
                file_idx = keys.index(future.key)
                if result[0] is not None and result[1] is not None:
                    # Writing changepoints to h5 file as batches complete
                    f_cps.create_dataset(f'cps_score/{uuids_batch[file_idx]}', data=result[1],
                                         dtype='float32', compression='gzip')
                    f_cps.create_dataset(f'cps/{uuids_batch[file_idx]}', data=result[0] / fps,
                                         dtype='float32', compression='gzip')

    # once complete, close open h5 file pointers
    [h5p.close() for h5p in h5_file_pointers]
