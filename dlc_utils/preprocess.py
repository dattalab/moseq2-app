import os
import glob
import pandas as pd
import numpy as np
import math
import tqdm
import cv2
import subprocess
import datetime
import ruamel.yaml as yaml

def get_crop_rotated(index_filepath, frames, front_tip, rear_tip):
    
    with open(index_filepath, 'r') as f:
        indexfile = yaml.safe_load(f)
    f.close()
    
    cropped_videos = []
    for vid, sess in enumerate(indexfile['files']):
        video = sess['videos'][0]
        print('Extracting', video)
    
        dx = frames[vid][front_tip]['x'] - frames[vid][rear_tip]['x']
        dy = frames[vid][front_tip]['y'] - frames[vid][rear_tip]['y']
        centroid_x = (frames[vid][front_tip]['x'] + frames[vid][rear_tip]['x'])/2
        centroid_y = (frames[vid][front_tip]['y'] + frames[vid][rear_tip]['y'])/2

        # Creating vectors of animal body and getting angle to rotate cropped image
        vectors = []
        vec_mags = []
        rotate_angles = []
        for x, y in zip(dx, dy):
            vec_mag = math.sqrt(x**2 + y**2)
            ux = x/vec_mag
            uy = y/vec_mag

            angle = math.acos(ux)
            rotate_angles.append(-angle)
            vectors.append([x, y])
            vec_mags.append(vec_mag)

        # Get Crop Coordinates
        crop_size = 144 # given max(vec_mags) rounded up to next square digit
        crop_coords = []

        for cx, cy in zip(centroid_x, centroid_y):
            # add edge conditions for exceeding image size
            cx0 = cx - (crop_size/2)
            cy0 = cy - (crop_size/2)
            cx1 = cx + (crop_size/2)
            cy1 = cy + (crop_size/2) 

            if cx0 < 0:
                cx0 = 0
            elif cx1 > 640:
                cx1 = 640
                cx0 = cx1-144

            if cy0 < 0:
                cy0 = 0
            elif cy1 > 480:
                cy1 = 480
                cy0 = cy1-144

            crop_0 = [cx0, cy0]
            crop_1 = [cx1, cy1]

            crop_coords.append((crop_0, crop_1))

        nframes = len(crop_coords)
        chunk_size = 4000
        frame_batches = list(gen_batch_sequence(nframes, chunk_size, 0))
        cropped_video = []
        fi = 0

        cropped_frames = np.zeros((nframes, crop_size, crop_size))
        win = (crop_size // 2, crop_size // 2 + 1)
        border = (crop_size, crop_size, crop_size, crop_size)

        for i, frame_range in enumerate(frame_batches):
            temp = [f + 0 for f in frame_range]
            raw_frames = read_frames(video, temp)

            for frame in tqdm.tqdm(raw_frames, desc='Crop Rotating'):
                img = np.zeros((raw_frames.shape[1], raw_frames.shape[2]))
                img[:, :] = frame[:, :]

                use_frame = cv2.copyMakeBorder(img, *border, cv2.BORDER_CONSTANT, 0)

                col_coords = np.arange(int(crop_coords[fi][0][0]),
                               int(crop_coords[fi][1][0])).astype('int16')

                row_coords = np.arange(int(crop_coords[fi][0][1]),
                               int(crop_coords[fi][1][1])).astype('int16')

                col_coords = col_coords+crop_size
                row_coords = row_coords+crop_size


                if (np.any(row_coords >= use_frame.shape[0]) or np.any(row_coords < 1)
                        or np.any(col_coords >= use_frame.shape[1]) or np.any(col_coords < 1)):
                    continue

                #rot_mat = cv2.getRotationMatrix2D((crop_size // 2, crop_size // 2),-np.rad2deg(rotate_angles[fi]), 1)

                cropped_frames[fi, :, :] = use_frame[row_coords[0]:row_coords[-1]+1, col_coords[0]:col_coords[-1]+1]
                fi += 1
                
        cropped_videos.append(cropped_frames)

    return cropped_videos

def read_frames(filename, frames, threads=6, fps=30,
                pixel_format='gray16le', frame_size=(640,480),
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .mp4/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices
    Returns:
        3d numpy array:  frames x h x w
    """
    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]
    if get_cmd:
        return command
    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    video = np.frombuffer(out, dtype='float16').reshape((len(frames), frame_size[1], frame_size[0]))
    return video

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """Generate a sequence with overlap
    """
    seq = range(offset, nframes)
    for i in range(offset, len(seq) - overlap, chunk_size - overlap):
        yield seq[i:i + chunk_size]

def hampel_filter(df, win_size=10, threshold=3, replace_val="nan"):
    df_use = df.copy()
    med = df_use.rolling(win_size).median()  # calculate rolling median
    diff = (df_use - med).abs()
    mad = diff.rolling(win_size).median()  # calculate rolling median absolute diff
    threshold = threshold * mad  # threshold
    idx = diff > threshold
    if replace_val == "nan":
        df_use[idx] = np.nan
    elif replace_val == "med":
        df_use[idx] = med[idx]

    return (df_use)

def pack_data(index_filepath, data, coords, fps_in=30, zscore=True):
    
    with open(index_filepath, 'r') as f:
        indexfile = yaml.safe_load(f)
    f.close()
    
    datas = {}
    for i, sess in enumerate(indexfile['files']):
        gauss_sig = 2 # smoothing
        interpolation = "cubic" #interpolation type for outliers
        hampel_win = 5 # hampel filter length
        hampel_threshold = 10 # hampel filter MAD * threshold
        fps = fps_in # camera fps

        use_data = data[i]
        use_data['timestamp'] = list(use_data.index)

        use_data.loc[:, "timestamp"] /= fps
        use_data.set_index(use_data["timestamp"], inplace=True)
        use_data = use_data[coords[i]]

        # use a hampel_filter to replace outliers with nans, interpolate using cubic spline interpolation
        use_data = hampel_filter(use_data, win_size=hampel_win, threshold=hampel_threshold).interpolate(interpolation)

        # use gaussian smoothing
        use_data = use_data.rolling(int(gauss_sig * 6), min_periods=0, win_type="gaussian").mean(std=gauss_sig)

        # zscore each session to remove dependencies on the exact location of the keypoint
        if zscore:
            use_data = (use_data - use_data.mean()) / use_data.std()
        datas[sess['SessionIndex']] = use_data
        
    return datas

def load_dlc_modeling_data(index_filepath):
    
    with open(index_filepath, 'r') as f:
        indexfile = yaml.safe_load(f)
    f.close()
    
    full_data = []
    all_coords = []
    
    for sess in indexfile['files']:
        h5path = sess['path'][0]
        data = pd.read_hdf(h5path)
        data.columns = data.columns.droplevel()
        full_data.append(data)

        columns = data.columns.values

        # get the coordinates
        coords = [_ for _ in columns if (_[-1] == "x") or (_[-1] == "y") or (_[-1] == "z")]
        all_coords.append(coords)
    
    return full_data, all_coords
