import os
import glob
from flask import request, jsonify
from app import app, data_path, data_config#, mongo
import logger
import subprocess
from ast import literal_eval
from moseq2_extract.gui import *
import pickle


global config_dict

@app.route('/get-local-dir', methods=['GET'])
def check_local_data_dir():
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd = cwd+data_path

        if len(os.listdir(cwd)) == 0:
            return jsonify({'ok': False, 'message': 'No data files found!'}), 400
        else:
            files = [f.replace(cwd, '') for f in glob.glob(cwd + "**/*.*", recursive=True)]
            return jsonify({'ok': True, 'message': 'Successfully found files.', 'files': files}), 200
    else:
        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/generate-config', methods=['GET', 'POST', 'PATCH'])
def extract_generate_conf():
    cwd = os.getcwd()
    cwd1 = cwd + data_path
    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd
        os.system(cd_cmd)
        data = request.form.to_dict()

        if data == {}:
            os.system(f'moseq2-extract generate-config -o {cwd1}/config.yaml')
            if os.path.exists(cwd1+'config.yaml'):
                return jsonify({'ok': True, 'message': cwd1+'/config.yaml'}), 200

        else:
            for k, v in data.items():
                if ',' in v:
                    data[k] = literal_eval('('+v+')')
                elif v == 'on':
                    data[k] = True
                elif v == 'off':
                    data[k] = False
                elif v.isdigit() or '-' in v:
                    data[k] = int(v)

            data['flip_classifier'] = ''
            data['config_file'] = cwd1+'config.yaml'

            generate_config_command(cwd1+'config.yaml', data)
            with open(cwd+data_config+'param_data.pkl', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #os.system(f'moseq2-extract generate-config -o {cwd1}config.yaml -g {data}')
            if os.path.exists(cwd1+'config.yaml'):
                return jsonify({'ok': True, 'message': cwd1+'/config.yaml'}), 200

        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/gen-gridsearch-config', methods=['POST'])
def batch_gen_gridsearch_conf(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/extract-raw', methods=['POST'])
def extract_raw():

    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'POST':
        #start cli command with default params unless get dict is not empty
        cd_cmd = 'cd '+cwd
        os.system(cd_cmd)

        input_files = request.form.to_dict()

        if input_files == {}:
            # run extract with default cmd
            filepath = cwd1+'depth.dat'
            configfile = 'config.yaml'

            os.system(f'moseq2-extract extract {filepath}')

            if os.path.exists(cwd1+'proc/'):
                cp_cmd = f'cp {cwd1}proc/results_00.mp4 {cwd}/modules/app/static/img/results_00.mp4'
                os.system(cp_cmd)
                return jsonify({'ok': True, 'message': 'Extraction successful'}), 200
            else:
                return jsonify({'ok': True, 'message': 'Extraction not successful'}), 400
        else:
            with open(cwd+data_config+'param_data.pkl', 'rb') as handle:
                data = pickle.load(handle)

            # Update to support multiple files
            extract_command(cwd1+input_files['depth-file'], data['crop_size'], data['bg_roi_dilate'], data['bg_roi_shape'], data['bg_roi_index'], data['bg_roi_weights'],
                            data['bg_roi_depth_range'], data['bg_roi_gradient_filter'], data['bg_roi_gradient_threshold'], data['bg_roi_gradient_kernel'], data['bg_roi_fill_holes'],
                            data['min_height'], data['max_height'], data['fps'], data['flip_classifier'], data['flip_classifier_smoothing'],data['use_tracking_model'], 
                            data['tracking_model_ll_threshold'], data['tracking_model_mask_threshold'], data['tracking_model_ll_clip'], data['tracking_model_segment'],
                            data['tracking_model_init'], data['cable_filter_iters'], data['cable_filter_shape'], data['cable_filter_size'], data['tail_filter_iters'],
                            data['tail_filter_size'], data['tail_filter_shape'], [data['spatial_filter_size']], [data['temporal_filter_size']], data['chunk_size'], data['chunk_overlap'],
                            cwd1+'proc/', data['write_movie'], data['use_plane_bground'], 'uint8', data['centroid_hampel_span'], data['centroid_hampel_sig'],
                            data['angle_hampel_span'], data['angle_hampel_sig'], data['model_smoothing_clips'], data['frame_trim'], data['config_file'], 
                            data['compress'], data['chunk_size'], data['num_compress_threads'])

            if os.path.exists(cwd1+'proc/'):
                os.system(f'cp {cwd1}proc/bground.tiff {cwd}/modules/app/static/output_imgs/bground.png')
                os.system(f'cp {cwd1}proc/first_frame.tiff {cwd}/modules/app/static/output_imgs/first_frame.png')
                os.system(f'cp {cwd1}proc/roi_00.tiff {cwd}/modules/app/static/output_imgs/roi_00.png')
                os.system(f'cp {cwd1}proc/results_00.mp4 {cwd}/modules/app/static/output_imgs/results_00.mp4')

                return jsonify({'ok': True, 'message': 'Extraction successful'}), 200
        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/extract-batch', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def batch_extract_raw(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/aggregate-extract-results', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def batch_agg_extract_results(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/download-flip-file', methods=['GET'])
def extract_download_flip(path=None):
    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'GET':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd
        os.system(cd_cmd)

        filepath = cwd1 + 'depth.dat'

        query = request.args

        if query is not None:
            print(query)
            index = query["flip-id"]
            print(index)

            os.system(f'moseq2-extract download-flip-file -s {index} --output-dir {cwd1}/')

            return jsonify({'ok': True, 'message': "flip file downloaded!"}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/find-roi', methods=['GET', 'POST', 'PATCH'])
def extract_find_roi(path=None):
    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'GET':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd
        os.system(cd_cmd)

        filepath = cwd1 + 'depth.dat'

        query = request.args

        if query == {}:
            os.system(f'moseq2-extract find-roi {filepath}')
            return jsonify({'ok': True, 'message': filepath}), 200

    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/copy-slice', methods=['GET'])
def extract_copy_slice(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
