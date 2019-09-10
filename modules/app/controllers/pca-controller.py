import os
from flask import request, jsonify
from app import app, data_path, data_config#, mongo
import logger
from moseq2_pca.gui import *
import json
from ast import literal_eval
import glob

@app.route('/train-pca', methods=['GET', 'POST'])
def pca_train(path=None):
    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        input_params = request.form.to_dict()

        if input_params != {}:
            for k, v in input_params.items():
                if ',' in v:
                    input_params[k] = literal_eval('('+v+')')
                elif v == 'on':
                    input_params[k] = True
                elif v == 'off':
                    input_params[k] = False
                elif v.isdigit() or '-' in v:
                    input_params[k] = int(v)

            ret = train_pca_command(cwd1, 'local', cwd1+'_pca', input_params['gaussian_filter_space'], input_params['gaussian_filter_time'],
                              input_params['med_filter_space'], input_params['med_filter_time'], False, input_params['missing_data_iters'],
                              input_params['mask_threshold'], input_params['mask_height_threshold'], input_params['min_height'], input_params['max_height'],
                              input_params['tailfilter_size'], input_params['tailfilter_shape'], input_params['use_fft'], input_params['recon_pcs'], input_params['rank'],
                              'pca', input_params['chunk_size'], input_params['visualize_results'], cwd1+'config.yaml', cwd1+'tmp/', input_params['queue'], True, input_params['nworkers'],
                              input_params['cores'], input_params['processes'], input_params['memory'], input_params['wall_time'], input_params['timeout'])
            if ret:
                pca_files = [f.replace(cwd1, '') for f in glob.glob(cwd1 + "_pca/*", recursive=True)]
                print(pca_files)
                # save pca model path in sidebar.json
                with open(cwd + data_config + 'sidebar-progress.json') as json_file:
                    data = json.load(json_file)
                    for f in pca_files:
                        if f not in data['pca_files']:
                            data['pca_files'].append(f)

                with open(cwd + data_config + 'sidebar-progress.json', 'w') as outfile:
                    json.dump(data, outfile)

                return jsonify({'ok': True, 'message': 'Training was successful'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400


@app.route('/apply-pca', methods=['POST'])
def pca_apply(path=None):
    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        input_params = request.form.to_dict()

        if input_params != {}:
            for k, v in input_params.items():
                if ',' in v:
                    input_params[k] = literal_eval('(' + v + ')')
                elif v == 'on':
                    input_params[k] = True
                elif v == 'off':
                    input_params[k] = False
                elif v.isdigit() or '-' in v:
                    input_params[k] = int(v)

            ret = apply_pca_command(cwd1, 'local', cwd1 + '_pca', 'pca_scores', '/frames', '/frames_mask',
                              '/components', cwd1 + input_params['pca-file'], input_params['chunk_size'], input_params['fill_gaps'], input_params['fps'],
                              input_params['detrend_window'], cwd1+'config.yaml', 'debug', input_params['queue'], input_params['nworkers'],
                              input_params['cores'], input_params['processes'], input_params['memory'], input_params['wall_time'], input_params['timeout'])

            if os.path.exists(cwd1+'/_pca/pca_scores.h5') and ret:
                pca_files = [f.replace(cwd1, '') for f in glob.glob(cwd1 + "_pca/*", recursive=True)]
                # save pca model path in sidebar.json
                with open(cwd + data_config + 'sidebar-progress.json') as json_file:
                    data = json.load(json_file)
                    for f in pca_files:
                        if f not in data['pca_files']:
                            data['pca_files'].append(f)

                with open(cwd + data_config + 'sidebar-progress.json', 'w') as outfile:
                    json.dump(data, outfile)
                return jsonify({'ok': True, 'message': 'PCA Scores were successfully computed!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/clip-pca-scores', methods=['GET'])
def clip_pca_scores(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

@app.route('/compute-changepoints', methods=['GET', 'POST', 'PATCH'])
def pca_compute_cps(path=None):
    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        input_params = request.form.to_dict()

        if input_params != {}:
            for k, v in input_params.items():
                if ',' in v:
                    input_params[k] = literal_eval('(' + v + ')')
                elif v == 'on':
                    input_params[k] = True
                elif v == 'off':
                    input_params[k] = False
                elif v.isdigit() or '-' in v:
                    input_params[k] = int(v)
                elif isfloat(v):
                    input_params[k] = float(v)

            ret = compute_changepoints_command(cwd1, cwd1 + '_pca/', 'changepoints', 'local', None,
                                    cwd1 + input_params['pca-scores'], '/components', input_params['neighbors'],
                                    input_params['threshold'], input_params['k_lags'], input_params['sigma'],
                                    input_params['dims'],  input_params['fps'], '/frames', '/frames_mask',
                                    input_params['chunk_size'], cwd1 + 'config.yaml', cwd1+'tmp/', True,
                                    input_params['queue'], input_params['nworkers'],
                                    input_params['cores'], input_params['processes'], input_params['memory'],
                                    input_params['wall_time'], input_params['timeout'])
            if ret:
                pca_files = [f.replace(cwd1, '') for f in glob.glob(cwd1 + "_pca/*", recursive=True)]
                # save pca model path in sidebar.json
                with open(cwd + data_config + 'sidebar-progress.json') as json_file:
                    data = json.load(json_file)
                    for f in pca_files:
                        if f not in data['pca_files']:
                            data['pca_files'].append(f)

                with open(cwd + data_config + 'sidebar-progress.json', 'w') as outfile:
                    json.dump(data, outfile)
                return jsonify({'ok': True, 'message': 'Changepoints were successfully computed!'}), 200

        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/clip-scores', methods=['GET', 'POST', 'PATCH'])
def pca_clip_scores(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

