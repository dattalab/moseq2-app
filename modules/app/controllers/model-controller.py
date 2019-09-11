import os
from flask import request, jsonify
from app import app, data_path, data_config#, mongo
import logger
import json
from moseq2_model.gui import *
from ast import literal_eval

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

@app.route('/learn-model', methods=['POST'])
def model_learn(path=None):
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
                    input_params[k] = False
                elif v == 'off':
                    input_params[k] = True
                elif v.isdigit() or '-' in v:
                    input_params[k] = int(v)
                elif isfloat(v):
                    input_params[k] = float(v)

            modelname = "model.p"
            ret = learn_model_command(cwd1+input_params['score-file'], cwd1+modelname, input_params['hold_out'],
                                      input_params['hold_out_seed'], input_params['nfolds'], 4,
                                      input_params['num_iter'], input_params['restarts'], input_params['var_name'],
                                      input_params['save_every'], input_params['save_model'], input_params['max_states'],
                                      True, input_params['npcs'], input_params['whiten'], input_params['kappa'], input_params['gamma'],
                                      input_params['alpha'], input_params['noise_level'], input_params['nu'], input_params['nlags'],
                                      input_params['separate_trans'], input_params['robust'], "", "n/a")

    if os.path.exists(cwd1 + modelname) and ret:
        # save pca model path in sidebar.json
        with open(cwd + data_config + 'sidebar-progress.json') as json_file:
            data = json.load(json_file)
            if 'model_files' not in data.keys():
                data['model_files'] = []
            if modelname not in data['model_files']:
                data["model_files"].append(modelname)

        with open(cwd + data_config + 'sidebar-progress.json', 'w') as outfile:
            json.dump(data, outfile)
        return jsonify({'ok': True, 'message': 'Model trained.'}), 200
    else:
        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/learn-model-parameter-scan', methods=['GET', 'POST', 'PATCH'])
def batch_learn_model(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/aggregate-modeling-results', methods=['GET', 'POST', 'PATCH'])
def batch_agg_modeling_res(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/count-frames', methods=['GET', 'POST', 'PATCH'])
def model_count_frames(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd + data_path

        os.system(f'cd {cwd1}')

        read = os.popen(f'moseq2-model count-frames {cwd1}_pca/pca_scores.h5').read()

        if read != None:
            return jsonify({'ok': True, 'message': f'Frames Counted: {read}'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400