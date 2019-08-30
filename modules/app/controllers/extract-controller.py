import os
from flask import request, jsonify
from app import app, data_path#, mongo
import logger
import subprocess

@app.route('/check-local-dir', methods=['GET'])
def check_local_data_dir():
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd = cwd+data_path

        if len(os.listdir(cwd)) == 0:
            return jsonify({'ok': False, 'message': 'No data files found!'}), 400
        else:
            if ('depth.dat' in os.listdir(cwd)) and ('depth_ts.txt' in os.listdir(cwd)) and ('metadata.json' in os.listdir(cwd)):
                return jsonify({'ok': True, 'message': cwd}), 200
    else:
        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/generate-config', methods=['GET', 'POST', 'PATCH'])
def extract_generate_conf(path=None):
    if request.method == 'GET':
        # check if it exists
        cwd = os.getcwd()
        cwd1 = cwd + data_path

        if request.method == 'GET':
            # start cli command with default params unless get dict is not empty
            cd_cmd = 'cd ' + cwd
            os.system(cd_cmd)

            query = request.args

            if query == {}:
                os.system(f'moseq2-extract generate-config -o {cwd1}/config.yaml')
                if os.path.exists(cwd1+'config.yaml'):
                    return jsonify({'ok': True, 'message': cwd1+'/config.yaml'}), 200

        # if yes, send json warning, if accepted will receive a future PATCH request

        # if no, runs config file creation, returns success message and path to generated file

        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/gen-gridsearch-config', methods=['POST'])
def batch_gen_gridsearch_conf(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/extract-raw', methods=['GET'])
def extract_raw(path=None):

    cwd = os.getcwd()
    cwd1 = cwd + data_path

    if request.method == 'GET':
        #start cli command with default params unless get dict is not empty
        cd_cmd = 'cd '+cwd
        os.system(cd_cmd)

        query = request.args

        if query == {}:
            # run extract with default cmd
            filepath = cwd1+'depth.dat'
            configfile = 'config.yaml'
            '''
            extract(filepath, (80, 80), (10, 10), 'ellipse', 0, (1, .1, 1), (650, 750), False, 3000, 7, True, 10, 100,
                    30, None, 51, False, 100, -16, -100, True, 'raw', 0, 'rectangle', (5, 5), 1, (9, 9), 'ellipse', [3],
                    [0], 1000, 0, None, True, True, 0, 3, 0, 3, (0, 0), (0, 0), configfile, False, 3000, 3)
            '''

            os.system(f'moseq2-extract extract {filepath}')

            if os.path.exists(cwd1+'proc/'):
                cp_cmd = f'cp {cwd1}proc/results_00.mp4 {cwd}/modules/app/static/img/results_00.mp4'
                os.system(cp_cmd)
                return jsonify({'ok': True, 'message': 'Extraction successful'}), 200
            else:
                return jsonify({'ok': True, 'message': 'Extraction not successful'}), 400
        else:
            # run extract with user defined params
            if len(os.listdir(cwd1)) > 3:
                return jsonify({'ok': True, 'message': 'Extraction successful'}), 200
            else:
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
