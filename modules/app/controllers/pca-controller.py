import os
from flask import request, jsonify
from app import app, data_path#, mongo
import logger

@app.route('/train-pca', methods=['GET'])
def pca_train(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-pca train-pca -i {cwd1} -o {cwd1}_pca/')

            if os.path.exists(cwd1+'/_pca/'):
                cp_cmd = f'cp {cwd1}_pca/pca_components.png {cwd}/modules/app/static/img/pca_components.png'
                os.system(cp_cmd)
                cp_cmd1 = f'cp {cwd1}_pca/pca_scree.png {cwd}/modules/app/static/img/pca_scree.png'
                os.system(cp_cmd1)
                return jsonify({'ok': True, 'message': 'Training was successful'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400

@app.route('/apply-pca', methods=['GET'])
def pca_apply(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-pca apply-pca -i {cwd1} -o {cwd1}_pca/')

            if os.path.exists(cwd1+'/_pca/pca_scores.h5'):
                return jsonify({'ok': True, 'message': 'PCA Scores were successfully computed!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/clip-pca-scores', methods=['GET'])
def clip_pca_scores(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/compute-changepoints', methods=['GET', 'POST', 'PATCH'])
def pca_compute_cps(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd + data_path

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-pca compute-changepoints -i {cwd1} -o {cwd1}/_pca/')
            if os.path.exists(cwd1 + '/_pca/changepoints.h5'):
                cp_cmd = f'cp {cwd1}_pca/changepoints_dist.png {cwd}/modules/app/static/img/changpoints_dist.png'
                os.system(cp_cmd)
                return jsonify({'ok': True, 'message': 'Changepoints were successfully computed!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400

@app.route('/clip-scores', methods=['GET', 'POST', 'PATCH'])
def pca_clip_scores(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

