import os
from flask import request, jsonify
from app import app, data_path, data_config
import glob
import json

@app.route('/uploadFile', methods=['POST', 'PATCH'])
def save_uploaded_file():
    if request.method == 'POST':
        cwd = os.getcwd()
        cwd1 = cwd + data_path
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        f = request.files['file']
        if not os.path.exists(os.path.join(cwd1, os.path.split(f.filename)[0])):
            os.mkdir(os.path.join(cwd1, os.path.split(f.filename)[0]))

        f.save(os.path.join(cwd1, f.filename))
        if os.path.exists(cwd1+f.filename):
            with open(cwd+data_config+'sidebar-progress.json') as json_file:
                data = json.load(json_file)
                if f.filename not in data['local_files']:
                    data['local_files'].append(f.filename)

            with open(cwd+data_config+'sidebar-progress.json', 'w') as outfile:
                json.dump(data, outfile)
            return jsonify({'ok': True, 'message': f'{f.filename} have been successfully uploaded.', 'files': data['local_files']}), 200
        else:
            return jsonify({'ok': False, 'message': 'Files failed to upload.'}), 400


@app.route('/get-local-dir', methods=['GET'])
def check_local_data_dir():
    if request.method == 'GET':
        cwd = os.getcwd()

        if len(os.listdir(cwd)) == 0:
            return jsonify({'ok': False, 'message': 'No data files found!'}), 400
        else:
            with open(cwd + data_config + 'sidebar-progress.json') as json_file:
                data = json.load(json_file)
            files = data['local_files']
            return jsonify({'ok': True, 'message': 'Successfully found files.', 'files': files, 'extracted': data['extracted_files']}), 200
    else:
        return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400