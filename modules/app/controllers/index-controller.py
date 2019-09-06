import os
from flask import request, jsonify
from app import app, data_path
from werkzeug.utils import secure_filename

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
            return jsonify({'ok': True, 'message': f'{f.filename} have been successfully uploaded.'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Files failed to upload.'}), 400