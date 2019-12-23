import os
import glob
import ruamel.yaml as yaml
from moseq2_viz.gui import get_groups_command

def generate_index(experiments_basedir, savepath):
    experiments = os.listdir(experiments_basedir)

    output_dict = {
        'files': []
    }
    
    i = 0
    for experiment in experiments:
        exp = os.path.join(experiments_basedir, experiment)
        if os.path.isdir(exp):
            files = os.listdir(exp)
            an_files = [os.path.join(experiments_basedir, exp, f) for f in files if '.h5' in f]
            videos = [os.path.join(experiments_basedir, exp, f) for f in files if 'videos' in f]
            if len(videos) > 0:
                videos = videos[0]
                if os.path.exists(videos):
                    if os.path.isdir(videos):
                        video_files = [os.path.join(experiments_basedir, exp, f) for f in os.listdir(videos)]
            else:
                video_files = []
            
            if len(an_files) > 0:
                output_dict['files'].append(dict(path=an_files, videos=video_files, SessionName=experiment, SessionIndex=i, group='default'))
                i += 1


    with open(savepath, 'w') as f:
        yaml.safe_dump(output_dict, f)
    f.close()
    
    print(output_dict)

def get_groups_command(index_file):
    with open(index_file, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()

    groups = []
    sessionNames, sessionIndices = [], []
    for f in index_data['files']:
        groups.append(f['group'])
        sessionIndices.append(f['SessionIndex'])
        sessionNames.append(f['SessionName'])

    print('Total number of sessions:', len(set(sessionNames)))
    print('Total number of unique groups:', len(set(groups)))

    for i in range(len(sessionNames)):
        print('Session Name:', sessionNames[i], '; Session Index:', sessionIndices[i], '; group:', groups[i])


def set_group(index, groupname, index_filepath):
    with open(index_filepath, 'r') as f:
        indexfile = yaml.safe_load(f)
    f.close()
    
    for f in indexfile['files']:
        if index == f['SessionIndex']:
            f['group'] = groupname
            
    with open(index_filepath, 'w') as f:
        yaml.safe_dump(indexfile, f)
