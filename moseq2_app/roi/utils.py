'''

'''
class InteractiveFindRoiUtilites:
    '''

    '''

    def __init__(self):
        '''
        '''
        pass

    def get_session_config(self, session_config, overwrite):
        '''

        Parameters
        ----------
        session_config
        overwrite

        Returns
        -------
        '''

        # Read individual session config if it exists
        if session_config is None:
            self.generate_session_config(session_config)
        elif not exists(session_config):
            self.generate_session_config(session_config)
        else:
            if exists(session_config) and not overwrite:
                if os.stat(session_config).st_size <= 0:
                    self.generate_session_config(session_config)

        self.session_parameters = read_yaml(session_config)

        # Extra check: Handle broken session config files and generate default session params from config_data
        if self.session_parameters is None:
            self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}
        elif self.session_parameters == {}:
            self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}

        # add missing keys for newly found sessions
        if len(list(self.session_parameters.keys())) < len(self.keys):
            for key in self.keys:
                if key not in self.session_parameters:
                    self.session_parameters[key] = deepcopy(self.config_data)

    def generate_session_config(self, path):
        '''
        Generates the default/initial session configuration file.

        Returns
        -------
        '''
        warnings.warn('Session configuration file was not found. Generating a new one.')

        # Generate session config file if it does not exist
        with open(path, 'w+') as f:
            yaml.safe_dump(self.session_parameters, f)
    def compute_all_bgs(self):
        '''
        Computes all the background images before displaying the app to speed up user interaction.

        Returns
        -------
        '''

        for s, p in tqdm(self.sessions.items(), total=len(self.sessions.keys()), desc='Computing backgrounds'):
            try:
                # finfo is a key that points to a dict that contains the following keys:
                # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
                if 'finfo' not in self.session_parameters[s]:
                    self.session_parameters[s]['finfo'] = get_movie_info(p)
                    if p.endswith('.mkv'):
                        self.handle_mkv_files(s, p)

                # Compute background image; saving the image to a file
                self.session_parameters[s].pop('output_dir', None)
                get_bground_im_file(p, **self.session_parameters[s], output_dir=None)
            except:
                # Print error if an issue arises
                display(f'Error, could not compute background for session: {s}.')
                pass
