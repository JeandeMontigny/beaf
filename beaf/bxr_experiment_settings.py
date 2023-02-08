import os, h5py, json

# ---------------------------------------------------------------- #
class Bxr_Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')
        self.name = os.path.basename(file_path)
        # TODO: other brx event than spikes
        experiment_settings = json.loads(self.data.get("ExperimentSettings").__getitem__(0))
        self.mea_model = experiment_settings['MeaPlate']['Model']
        self.sampling_rate = experiment_settings['TimeConverter']['FrameRate']
        self.channel_idx = self.data.get('Well_A1').get('StoredChIdxs')[:]
        self.nb_channel = len(self.channel_idx)
        # in frame
        self.recording_length = self.data.get('Well_A1').get('SpikeTimes')[len(self.data.get('Well_A1').get('SpikeTimes'))-1]

        self.data.close()
        del(self.data)

    def get_mea_model(self):
        return self.mea_model

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_channel_idx(self):
        return self.channel_idx

    def get_nb_channel(self):
        return self.nb_channel

    def get_recording_length(self):
        """
        Return the number of frame in the recording (per channel)
        """
        return int(self.recording_length)

    def get_recording_length_sec(self):
        return self.recording_length / self.get_sampling_rate()


# ---------------------------------------------------------------- #
def get_bxr_experiment_setting(file_path):
    """
    TODO: description
    """
    return Bxr_Experiment_Settings(file_path)
