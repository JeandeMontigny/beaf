import os, h5py, json

# ---------------------------------------------------------------- #
class Brw_Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')
        self.name = os.path.basename(file_path)

        experiment_settings = json.loads(self.data.get("ExperimentSettings").__getitem__(0))
        try:
            self.recording_type = experiment_settings['DataSettings']['Raw']['$type']
        except:
            self.recording_type = experiment_settings['DataSettings']['EventsBasedRawRanges']['$type']

        self.mea_model = experiment_settings['MeaPlate']['Model']
        self.sampling_rate = experiment_settings['TimeConverter']['FrameRate']
        #TODO: check that recorded channels are actually listed in data.get("Well_A1").get("StoredChIdxs")
        self.channel_idx = self.data.get("Well_A1").get("StoredChIdxs")[:]
        self.nb_channel = len(self.channel_idx)
        # in frame
        self.recording_length = self.data.get("TOC")[len(self.data.get("TOC"))-1][1]
        self.min_analog_value = experiment_settings['ValueConverter']['MinAnalogValue']
        self.max_analog_value = experiment_settings['ValueConverter']['MaxAnalogValue']
        self.min_digital_value = experiment_settings['ValueConverter']['MinDigitalValue']
        self.max_digital_value = experiment_settings['ValueConverter']['MaxDigitalValue']

        self.data.close()
        del(self.data)

    def get_recording_type(self):
        return self.recording_type

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_mea_model(self):
        return self.mea_model

    def get_nb_channel(self):
        return int(self.nb_channel)

    def get_channel_idx(self):
        return self.channel_idx

    def get_recording_length(self):
        """
        Return the number of frame in the recording (per channel)
        """
        return int(self.recording_length)

    def get_recording_length_sec(self):
        return self.recording_length / self.get_sampling_rate()

# ---------------------------------------------------------------- #
def get_brw_experiment_setting(file_path):
    """
    TODO: description
    """
    return Brw_Experiment_Settings(file_path)
