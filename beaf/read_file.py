import os, sys, psutil, h5py, json
import numpy as np
from multiprocessing import Pool

# ---------------------------------------------------------------- #
#TODO: pritave/protected class members? (__name; _name respectively)
class Brw_File:
    """
    TODO: description
    """
    def __init__(self, brw_path):
        self.path = brw_path
        # dataset
        self.data = []
        # recording info from json
        self.info = []
        # recording from selected channels, for selected time windowns
        self.recording = []

    def read_raw_data(self, t_start, t_end, ch_to_extract, frame_chunk):
        frame_start =  int(np.floor(t_start * self.info.sampling_rate))
        frame_end = int(np.floor(t_end * self.info.sampling_rate))

        if frame_chunk > self.info.recording_length / self.info.nb_channel:
            print("Frame_chunk is bigger than the number of frame in the recording.")
            frame_chunk = frame_end - frame_start
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))

        # comparison in bit
        if  frame_chunk * self.info.nb_channel * 2 > psutil.virtual_memory().available:
            raise SystemExit("Memory size of the recording chunk to extract is bigger than your available system memory. Try again using a smaller frame_chunk value")

        id_frame_chunk = frame_chunk * self.info.nb_channel

        first_frame = frame_start * self.info.nb_channel
        last_frame = int(first_frame + id_frame_chunk)
        for chunk in range(0, nb_frame_chunk):
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * self.info.nb_channel

            data_chunk = self.data.get("Well_A1").get("Raw")[first_frame:last_frame+self.info.nb_channel]

            first_frame += id_frame_chunk + self.info.nb_channel
            last_frame = int(first_frame + id_frame_chunk)

            # for each frame in this data chunk
            for frame_nb in range(0, int(len(data_chunk)/self.info.nb_channel)):
                frame_start_id = frame_nb*self.info.nb_channel

                for ch_id in range(0, len(ch_to_extract)):
                    ch = ch_to_extract[ch_id]
                    self.recording[ch_id][1].append(convert_digital_to_analog(self.info, data_chunk[frame_start_id + ch]))


    #TODO
    def read_raw_compressed_data(self, t_start, t_end, ch_to_extract, frame_chunk):
        event_sparse_raw_toc = self.data.get("Well_A1").get("EventsBasedSparseRawTOC")
        event_sparse_raw = self.data.get("Well_A1").get("EventsBasedSparseRaw")[event_sparse_raw_toc[0]:event_sparse_raw_toc[1]]

        reconstructed_raw_data = []
        return reconstructed_raw_data


    def read(self, t_start, t_end, ch_to_extract, frame_chunk):
        self.info = get_brw_experiment_setting(self.path)
        self.data = h5py.File(self.path,'r')

        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)
                self.recording.append([ch, []])
        else:
            self.recording = [[ch, []] for ch in ch_to_extract]

        if t_end == "all": t_end = self.info.get_recording_length_sec()

        if self.info.recording_type == "RawDataSettings":
            self.read_raw_data(t_start, t_end, ch_to_extract, frame_chunk)
        elif self.info.recording_type == "NoiseBlankingCompressionSettings":
            self.read_raw_compressed_data(t_start, t_end, ch_to_extract, frame_chunk)

        self.data.close()

    def get_path(self):
        """
        TODO: description
        """
        return self.path

    def get_info(self):
        """
        TODO: description
        """
        return self.info

    def get_recording(self):
        """
        TODO: description
        """
        return self.recording

#TODO
class Bxr_File:
    """
    TODO: description
    """
    def __init__(self, bxr_path):
        self.path = bxr_path
        self.data = []
        self.info = []

    def read():
        self.data = h5py.File(self.path,'r')


class Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')

    def read(self):
        experiment_settings = json.loads(self.data.get("ExperimentSettings").__getitem__(0))
        try:
            self.recording_type = experiment_settings['DataSettings']['EventsBasedRawRanges']['$type']
        except:
            self.recording_type = experiment_settings['DataSettings']['Raw']['$type']
        self.mea_model = experiment_settings['MeaPlate']['Model']
        self.sampling_rate = experiment_settings['TimeConverter']['FrameRate']
        #TODO: if only recording zone (so less than 4096 channels), info from MeaPlate?
        self.nb_channel = 4096
        # if self.data.get("Well_A1").get("MeaPlate").get("") != "":
        #     total_channel_nb =
        self.recording_length = len(self.data.get("Well_A1").get("Raw"))
        self.min_analog_value = experiment_settings['ValueConverter']['MinAnalogValue']
        self.max_analog_value = experiment_settings['ValueConverter']['MaxAnalogValue']
        self.min_digital_value = experiment_settings['ValueConverter']['MinDigitalValue']
        self.max_digital_value = experiment_settings['ValueConverter']['MaxDigitalValue']


    def get_recording_type(self):
        return self.recording_type

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_mea_model(self):
        return self.mea_model

    def get_nb_channel(self):
        return self.nb_channel

    def get_recording_length(self):
        return self.recording_length

    def get_recording_length_sec(self):
        return round(self.recording_length / (self.get_nb_channel() * self.get_sampling_rate()), 3)

    def close(self):
        self.data.close()

# ---------------------------------------------------------------- #
def convert_digital_to_analog(info, value):
    digital_value = info.min_analog_value + value * (info.max_analog_value - info.min_analog_value) / (info.max_digital_value - info.min_digital_value)
    # clean saturated values
    #NOTE: keep? need implement a proper clean signal method
    if digital_value > 4095 or digital_value < -4095:
        digital_value = 0
    return digital_value


def read_brw_file(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000):
    """
    TODO: description
    """
    File = Brw_File(file_path)
    File.read(t_start, t_end, ch_to_extract, frame_chunk)

    return File


def read_bxr_file(file_path):
    """
    TODO: description
    """
    File = Bxr_File(file_path)
    File.read()

    return File


def get_brw_experiment_setting(file_path):
    """
    TODO: description
    """
    experiment_setting = Experiment_Settings(file_path)
    experiment_setting.read()
    experiment_setting.close()

    return experiment_setting
