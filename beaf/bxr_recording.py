import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt

from .utils import *
from .brw_experiment_settings import *

# ---------------------------------------------------------------- #
class Bxr_Recording:
"""
    TODO: description
    """
    def __init__(self, bxr_path):
        self.path = bxr_path
        self.data = []
        self.Info = []
        self.spike_times = []
        self.spike_channels = []
        self.sorted = []

    def read_spike_brx_data(self, t_start, t_end, ch_to_extract, sort):
        frame_start =  int(np.floor(t_start * self.Info.get_sampling_rate()))
        frame_end = int(np.floor(t_end * self.Info.get_sampling_rate()))

        id_start = 0; id_end = 0
        init_frame_start = False; init_frame_end = False
        if frame_start == 0:
            id_start = 0
            init_frame_start = True
        if frame_end >= self.Info.get_recording_length():
            id_end = self.Info.get_recording_length()
            init_frame_end = True
        if (not init_frame_start) and (not init_frame_end) :
            temps_spike_times = self.data.get('Well_A1').get('SpikeTimes')[:]
            for i in range(0, len(temps_spike_times)):
                # find first spike time > frame_start
                if (not init_frame_start) and temps_spike_times[i] >= frame_start:
                    id_start = i - 1
                    if frame_start < 0: frame_start = 0
                    init_frame_start = True
                # find last spike time < frame_end
                if (not init_frame_end) and temps_spike_times[i] >= frame_end:
                    id_end = i
                    init_frame_end = True
                if init_frame_start and init_frame_end:
                    break
            del(temps_spike_times)

        self.spike_times = self.data.get('Well_A1').get('SpikeTimes')[id_start:id_end]
        self.spike_channels = self.data.get('Well_A1').get('SpikeChIdxs')[id_start:id_end]

        # unsorted data (spike_times, spike_channels) are more convinient to clean data (ex, spike at same frame on most channels)
        if sort:
            self.sorted = [[] for ch in range(0, self.Info.get_nb_channel())]
            for i in range(0, len(self.spike_times)):
                if len(ch_to_extract) == self.Info.get_nb_channel() or self.spike_channels[i] in ch_to_extract:
                    self.sorted[self.spike_channels[i]].append(self.spike_times[i])


    def read(self, t_start, t_end, ch_to_extract, sort):
        # TODO: other brx event than spikes
        self.Info = get_bxr_experiment_setting(self.path)
        self.data = h5py.File(self.path,'r')

        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)

        if t_end == "all": t_end = self.Info.get_recording_length_sec()

        # if type == spikes:
        self.read_spike_brx_data(t_start, t_end, ch_to_extract, sort)

        self.data.close()


# ---------------------------------------------------------------- #
def read_bxr_file(file_path, t_start = 0, t_end = 60, ch_to_extract = [], sort = False):
    """
    TODO: description
    """
    Recording = Bxr_Recording(file_path)
    Recording.read(t_start, t_end, ch_to_extract, sort)

    return Recording
