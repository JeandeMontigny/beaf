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
        # recording from selected channels, for selected time windows
        # [[ch nb, [rec], [frame start, frame end] ], [...]]
        self.recording = []

    def read_raw_data(self, t_start, t_end, ch_to_extract, frame_chunk, verbose, use_c_lib):
        frame_start =  int(np.floor(t_start * self.info.sampling_rate))
        frame_end = int(np.floor(t_end * self.info.sampling_rate))
        # resize frame_chunk if it is larger than the recording length to extract
        if frame_chunk > self.info.recording_length / self.info.nb_channel:
            frame_chunk = frame_end - frame_start

        if frame_start > self.info.recording_length / self.info.nb_channel:
            raise SystemExit("Requested start time of recording to extract is higher that the recording length")

        # comparison in bit
        if  frame_chunk * self.info.nb_channel * 2 > psutil.virtual_memory().available:
            raise SystemExit("Memory size of the recording chunk to extract is bigger than your available system memory. Try again using a smaller frame_chunk value")

        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * self.info.nb_channel

        first_frame = frame_start * self.info.nb_channel
        last_frame = int(first_frame + id_frame_chunk)
        for chunk in range(0, nb_frame_chunk):
            if verbose:
                print("Reading chunk %s out of %s" %(chunk+1, nb_frame_chunk), end = "\r")
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * self.info.nb_channel

            data_chunk = self.data.get("Well_A1").get("Raw")[first_frame:last_frame+self.info.nb_channel]

            first_frame += id_frame_chunk + self.info.nb_channel
            last_frame = int(first_frame + id_frame_chunk)

            if use_c_lib:
                import ctypes
                c_lib = ctypes.CDLL("../beaf/c_lib/c_lib.dll")
                c_lib.process_raw_data_chunk.argtypes = ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int,ctypes.c_int
                c_lib.process_raw_data_chunk.restype = None
                rec = [[0.0] * int(self.info.recording_length)] * len(ch_to_extract)

                c_lib.process_raw_data_chunk(data_chunk.tolist(), int(self.info.recording_length), ch_to_extract, rec, int(self.info.max_analog_value), int(self.info.min_analog_value), int(self.info.max_digital_value), int(self.info.min_digital_value))

                for ch_id in range(0, len(ch_to_extract)):
                    self.recording[ch_id][1] = rec[ch_id]
                del(rec)

            else:
                # for each frame in this data chunk
                for frame_nb in range(0, self.info.recording_length):
                    frame_start_id = frame_nb*self.info.nb_channel

                    for ch_id in range(0, len(ch_to_extract)):
                        ch = ch_to_extract[ch_id]
                        self.recording[ch_id][1].append(convert_digital_to_analog(self.info, data_chunk[frame_start_id + ch]))

        for ch_id in range (0, len(ch_to_extract)):
            self.recording[ch_id][2].append([frame_start, frame_end])

        if verbose:
            print("\ndone")


    def read_raw_compressed_data(self, t_start, t_end, ch_to_extract, frame_chunk):
        # data chunk [start-end[ in number of frame
        toc = self.data.get("TOC")
        # data chunk start in number of element in EventsBasedSparseRaw list (EventsBasedSparseRaw[id])
        event_sparse_raw_toc = self.data.get("Well_A1").get("EventsBasedSparseRawTOC")

        # EventsBasedSparseRaw dataset is stored in byte (8 bit). Values have different format:
        #     'ChId' and  'Size' are int (32 bit) values: encoded on bytes of EventsBasedSparseRaw
        #     'range begin' and 'range end' are long (64 bit): encoded on 8 bytes of EventsBasedSparseRaw
        #     'sample' values are short (16 bit); encoded on 2 bytes of EventsBasedSparseRaw
        #
        #             Data chunk n                        Data chunk n+1
        # ¦                                ¦                                  ¦
        # ChData 1, ChData 2, ..., ChData n; ChData 1, ChData 2, ..., ChData n;
        #     _____¦        ¦___________________________
        #     ChID, size, range 1, range 2, ..., range n
        #     ___________¦       ¦_______________________________________
        #     range beging, range end, sample 1, sample 2, ...., sample n

        # size: number of bytes composing the ranges of this ChData (so ChData size without ChID + size)
        # range begin: frame number relative to the begininng of the rec of sample 1
        # range end: frame number +1 relative to the begininng of the rec of next last sample of this range data
        #   i.e. range end - range begin = nb of sample for this range

        # get data chunk corresponding to t_start-t_end, using toc (in frame)
        frame_start =  int(np.floor(t_start * self.info.sampling_rate))
        frame_end = int(np.floor(t_end * self.info.sampling_rate))
        if frame_start > toc[len(toc)-1][1]:
            raise SystemExit("Requested start time of recording to extract is higher that the recording length")
        if frame_end > toc[len(toc)-1][1]:
            frame_end = self.info.recording_length

        chunk_nb_start = 0; chunk_nb_end = 0
        for chunk_nb in range(0, len(toc)):
            if toc[chunk_nb][0] <= frame_start:
                chunk_nb_start = chunk_nb
            if toc[chunk_nb][1] >= frame_end:
                chunk_nb_end = chunk_nb +1
                break
        print(chunk_nb_end - chunk_nb_start, "data chunks to read")

        for data_chunk_nb in range(chunk_nb_start, chunk_nb_end):
            chunk_start_id = event_sparse_raw_toc[data_chunk_nb]
            if data_chunk_nb < len(event_sparse_raw_toc)-1:
                chunk_end_id = event_sparse_raw_toc[data_chunk_nb+1]
            else:
                chunk_end_id = len(self.data.get("Well_A1").get("EventsBasedSparseRaw"))

            data_chunk = self.data.get("Well_A1").get("EventsBasedSparseRaw")[chunk_start_id:chunk_end_id]

            i = 0
            while i < len(data_chunk):
                ch_id = int.from_bytes([data_chunk[i], data_chunk[i+1], data_chunk[i+2], data_chunk[i+3]], byteorder='little')
                size = int.from_bytes([data_chunk[i+4], data_chunk[i+5], data_chunk[i+6], data_chunk[i+7]], byteorder='little')
                # update i to be the index of first range
                i += 8
                if len(ch_to_extract) == 4096 or (ch_id in ch_to_extract):
                    rec_ch_id = 0
                    for ch in range(0, len(ch_to_extract)):
                        if self.recording[ch][0] == ch_id:
                            rec_ch_id = ch
                    j = 0
                    while j < size:
                        range_begin = int.from_bytes([data_chunk[i+j+k] for k in range(0, 8)], byteorder='little')
                        range_end = int.from_bytes([data_chunk[i+j+k+8] for k in range(0, 8)], byteorder='little')
                        # if is not within desired time windows to extrat, break
                        if range_begin < frame_start or range_begin > frame_end:
                            break
                        for k in range(0, range_end - range_begin):
                            sample = int.from_bytes([data_chunk[i+16+k*2], data_chunk[i+17+k*2]], byteorder='little')
                            self.recording[rec_ch_id][1].append(convert_digital_to_analog(self.info, sample))
                        self.recording[rec_ch_id][2].append([range_begin, range_end])
                        j += 16 + (range_end - range_begin)*2

                # update i to be the index of next ChData
                i += size


    def read(self, t_start, t_end, ch_to_extract, frame_chunk, verbose, use_c_lib):
        self.info = get_brw_experiment_setting(self.path)
        self.data = h5py.File(self.path,'r')

        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)
                self.recording.append([ch, [], []])
        else:
            self.recording = [[ch, [], []]  for ch in ch_to_extract]

        if t_end == "all": t_end = self.info.get_recording_length_sec()

        if self.info.recording_type == "RawDataSettings":
            self.read_raw_data(t_start, t_end, ch_to_extract, frame_chunk, verbose, use_c_lib)
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
        self.spike_times = []
        self.spike_channels = []
        self.sorted = []

    def read_spike_brx_data(self, t_start, t_end, ch_to_extract, sort):
        frame_start =  int(np.floor(t_start * self.info.sampling_rate))
        frame_end = int(np.floor(t_end * self.info.sampling_rate))

        id_start = 0; id_end = 0
        init_frame_start = False; init_frame_end = False
        if frame_start == 0:
            id_start = 0
            init_frame_start = True
        if frame_end >= self.info.recording_length:
            id_end = self.info.recording_length
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
            self.sorted = [[] for ch in range(0, self.info.nb_channel)]
            for i in range(0, len(self.spike_times)):
                if len(ch_to_extract) == self.info.nb_channel or self.spike_channels[i] in ch_to_extract:
                    self.sorted[self.spike_channels[i]].append(self.spike_times[i])


    def read(self, t_start, t_end, ch_to_extract, sort):
        # TODO: other brx event than spikes
        self.info = get_bxr_experiment_setting(self.path)
        self.data = h5py.File(self.path,'r')

        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)

        if t_end == "all": t_end = self.info.get_recording_length_sec()

        # if type == spikes:
        self.read_spike_brx_data(t_start, t_end, ch_to_extract, sort)

        self.data.close()

class Brw_Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')
        self.recording_type = ""
        # in frame
        self.recording_length = np.nan

    def read(self):
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

        if self.recording_type == "RawDataSettings":
            self.recording_length = len(self.data.get("Well_A1").get("Raw")) / self.nb_channel
        elif self.recording_type == "NoiseBlankingCompressionSettings":
            self.recording_length = self.data.get("TOC")[len(self.data.get("TOC"))-1][1]

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
        return int(self.recording_length)

    def get_recording_length_sec(self):
        return self.recording_length / self.get_sampling_rate()

    def close(self):
        self.data.close()


class Bxr_Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')
        # in frame
        self.recording_length = 0

    def read(self):
        # TODO: other brx event than spikes
        experiment_settings = json.loads(self.data.get("ExperimentSettings").__getitem__(0))

        self.mea_model = experiment_settings['MeaPlate']['Model']
        self.sampling_rate = experiment_settings['TimeConverter']['FrameRate']
        self.channel_idx = self.data.get('Well_A1').get('StoredChIdxs')[:]
        self.nb_channel = len(self.channel_idx)
        self.recording_length = self.data.get('Well_A1').get('SpikeTimes')[len(self.data.get('Well_A1').get('SpikeTimes'))-1]


    def get_mea_model(self):
        return self.mea_model

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_nb_channel(self):
        return self.nb_channel

    def get_recording_length(self):
        return int(self.recording_length)

    def get_recording_length_sec(self):
        return self.recording_length / self.get_sampling_rate()

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


def read_brw_file(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000, verbose=False, use_c_lib = False):
    """
    TODO: description
    """
    File = Brw_File(file_path)
    File.read(t_start, t_end, ch_to_extract, frame_chunk, verbose, use_c_lib)

    return File


def read_bxr_file(file_path, t_start = 0, t_end = 60, ch_to_extract = [], sort = False):
    """
    TODO: description
    """
    File = Bxr_File(file_path)
    File.read(t_start, t_end, ch_to_extract, sort)

    return File


def get_brw_experiment_setting(file_path):
    """
    TODO: description
    """
    experiment_setting = Brw_Experiment_Settings(file_path)
    experiment_setting.read()
    experiment_setting.close()

    return experiment_setting


def get_bxr_experiment_setting(file_path):
    """
    TODO: description
    """
    experiment_setting = Bxr_Experiment_Settings(file_path)
    experiment_setting.read()
    experiment_setting.close()

    return experiment_setting
