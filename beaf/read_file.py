import os, sys, psutil, h5py, json
import numpy as np
from multiprocessing import Pool
import spikeinterface.extractors as se
from .utils import *

# TODO: reduce code repetition between read_raw_* functions
# TODO: pritave/protected class members? (__name; _name respectively)
# ---------------------------------------------------------------- #
class Brw_SpikeInterface:
    """
    TODO: description
    """
    def read_raw_data_recording(self, brw_path, info, t_start, t_end, ch_to_extract, frame_chunk):
        frame_start, frame_end = get_file_frame_start_end(info, t_start, t_end, frame_chunk)
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * info.get_nb_channel()
        first_frame = frame_start * info.get_nb_channel()
        last_frame = first_frame + id_frame_chunk

        hdf_file = h5py.File(brw_path,'r')
        traces_list = []
        for chunk in range(0, nb_frame_chunk):
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * info.get_nb_channel()

            data_chunk = hdf_file.get("Well_A1").get("Raw")[first_frame:last_frame+info.get_nb_channel()]

            first_frame += id_frame_chunk + info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            for frame_nb in range(0, int(len(data_chunk)/info.get_nb_channel())):
                frame_data = []
                frame_start_id = frame_nb*info.get_nb_channel()

                for ch_id in range(0, len(ch_to_extract)):
                    ch = ch_to_extract[ch_id]
                    frame_data.append(convert_digital_to_analog(info, data_chunk[frame_start_id + ch - 1]))
                traces_list.append(frame_data)

        hdf_file.close()

        NR = se.NumpyRecording(traces_list=np.array(traces_list), sampling_frequency=info.get_sampling_rate(), channel_ids=ch_to_extract)

        return NR


    def read_raw_compressed_data(self, brw_path, info, t_start, t_end, ch_to_extract, frame_chunk):
        # TODO: create spikeinterface NumpyRecording object
        #       problem with recording not of the same length
        #           solution using RecordingSegment? a RecordingExtractor segment for each snippet
        #       reconstruct ch? using get_reconstructed_ch_raw_compressed
        #           heavy and time consuming

        # data chunk [start-end[ in number of frame
        toc = self.data.get("TOC")
        # data chunk start in number of element in EventsBasedSparseRaw list (EventsBasedSparseRaw[id])
        event_sparse_raw_toc = self.data.get("Well_A1").get("EventsBasedSparseRawTOC")
        frame_start, frame_end = get_file_frame_start_end(info, t_start, t_end)

        chunk_nb_start = 0; chunk_nb_end = 0
        for chunk_nb in range(0, len(toc)):
            if toc[chunk_nb][0] <= frame_start:
                chunk_nb_start = chunk_nb
            if toc[chunk_nb][1] >= frame_end:
                chunk_nb_end = chunk_nb +1
                break

        traces_list = []
        for data_chunk_nb in range(chunk_nb_start, chunk_nb_end):
            chunk_start_id = event_sparse_raw_toc[data_chunk_nb]
            if data_chunk_nb < len(event_sparse_raw_toc)-1:
                chunk_end_id = event_sparse_raw_toc[data_chunk_nb+1]
            else:
                chunk_end_id = len(self.data.get("Well_A1").get("EventsBasedSparseRaw"))

            data_chunk = self.data.get("Well_A1").get("EventsBasedSparseRaw")[chunk_start_id:chunk_end_id]

            # get the time of the first snippet  within t_start-t_end
            # check if any other channel in ch_to_extract has data for this time t
            #   if not, add 0 or np.nan for this time t
            #   if yes, extract these data as well to create the first frame in traces_list
            # get the following snippet (that can have already been partially extracted during the previous snippet)

            # naive algo:
            # for t in range(frame_start, frame_end):
            #   frame_data = []
            #   for ch with data at this time t:
            #       frame_data.append(channel's data)
            #   if no ch with data: skip to next snippet (or fill with 0 or artificial noise)

        hdf_file = h5py.File(brw_path,'r')
        hdf_file.close()

        return 0


    def read(self, brw_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe):
        info = get_brw_experiment_setting(brw_path)

        if t_end == "all": t_end = info.get_recording_length_sec()
        if ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)

        if info.get_recording_type() == "RawDataSettings":
            NR = self.read_raw_data_recording(brw_path, info, t_start, t_end, ch_to_extract, frame_chunk)
        if info.get_recording_type() == "NoiseBlankingCompressionSettings":
            NR = self.read_raw_compressed_data(brw_path, info, t_start, t_end, ch_to_extract, frame_chunk)

        if attach_probe:
            geom = []
            for ch_nb in ch_to_extract:
                ch_coord = get_ch_coord(ch_nb)
                geom.append([ch_coord[0]*60, ch_coord[1]*60])

            # create and attach probe
            probe = Probe(ndim=2, si_units='um')
            probe.set_contacts(positions=geom, shapes='square', shape_params={'width': 21})
            square_contour = [(-60, -60), (3900, -60), (3900, 3900), (-60, 3900)]
            probe.set_planar_contour(square_contour)
            # WARNING: device_channel_indices does not match channel number
            probe.set_device_channel_indices(range(len(ch_to_extract)))
            NR = NR.set_probe(probe)

        return NR


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

    def read_raw_data(self, t_start, t_end, ch_to_extract, frame_chunk, verbose):
        frame_start, frame_end = get_file_frame_start_end(self.info, t_start, t_end, frame_chunk)

        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * self.info.get_nb_channel()

        first_frame = frame_start * self.info.get_nb_channel()
        last_frame = first_frame + id_frame_chunk
        for chunk in range(0, nb_frame_chunk):
            if verbose:
                print("Reading chunk %s out of %s" %(chunk+1, nb_frame_chunk), end = "\r")
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * self.info.get_nb_channel()

            data_chunk = self.data.get("Well_A1").get("Raw")[first_frame:last_frame+self.info.get_nb_channel()]

            first_frame += id_frame_chunk + self.info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            # for each frame in this data chunk
            for frame_nb in range(0, int(len(data_chunk)/self.info.get_nb_channel())):
                frame_start_id = frame_nb*self.info.nb_channel

                for ch_id in range(0, len(ch_to_extract)):
                    ch = ch_to_extract[ch_id]
                    self.recording[ch_id][1].append(convert_digital_to_analog(self.info, data_chunk[frame_start_id + ch - 1]))

        for ch_id in range(0, len(ch_to_extract)):
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
        frame_start, frame_end = get_file_frame_start_end(self.info, t_start, t_end)

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
                ch_nb = int.from_bytes([data_chunk[i], data_chunk[i+1], data_chunk[i+2], data_chunk[i+3]], byteorder='little')
                size = int.from_bytes([data_chunk[i+4], data_chunk[i+5], data_chunk[i+6], data_chunk[i+7]], byteorder='little')
                # update i to be the index of first range
                i += 8
                if len(ch_to_extract) == 4096 or (ch_nb in ch_to_extract):
                    ch_id = 0
                    for ch in range(0, len(ch_to_extract)):
                        if self.recording[ch][0] == ch_nb:
                            ch_id = ch
                    j = 0
                    while j < size:
                        range_begin = int.from_bytes([data_chunk[i+j+k] for k in range(0, 8)], byteorder='little')
                        range_end = int.from_bytes([data_chunk[i+j+k+8] for k in range(0, 8)], byteorder='little')
                        # if is not within desired time windows to extrat, break
                        if range_begin < frame_start or range_begin > frame_end:
                            break
                        for k in range(0, range_end - range_begin):
                            sample = int.from_bytes([data_chunk[i+16+k*2], data_chunk[i+17+k*2]], byteorder='little')
                            self.recording[ch_id][1].append(convert_digital_to_analog(self.info, sample))
                        self.recording[ch_id][2].append([range_begin, range_end])
                        j += 16 + (range_end - range_begin)*2

                # update i to be the index of next ChData
                i += size


    def read(self, t_start, t_end, ch_to_extract, frame_chunk, verbose):
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
            self.read_raw_data(t_start, t_end, ch_to_extract, frame_chunk, verbose)
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
        frame_start =  int(np.floor(t_start * self.info.get_sampling_rate()))
        frame_end = int(np.floor(t_end * self.info.get_sampling_rate()))

        id_start = 0; id_end = 0
        init_frame_start = False; init_frame_end = False
        if frame_start == 0:
            id_start = 0
            init_frame_start = True
        if frame_end >= self.info.get_recording_length():
            id_end = self.info.get_recording_length()
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
            self.sorted = [[] for ch in range(0, self.info.get_nb_channel())]
            for i in range(0, len(self.spike_times)):
                if len(ch_to_extract) == self.info.get_nb_channel() or self.spike_channels[i] in ch_to_extract:
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

    def get_recording_type(self):
        return self.recording_type

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_mea_model(self):
        return self.mea_model

    def get_nb_channel(self):
        return int(self.nb_channel)

    def get_recording_length(self):
        """
        Return the number of frame in the recording (per channel)
        """
        return int(self.recording_length)

    def get_recording_length_sec(self):
        return self.recording_length / self.get_sampling_rate()


class Bxr_Experiment_Settings:
    """
    TODO: description
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path,'r')
        # TODO: other brx event than spikes
        experiment_settings = json.loads(self.data.get("ExperimentSettings").__getitem__(0))
        self.mea_model = experiment_settings['MeaPlate']['Model']
        self.sampling_rate = experiment_settings['TimeConverter']['FrameRate']
        self.channel_idx = self.data.get('Well_A1').get('StoredChIdxs')[:]
        self.nb_channel = len(self.channel_idx)
        # in frame
        self.recording_length = self.data.get('Well_A1').get('SpikeTimes')[len(self.data.get('Well_A1').get('SpikeTimes'))-1]

        self.data.close()

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


# ---------------------------------------------------------------- #
def get_file_frame_start_end(info, t_start, t_end, frame_chunk=100000):
    frame_start =  int(np.floor(t_start * info.get_sampling_rate()))
    frame_end = int(np.floor(t_end * info.get_sampling_rate()))

    if info.get_recording_type() == "RawDataSettings":
        # resize frame_chunk if it is larger than the recording length to extract
        if frame_chunk > info.get_recording_length():
            frame_chunk = frame_end - frame_start

        if frame_start > info.get_recording_length():
            raise SystemExit("Requested start time of recording to extract is higher than the recording length")
        # comparison in bit
        if  frame_chunk * info.get_nb_channel() * 2 > psutil.virtual_memory().available:
            raise SystemExit("Memory size of the recording chunk to extract is bigger than your available system memory. Try again using a smaller frame_chunk value")

    if info.get_recording_type == "NoiseBlankingCompressionSettings":
        hdf_file = h5py.File(info.file_path,'r')
        toc = hdf_file.get("TOC")
        if frame_start > toc[len(toc)-1][1]:
            raise SystemExit("Requested start time of recording to extract is higher that the recording length")
        if frame_end > toc[len(toc)-1][1]:
            frame_end = self.info.get_recording_length()

        hdf_file.close()

    return frame_start, frame_end


def convert_digital_to_analog(info, value):
    digital_value = info.min_analog_value + value * (info.max_analog_value - info.min_analog_value) / (info.max_digital_value - info.min_digital_value)
    # clean saturated values
    if digital_value > 4095 or digital_value < -4095:
        digital_value = 0
    return digital_value


def read_brw_SpikeInterface(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000, attach_probe=True):
    BNR = Brw_SpikeInterface()
    NR = BNR.read(file_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe)
    return NR


def read_brw_file(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000, verbose=False):
    """
    TODO: description
    """
    File = Brw_File(file_path)
    File.read(t_start, t_end, ch_to_extract, frame_chunk, verbose)

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
    return Brw_Experiment_Settings(file_path)


def get_bxr_experiment_setting(file_path):
    """
    TODO: description
    """
    return Bxr_Experiment_Settings(file_path)
