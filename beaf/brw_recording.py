import os, sys, h5py, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool

from .utils import *
from .brw_experiment_settings import *

# TODO: pritave/protected class members? (__name; _name respectively)
# ---------------------------------------------------------------- #
class Brw_Recording:
    """
    Brw_Recording class to read, store and visualise .brw recordings datas
    """
    def __init__(self, brw_path):
        self.path = brw_path
        # dataset. is emptied at the end of data extraction
        self.data = []
        # Brw_Experiment_Settings object, recording info from json
        self.Info = []
        # recording from selected channels, for selected time windows
        # [[ch nb, [rec], [frame start, frame end] ], [...]]
        self.recording = []

        # WIP: list of ch to extract. only potentially used for multiprocessing
        self.ch_to_extract = []
        # WIP: the current data chunk. only potentially used for multiprocessing
        self.data_chunk = []
        # WIP: digital to analog invariant value. only potentially used for multiprocessing
        self.converter_invariant = 0
        # WIP: the size, if frame, of each data chunk. only potentially used for multiprocessing
        self.frame_chunk = 0
        # WIP: the current chunk number. only potentially used for multiprocessing
        self.chunk_nb = 0


    # -------- getters -------- #

    def get_path(self):
        """
        Return the path of the rdw file.
        """
        return self.path

    def get_Info(self):
        """
        Return the Info object attached to this Brw_Recording. See Brw_Experiment_Settings for more information.
        """
        return self.Info

    def get_recording(self):
        """
        Return the lists of recordings.
        """
        return self.recording


    # -------- read -------- #

    # TODO: try using a simple return, containing all frame for this channel
    def get_ch_rec(self, ch) :
        ch_id = self.ch_to_extract.index(ch)

        for frame_nb in range(int(len(self.data_chunk)/self.Info.get_nb_channel())):
            frame_start_id = frame_nb * self.Info.get_nb_channel()
            self.recording[ch_id][1][frame_nb+(self.chunk_nb*self.frame_chunk)] = convert_digital_to_analog(self.Info.min_analog_value, self.data_chunk[frame_start_id + ch - 1], self.converter_invariant)


    # TODO: not working. modifying an object passed as argument duplicate that object.
    #       modifications are then done only to that copied object, and not on the original one.
    #       try using shared memory
    def read_raw_data_multiproc(self, t_start, t_end, ch_to_extract, frame_chunk, verbose):
        frame_start, frame_end = get_file_frame_start_end(self.Info, t_start, t_end, frame_chunk)
        self.ch_to_extract = ch_to_extract
        self.frame_chunk = frame_chunk

        for ch_id in range(len(ch_to_extract)):
            self.recording[ch_id][1] = [0.0] * (frame_end - frame_start + 1)

        self.converter_invariant = (self.Info.max_analog_value - self.Info.min_analog_value) / (self.Info.max_digital_value - self.Info.min_digital_value)


        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * self.Info.get_nb_channel()

        first_frame = frame_start * self.Info.get_nb_channel()
        last_frame = first_frame + id_frame_chunk
        for chunk in range(nb_frame_chunk):
            if verbose:
                print("Reading chunk %s out of %s" %(chunk+1, nb_frame_chunk), end = "\r")
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * self.Info.get_nb_channel()

            self.data = h5py.File(self.path,'r')
            data_chunk = self.data.get("Well_A1").get("Raw")[first_frame:last_frame+self.Info.get_nb_channel()]
            self.chunk_nb = chunk
            # TODO: object is duplicated when run with t.map
            #       thus, memory size is thread_nb times greater
            self.data_chunk = data_chunk

            self.data.close()
            # data has to be cleared as h5py objects cannot be pickled
            self.data = []

            first_frame += id_frame_chunk + self.Info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            # python does not take advantage of multi threading, so better to use cpu_count (i.e. nb of thread) / 2
            thread_nb = int(os.cpu_count()/2)
            t = Pool(thread_nb)
            t.map(self.get_ch_rec, ch_to_extract)
            t.close()
            self.data_chunk = []

        for ch_id in range(len(ch_to_extract)):
            self.recording[ch_id][2].append([frame_start, frame_end])

        if verbose:
            print("\ndone")


    def read_raw_data_dll(self, t_start, t_end, ch_to_extract, frame_chunk, verbose, dll_path):
        import clr
        # add reference to IO.dll and Common.dll
        clr.AddReference(os.path.join(dll_path, "3Brain.BrainWave.IO.dll"))
        clr.AddReference(os.path.join(dll_path, "3Brain.BrainWave.Common.dll"))

        # import C variable types and 3Brain methods
        from System import Int32, Double, Boolean
        from _3Brain.BrainWave.IO import BrwFile
        from _3Brain.BrainWave.Common import (MeaFileExperimentInfo, RawDataSettings, ExperimentType, MeaPlate)
        from _3Brain.Common import (MeaPlateModel, MeaChipRoi, MeaDataType)

        consumer = object()

        # open hdf5 file using dll method
        data = BrwFile.Open(self.path)
        # get this hdf5 recording info using dll method
        info = data.get_MeaExperimentInfo()

        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = get_file_frame_start_end(self.Info, t_start, t_end, frame_chunk)
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))

        # will contain the data of the currend chunk, read using dll ReadRawData method
        data_chunk = []
        # for each chunk needed to read the recording from t_start to t_end
        for chunk in range(nb_frame_chunk):
            if verbose:
                print("Reading chunk %s out of %s" %(chunk+1, nb_frame_chunk), end = "\r")
            # if this is the last chunk, needs to reduce the chunk size to read, to avoid reading beyond the end of the brw file
            if chunk == nb_frame_chunk-1:
                last_chunk = frame_end - int(frame_start + chunk * frame_chunk)
                # read this chunk
                data_chunk = data.ReadRawData(int(frame_start + chunk * frame_chunk), last_chunk, data.get_SourceChannels(), consumer)
            # ReadRawData returns a 3D array. first index is the well number (index 0 if single well),
            # second index is the channel, third index the time frame
            else:
                # read this chunk
                data_chunk = data.ReadRawData(int(frame_start + chunk * frame_chunk), frame_chunk, data.get_SourceChannels(), consumer)

            # for each channel to extract
            for ch_id in range(len(ch_to_extract)):
                # convert to voltage
                ch_data = np.fromiter(info.DigitalToAnalog(data_chunk[0][ch_to_extract[ch_id]]), float)
                # add this chunk data at the end of this ch data array
                self.recording[ch_id][1] = np.concatenate([self.recording[ch_id][1], ch_data])

        # Close Files
        data.Close()

        # for each extracted channel, add the recording snippet frame_start and frame_end to their corresponding self.recording
        for ch_id in range(len(ch_to_extract)):
            self.recording[ch_id][2].append([frame_start, frame_end])

        if verbose:
            print("\ndone")


    def read_raw_data(self, t_start, t_end, ch_to_extract, frame_chunk, verbose):
        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = get_file_frame_start_end(self.Info, t_start, t_end, frame_chunk)
        # create 0 lists of desired recording length to avoid memory reallocation
        for ch_id in range(len(ch_to_extract)):
            self.recording[ch_id][1] = [0.0] * (frame_end - frame_start + 1)
        # calculate the invariant term for digital to analog conversion
        converter_invariant = (self.Info.max_analog_value - self.Info.min_analog_value) / (self.Info.max_digital_value - self.Info.min_digital_value)

        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * self.Info.get_nb_channel()

        # used to get the proper point value for each channel from the data_chunk
        rec_ch_idx = {}
        for id in range(self.Info.get_nb_channel()):
            rec_ch_idx[self.Info.get_channel_idx()[id]] = id

        # first frame position of the first chunk
        first_frame = frame_start * self.Info.get_nb_channel()
        # last frame position of the first chunk
        last_frame = first_frame + id_frame_chunk

        # for each chunk needed to read the recording from t_start to t_end
        for chunk in range(nb_frame_chunk):
            if verbose:
                print("Reading chunk %s out of %s" %(chunk+1, nb_frame_chunk), end = "\r")
            # if this is the last chunk of data, need to update last_frame to be the index of last needed frame (avoid out of bound)
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * self.Info.get_nb_channel()

            # open hdf5 file
            self.data = h5py.File(self.path,'r')
            # read data from first_frame to last_frame of this data chunk
            data_chunk = self.data.get("Well_A1").get("Raw")[first_frame:last_frame+self.Info.get_nb_channel()]
            # close the hdf5 file
            self.data.close()
            # data has to be cleared as h5py objects cannot be pickled
            self.data = []

            # update first_frame and last_frame position for next data chunk
            first_frame += id_frame_chunk + self.Info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            # for each frame in this data chunk
            for frame_nb in range(int(len(data_chunk)/self.Info.get_nb_channel())):
                # get the position of the current frame
                frame_start_id = frame_nb * self.Info.get_nb_channel()

                # for each channel to extract
                for ch_id in range(len(ch_to_extract)):
                    # get the channel number
                    ch = ch_to_extract[ch_id]
                    # convert digital value to analog and add this frame for this channel to corresponding self.recording list
                    self.recording[ch_id][1][frame_nb+(chunk*frame_chunk)] = convert_digital_to_analog(self.Info.min_analog_value, data_chunk[frame_start_id + rec_ch_idx.get(ch) - 1], converter_invariant)

        # for each extracted channel, add the recording snippet frame_start and frame_end to their corresponding self.recording
        for ch_id in range(len(ch_to_extract)):
            self.recording[ch_id][2].append([frame_start, frame_end])

        if verbose:
            print("\ndone")


    def read_raw_compressed_data(self, t_start, t_end, ch_to_extract, frame_chunk):
        self.data = h5py.File(self.path,'r')
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

        # get frame_start and frame_end corresponding to t_start t_end
        frame_start, frame_end = get_file_frame_start_end(self.Info, t_start, t_end)
        # calculate the invariant term for digital to analog conversion
        converter_invariant = (self.Info.max_analog_value - self.Info.min_analog_value) / (self.Info.max_digital_value - self.Info.min_digital_value)

        # get the first and last chunk of data to read to get the recording from t_start to t_end
        chunk_nb_start = 0; chunk_nb_end = 0
        for chunk_nb in range(len(toc)):
            if toc[chunk_nb][0] <= frame_start:
                chunk_nb_start = chunk_nb
            if toc[chunk_nb][1] >= frame_end:
                chunk_nb_end = chunk_nb +1
                break
        print(chunk_nb_end - chunk_nb_start, "data chunks to read")

        # for each chunk between the first and last chunk
        for data_chunk_nb in range(chunk_nb_start, chunk_nb_end):
            # # open hdf5 file
            self.data = h5py.File(self.path,'r')
            # get the start id of this chunk
            chunk_start_id = event_sparse_raw_toc[data_chunk_nb]
            # get the end id of this chunk
            if data_chunk_nb < len(event_sparse_raw_toc)-1:
                chunk_end_id = event_sparse_raw_toc[data_chunk_nb+1]
            else:
                # if this is the last chunk of the brw file
                chunk_end_id = len(self.data.get("Well_A1").get("EventsBasedSparseRaw"))

            # read data from chunk_start_id to chunk_end_id of this data chunk
            data_chunk = self.data.get("Well_A1").get("EventsBasedSparseRaw")[chunk_start_id:chunk_end_id]
            self.data.close()
            # data has to be cleared as h5py objects cannot be pickled
            self.data = []

            # index to parse the data
            i = 0
            # for each ChData to read
            while i < len(data_chunk):
                # get the channel number from bytes to int
                ch_nb = int.from_bytes([data_chunk[i], data_chunk[i+1], data_chunk[i+2], data_chunk[i+3]], byteorder='little')
                # get the size of this ChData from bytes to int
                size = int.from_bytes([data_chunk[i+4], data_chunk[i+5], data_chunk[i+6], data_chunk[i+7]], byteorder='little')
                # update i to be the index of first range
                i += 8

                # if need to extract this channel (ch_nb)
                if len(ch_to_extract) == 4096 or (ch_nb in ch_to_extract):
                    ch_id = 0
                    # get the id of this channel
                    for ch in range(len(ch_to_extract)):
                        if self.recording[ch][0] == ch_nb:
                            ch_id = ch

                    j = 0
                    # for each range in this ChData
                    while j < size:
                        # get the range begin for this range from bytes
                        range_begin = int.from_bytes([data_chunk[i+j+k] for k in range(8)], byteorder='little')
                        # get the range end for this range from bytes
                        range_end = int.from_bytes([data_chunk[i+j+k+8] for k in range(8)], byteorder='little')
                        # if is not within desired time windows to extrat, break
                        if range_begin < frame_start or range_begin > frame_end:
                            break

                        # get each frame in this range
                        for k in range(range_end - range_begin):
                            # get frame for bytes
                            sample = int.from_bytes([data_chunk[i+16+k*2], data_chunk[i+17+k*2]], byteorder='little')
                            # convert digital value to analog and add this frame corresponding self.recording list channel
                            self.recording[ch_id][1].append(convert_digital_to_analog(self.Info.min_analog_value, sample, converter_invariant))
                        # add the recording snippet frame start and frame end to their corresponding self.recording
                        self.recording[ch_id][2].append([range_begin, range_end])
                        # update the index j to be the next range of this ChData
                        j += 16 + (range_end - range_begin)*2

                # update i to be the index of next ChData
                i += size


    def read(self, t_start, t_end, ch_to_extract, frame_chunk, multiproc, verbose, use_dll, dll_path):
        # get Brw_Experiment_Settings info for this brw file
        self.Info = get_brw_experiment_setting(self.path)

        # if all channels are to be extracted
        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in self.Info.get_channel_idx():
                # add this channel to the list of channels to extract
                ch_to_extract.append(ch)
                # initialise self.recording for this channel
                self.recording.append([ch, [], []])
        # if only want to extract channels specified is ch_to_extract
        else:
            for ch in ch_to_extract:
                if ch in self.Info.get_channel_idx():
                    # initialise self.recording for this channel
                    self.recording.append([ch, [], []])
                else:
                    print("Channel", ch, "is not in the recording.")

        # t_end is "all", set t_end to recording length
        if t_end == "all": t_end = self.Info.get_recording_length_sec()
        # if the specified frame_chunk is larger than the number of frame in the interval to extract
        if frame_chunk > (t_end - t_start) * self.Info.get_sampling_rate():
            # set frame_chunk to the number of frame in the interval to extract
            frame_chunk = int((t_end - t_start) * self.Info.get_sampling_rate())

        # check recording type and use the relevant method to read data
        if self.Info.recording_type == "RawDataSettings":
            if use_dll:
                self.read_raw_data_dll(t_start, t_end, ch_to_extract, frame_chunk, verbose, dll_path)
            else:
                if multiproc:
                    self.read_raw_data_multiproc(t_start, t_end, ch_to_extract, frame_chunk, verbose)
                else:
                    self.read_raw_data(t_start, t_end, ch_to_extract, frame_chunk, verbose)

        elif self.Info.recording_type == "NoiseBlankingCompressionSettings":
            self.read_raw_compressed_data(t_start, t_end, ch_to_extract, frame_chunk)



    def save_recording(self, file_path):
        """
        Save the Brw_Recording object as a pickle file
        """
        with open(file_path, "wb") as file_handler:
            pickle.dump(self, file_handler)


    def merge_recordings(self, Rec_b):
        """
        merge two Brw_Recording object if they are of the same sampling frequency and have the same number of channels
        """
        if self.Info.get_sampling_rate() != Rec_b.Info.get_sampling_rate():
            print("Recordings to merge must have the same sampling rate, but have", self.Info.get_sampling_rate(), "and", Rec_b.Info.get_sampling_rate(), "respectively.")
            return

        if self.Info.recording_type == "RawDataSettings":
            if len(self.recording) != len(Rec_b.recording):
                print("Recordings to merge must have the same number of channels, but have", len(self.recording[1]), "and", len(Rec_b.recording[1]), "respectively.")
                return

            for ch_id in range(len(self.recording)):
                self.recording[ch_id][1] = np.concatenate((self.recording[ch_id][1], Rec_b.recording[ch_id][1]))
                self.recording[ch_id][2] = [[self.recording[ch_id][2][0][0], Rec_b.recording[ch_id][2][0][1]]]

        if self.Info.recording_type == "NoiseBlankingCompressionSettings":
            # TODO
            print("not implemented yet for NoiseBlankingCompression format.")
            return


    # -------- visualisation -------- #

    def plot_raw(self, ch_to_display, t_start=0, t_end="all", y_min=None, y_max=None, visualisation="aligned", artificial_noise=False, n_std=15, seed=0):
        """
        Plot the raw signal (time series) for the specified channels and time interval.

        Parameters
        ----------
        ch_to_display: list
            channels to extract
        t_start: float
            first time point, in seconds
        t_end: float
            last time point, in seconds
        y_min: None or float
            when specified, set the minimum value for the y axis
        y_max: None or float
            when specified, set the maximum value for the y axis
        visualisation: string
            Specify the dispay mode for raw compressed format. default value is "aligned". Can be "aligned", "reconstructed", "continuous" or "superimposed"
        artificial_noise: Bool
            Specific to raw compressed format. If True, add artificial noise to the recording
        n_std: float
            Specific to raw compressed format. If artificial_noise is True, set the standard deviation of the artificial noise
        seed: int
            Specific to raw compressed format. If artificial_noise is True, set the random generator seed to create artificial noise
        """
        # distribtue to sub functions plot_raw_format or plot_raw_compressed depending on recording format
        if self.Info.recording_type == "RawDataSettings":
            self.plot_raw_format(ch_to_display, t_start, t_end, y_min, y_max)
        if self.Info.recording_type == "NoiseBlankingCompressionSettings":
            self.plot_raw_compressed(ch_to_display, t_start, t_end, y_min, y_max, visualisation, artificial_noise, n_std, seed)


    def plot_raw_format(self, ch_to_display, t_start, t_end, y_min, y_max):
    # TODO: plot in lines or in MEA shape
        ch_to_display = check_ch_to_display(self.Info, ch_to_display)
        plt.rcdefaults()

        rec_frame_start = self.recording[0][2][0][0]
        rec_frame_end = self.recording[0][2][0][1]
        frame_start, frame_end = self.get_frame_start_end(t_start, t_end, ch_to_display)

        fig = plt.figure()

        fig_nb = 1
        for ch in ch_to_display:
            ch_id = 0
            for idx in range(len(self.recording)):
                if self.recording[idx][0] == ch:
                    ch_id = idx
                    break

            ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)
            plt.plot([x/self.Info.get_sampling_rate() for x in range(int(frame_start+rec_frame_start), int(frame_end+rec_frame_start))], self.recording[ch_id][1][int(frame_start):int(frame_end)], c='black')

            if y_min or y_max:
                plt.ylim(y_min, y_max)
            plt.xlabel("sec")
            plt.ylabel("µV")
            plt.title(self.recording[ch_id][0])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fig_nb += 1

        plt.show()
        plt.close()


    def plot_raw_compressed(self, ch_to_display, t_start, t_end, y_min, y_max, visualisation, artificial_noise, n_std, seed):
        # TODO: plot in lines or in MEA shape
        ch_to_display = check_ch_to_display(self.Info, ch_to_display)
        plt.rcdefaults()

        t_last_event = 0
        if t_end == "all":
            # get the latest event from all channels to display
            for ch in ch_to_display:
                for idx in range(self.Info.get_nb_channel()):
                    if self.recording[idx][0] == ch:
                        if self.recording[idx][1] and t_last_event < self.recording[idx][2][len(self.recording[idx][2])-1][1]:
                            t_last_event = self.recording[idx][2][len(self.recording[idx][2])-1][1]
                        break
            t_end = (t_last_event + 0.1 * self.Info.get_sampling_rate() ) / self.Info.get_sampling_rate()

        fig = plt.figure()

        fig_nb = 1
        for ch_nb in ch_to_display:
            ch_id = 0
            for idx in range(len(self.recording)):
                if self.recording[idx][0] == ch_nb:
                    ch_id = idx
                    break
            # create new subplot
            ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)

            if visualisation == "aligned":
                self.plot_raw_compressed_a(t_start, t_end, ch_nb)
            if visualisation == "reconstructed":
                self.plot_raw_compressed_r(t_start, t_end, ch_nb, artificial_noise, n_std, seed)
            if visualisation == "continuous" or visualisation == "superimposed":
                self.plot_raw_compressed_c_s(visualisation, t_start, t_end, ch_id)

            if y_min or y_max:
                plt.ylim(y_min, y_max)
            plt.title(self.recording[ch_id][0])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fig_nb += 1

        plt.show()
        plt.close()


    def plot_raw_compressed_a(self, t_start, t_end, ch_nb, plot_zeros=False, artificial_noise=False, n_std=15, seed=0):
        ch_id = 0
        for idx in range(len(self.recording)):
            if self.recording[idx][0] == ch_nb:
                ch_id = idx
                break

        frame_end = int(t_start*self.Info.get_sampling_rate())
        snip_stop = 0
        for snip_id in range(len(self.recording[ch_id][2])):
            frame_start = self.recording[ch_id][2][snip_id][0]
            snip_start = snip_stop
            snip_stop = snip_start + self.recording[ch_id][2][snip_id][1] - self.recording[ch_id][2][snip_id][0]

            if self.recording[ch_id][2][snip_id][1] < t_end * self.Info.get_sampling_rate() and frame_start > t_start * self.Info.get_sampling_rate():
                if artificial_noise:
                    # add artificial noise before this snippet
                    plt.plot([x/self.Info.get_sampling_rate() for x in range(frame_end, frame_start)], [np.random.normal(0, n_std) for y in range(frame_start - frame_end)], c='black')
                if plot_zeros:
                    # add horizontal line at y=0 before this snippet
                    plt.hlines(y=0, xmin=frame_end/self.Info.get_sampling_rate(), xmax=frame_start/self.Info.get_sampling_rate(), color='black')
                # plot snippet
                plt.plot([x/self.Info.get_sampling_rate() for x in range(self.recording[ch_id][2][snip_id][0], self.recording[ch_id][2][snip_id][1])], self.recording[ch_id][1][snip_start:snip_stop], c='black')
                frame_end = self.recording[ch_id][2][snip_id][1]

        if artificial_noise:
            # add artificial noise from last snippet to t_end
            plt.plot([x/self.Info.get_sampling_rate() for x in range(frame_end, int(t_end*self.Info.get_sampling_rate()))], [np.random.normal(0, n_std) for y in range(frame_end, int(t_end*self.Info.get_sampling_rate()))], c='black')
        if plot_zeros:
            # add 0 data from last snippet to t_end
            plt.hlines(y=0, xmin=frame_end/self.Info.get_sampling_rate(), xmax=t_end, color='black')

        plt.xlim(t_start, t_end)
        plt.xlabel("sec")
        plt.ylabel("µV")


    def plot_raw_compressed_r(self, t_start, t_end, ch_nb, artificial_noise, n_std, seed):
        # plot raw_compressed data in sec with 0 or artificial noise between snippets
        if artificial_noise:
            self.plot_raw_compressed_a(t_start, t_end, ch_nb, False, artificial_noise, n_std, seed)
        else:
            self.plot_raw_compressed_a(t_start, t_end, ch_nb, plot_zeros=True)


    def plot_raw_compressed_c_s(self, visualisation, t_start, t_end, ch_id):
        snip_stop = 0
        temps=[]
        for snip_id in range(len(self.recording[ch_id][2])):
            if visualisation == "superimposed":
                snip_start = snip_stop
                snip_stop = snip_start + self.recording[ch_id][2][snip_id][1] - self.recording[ch_id][2][snip_id][0]
            if self.recording[ch_id][2][snip_id][1] < t_end * self.Info.get_sampling_rate() and self.recording[ch_id][2][snip_id][0] > t_start * self.Info.get_sampling_rate():
                if visualisation == "superimposed":
                    # plot superimposed snippets
                    plt.plot(self.recording[ch_id][1][snip_start:snip_stop], label="spike "+ str(snip_id), c='black')
                else:
                    snip_start = snip_stop
                    snip_stop = snip_start + self.recording[ch_id][2][snip_id][1] - self.recording[ch_id][2][snip_id][0]
                    temps += self.recording[ch_id][1][snip_start:snip_stop]
                    # # plot line between snippets for continuous visu
                    plt.axvline(snip_stop, c='grey')

        if visualisation == "continuous":
            plt.plot(temps, c='black')

        plt.xlabel("frame")
        plt.ylabel("µV")


    def plot_mea(self, ch_to_display="all", label=[], background=False, flip=False):
        """
        Plot the position of specified electrodes on a representation of the MEA.

        Parameters
        ----------
        ch_to_display: list of int, or "all"
            list of electrode to display
        label: list of int
            if the list is not empty, will display the electrode number in red, above the specified electrode
        background: Bool
            if True, plot all electrodes in gray as a background
        flip: Bool
            if True, flip the array to match the way it is displayed on the BrainWave software (electrode 0,0 top lef)
        """
        ch_to_display = check_ch_to_display(self.Info, ch_to_display)
        plt.rcdefaults()
        # electrodes are 60um away, so x coord * 60 and y coord * 60 to get their position is space

        # if background is True, plot all electrodes in gray
        if background:
            x_coords = []
            y_coords = []
            for i in range(64):
                for j in range(64):
                    x_coords.append(i*60)
                    y_coords.append(j*60)
            plt.scatter(x_coords, y_coords, marker="s", s=1, c="silver")

        if len(ch_to_display) == 0:
            print("No channel to display.")
            return

        for ch_id in range(len(self.recording)):
            ch_nb = self.recording[ch_id][0]
            if ch_to_display=="all" or ch_nb in ch_to_display:
                ch_coord = get_ch_coord(ch_nb)
                if flip:
                    # flip map to match BrainWave visualisation (0,0 top left)
                    plt.scatter(ch_coord[1]*60, -ch_coord[0]*60, marker="s", s=1, c='red')
                else:
                    plt.scatter(ch_coord[0]*60, ch_coord[1]*60, marker="s", s=1, c='red')
                if ch_nb in label:
                    plt.text(ch_coord[0]*60, ch_coord[1]*60, ch_nb)

        plt.gca().set_aspect('equal')
        plt.xlim(0, 3839)
        plt.ylim(0, 3839)
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.show()
        plt.close()


    def plot_activity_map(self, label=[], t_start=0, t_end="all", method="std", threshold=-100, min_range=None, max_range=None, cmap='viridis', save_path=False, flip=False):
        """
        Plot the activity map (heatmap) for the specified time window.

        Parameters
        ----------
        label:  list of int
            list of channel to label on the activity map
        t_start: float
            first time point, in seconds
        t_end: float
            last time point, in seconds
        method: string
            set the method to use for the activity value calculation. can be "min", "max", "min-max" or "std". default value is "std"
        min_range: None or float
            set the vmin value of the pyplot scatter plot
        max_range: None or float
            set the vmax value of the pyplot scatter plot
        cmap: string
            set the cmap colours. Can for instance be viridis, plasma, magma, hot or gray
        save_path: None or string
            if not None, set the path and file name to save this activity plot as a png file
        flip: Bool
            if True, flip the activity map to match BrainWave visualisation (0,0 top left)
        """
        # TODO: more methods for activity map
        plt.rcdefaults()
        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = self.get_frame_start_end(t_start, t_end)

        # store x and y coordinates
        x_list = []
        y_list = []
        # store the activity value, used for the colour map
        intensity_list = []
        # for each channel in the recording
        for ch_id in range(len(self.recording)):
            rec = []
            if self.Info.recording_type == "RawDataSettings":
                rec = self.recording[ch_id][1][int(frame_start):int(frame_end)]
            if self.Info.recording_type == "NoiseBlankingCompressionSettings":
                snip_stop = 0
                for snip_id in range(len(self.recording[ch_id][2])):
                    if self.recording[ch_id][2][snip_id][1] > frame_end:
                        break
                    if self.recording[ch_id][2][snip_id][0] > frame_start:
                        snip_start = snip_stop
                        snip_stop = snip_start + self.recording[ch_id][2][snip_id][1] - self.recording[ch_id][2][snip_id][0]
                        rec += self.recording[ch_id][1][snip_start:snip_stop]

            # choose what function to use, depending on selected method. these fonctions are in utils.py
            val = 0
            if method == "min" or method == "max" or method == "min-max":
                val = ch_rec_min_max(rec, method)
            if method == "std":
                val = ch_rec_std(rec)
            if method == "spike_number":
                val = ch_rec_spikenb(rec, threshold)

            # get ch coord for this channel
            x, y = get_ch_coord(self.recording[ch_id][0])
            if flip:
                # flip map to match BrainWave visualisation (0,0 top left)
                x_list.append(y)
                y_list.append(-x)
            else:
                x_list.append(x)
                y_list.append(y)
            intensity_list.append(val)

        # cmap colours: viridis, plasma, magma, hot, gray
        plt.scatter(x_list, y_list, c=intensity_list, marker="s", cmap=cmap, vmin=min_range, vmax=max_range)
        plt.colorbar(label=method)
        # set the plot square
        plt.gca().set_aspect('equal')
        plt.xlim(0,63)
        plt.ylim(0,63)
        if flip:
            plt.ylim(-63,0)

        # add label for ch in label list
        for ch in label:
            if flip:
                x, y = get_ch_coord(ch)
                ch_coord = [y, -x]
            else:
                ch_coord = get_ch_coord(ch)
            plt.scatter(ch_coord[0], ch_coord[1], marker='s', s=1, c='red')
            plt.text(ch_coord[0], ch_coord[1], ch, c='red')

        if save_path:
            # save plot as a png file
            plt.savefig(save_path + ".png")

        plt.show()
        plt.close()


    # -------- processing -------- #

    def low_pass_filter(self, ch_to_process, highcut):
        if ch_to_process == "all":
            ch_to_process = [self.recording[ch_id][0] for ch_id in range(len(self.recording))]

        b, a = scipy.signal.butter(order, Wn, btype='lowpass')

        for ch_nb in ch_to_process:
            filtered_data = scipy.signal.filtfilt(b, a, signal)


    def high_pass_filter(self, ch_to_process, lowcut):
        # TODO
        return


    def band_pass_filter():
        # TODO
        return


    def down_sample(self, freq, ch_to_process, t_start=0, t_end="all", overwrite=False):
        """
        Perform a down sampling on the raw signal for the specified channels and time interval. Can either retun a down sample data, or overwrite the existing data.

        Parameters
        ----------
        freq: foat
            frequency to down sample the recording to
        ch_to_process: list of int
            list of channels to process
        t_start: float
            first time point, in seconds
        t_end: float
            last time point, in seconds
        overwrite: Bool
            if set to True, will overwrite the data of the Brw_Recording object that called this method
        """
        if freq > self.Info.get_sampling_rate():
            print("required resampling rate is higher than the initial sampling rate")
        if ch_to_process == "all":
            ch_to_process = [self.recording[ch_id][0] for ch_id in range(len(self.recording))]

        frame_start, frame_end = self.get_frame_start_end(t_start, t_end, ch_to_process)
        # number of sample to get from the initial data
        samps = int(np.ceil((frame_end/self.Info.get_sampling_rate() - frame_start/self.Info.get_sampling_rate()) * freq))

        if self.Info.recording_type == "RawDataSettings":
            down_sampled_data = []
            # for each channel to process
            for ch_nb in ch_to_process:
                # get the channel id corresponding to this channel number
                for ch_id in range(len(self.recording)):
                    if self.recording[ch_id][0] == ch_nb:
                        break
                # re sample this channel
                down_sampled_ch_data = signal.resample(self.recording[ch_id][1][frame_start:frame_end], samps)

                if overwrite:
                    # overwrite this channel data with new sampling data
                    self.recording[ch_id][1] = down_sampled_ch_data
                    # update the recording snippet frame_start and frame_end accordingly to new sampling rate
                    for rec_segment in range(len(self.recording[ch_id][2])):
                        self.recording[ch_id][2][rec_segment][0] = np.ceil(self.recording[ch_id][2][rec_segment][0] / (self.Info.get_sampling_rate()/ freq))
                        self.recording[ch_id][2][rec_segment][1] = np.floor(self.recording[ch_id][2][rec_segment][1] / (self.Info.get_sampling_rate()/ freq))
                    # update Info.sampling_rate
                    self.Info.sampling_rate = freq
                # if no overwrite
                else:
                    # add channel data to down_sampled_data list
                    down_sampled_data.append([ch_nb, down_sampled_ch_data, [int(np.floor(frame_start/self.Info.get_sampling_rate() * freq)), int(np.floor(frame_end/self.Info.get_sampling_rate() * freq))]])

            return down_sampled_data

        if self.Info.recording_type == "NoiseBlankingCompressionSettings":
            # TODO
            print("not implemented yet for NoiseBlankingCompression format.")
            return


    # -------- utils -------- #

    def get_frame_start_end(self, t_start, t_end, ch_list="all"):
        """
        get the first and last frame position number of a recording corresponding to the specified t_start and t_end
        """
        frame_start = 0
        frame_end = 0
        if self.Info.recording_type == "RawDataSettings":
            rec_frame_start = self.recording[0][2][0][0]
            rec_frame_end = self.recording[0][2][0][1]

            frame_start = t_start * self.Info.get_sampling_rate() - rec_frame_start
            if t_start * self.Info.get_sampling_rate() < rec_frame_start:
                frame_start = 0
            if frame_start > len(self.recording[0][1]):
                raise SystemExit("Requested start time of recording to display is higher than the recording length")

            if t_end == "all" or t_end * self.Info.get_sampling_rate() > rec_frame_end:
                frame_end = rec_frame_end - rec_frame_start
            else:
                frame_end = t_end * self.Info.get_sampling_rate() - rec_frame_start

        elif self.Info.recording_type == "NoiseBlankingCompressionSettings":
            # set frame_start
            frame_start = t_start * self.Info.get_sampling_rate()

            if t_end == "all":
                # it t_end is all the recording, set frame_end to last frame of the recording
                frame_end = self.Info.get_recording_length()

            # if t_end is not the end of the recording, have to find the last event before t_end
            else:
                if ch_list == "all":
                    ch_list = [i for i in range(4096)]
                # will store the frame time of the last event before t_end
                t_last_event = 0
                for idx in range(len(self.recording)):
                    if len(self.recording[idx][2]) != 0 and t_last_event < self.recording[idx][2][len(self.recording[idx][2])-1][1] and self.recording[idx][0] in ch_list:
                        t_last_event = self.recording[idx][2][len(self.recording[idx][2])-1][1]
                frame_end = t_last_event

        return int(frame_start), int(frame_end)

# ---------------------------------------------------------------- #
def read_brw_recording(file_path, t_start=0, t_end=60, ch_to_extract=[], frame_chunk=100000,
                       multiproc=False, verbose=False, use_dll=False, dll_path="C:\\Program Files\\3Brain\\BrainWave 5"):
    """
    Extract data from a brw file and return a Brw_Recording object.

    Parameters
    ----------
    file_path: String
    t_start: float
        first time point, in seconds
    t_end: float
        last time point, in seconds
    ch_to_extract: list of int
        channels to extract
    frame_chunk: int
        set the chunk size of data to read at a time. Large frame_chunk require more memory.
    multiproc: Bool
        WIP, not implemented yet. it set to True, use multiprocessing to read raw data
    verbose: Bool
        if set to True, will print more information during read operation
    use_dll: Bool
        if set to True, will use 3Brain dll during read operation
    dll_path: String
        set the path of 3Brain dll. Default value, is "C:\\Program Files\\3Brain\\BrainWave 5"
    """
    # check if ch_to_extract is a single int
    if type(ch_to_extract) == int:
        # if it is, raise SystemExit
        raise SystemExit("Error in \'" + inspect.stack()[1].function + "\' function: ch_to_extract is an integer. Please, enter the channel(s) to extract as a list of integers")

    Recording = Brw_Recording(file_path)
    Recording.read(t_start, t_end, ch_to_extract, frame_chunk, multiproc, verbose, use_dll, dll_path)

    return Recording


def load_recording(file_path):
    """
    Load a Brw_Recording object from a pickle file.

    Parameters
    ----------
    file_path: string
        path of the pickle file
    """
    with open(file_path, "rb") as file_handler:
        return pickle.load(file_handler)
