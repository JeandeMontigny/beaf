import os, sys, h5py
import spikeinterface.extractors as se
from probeinterface import *

from .brw_recording import *

# TODO: inheritance for Brw_SpikeInterface and Brw_Recording to avoid code duplication?
# ---------------------------------------------------------------- #
class Brw_SpikeInterface:
    """
    Brw_SpikeInterface class to read and store .brw recordings datas and convert them to SpikeInterface NumpyRecording class
    """
    def read_raw_data_dll(self, brw_path, t_start, t_end, ch_to_extract, frame_chunk, dll_path):
        import clr
        clr.AddReference(os.path.join(dll_path, "3Brain.BrainWave.IO.dll"))
        clr.AddReference(os.path.join(dll_path, "3Brain.BrainWave.Common.dll"))

        from System import Int32, Double, Boolean
        from _3Brain.BrainWave.IO import BrwFile
        from _3Brain.BrainWave.Common import (MeaFileExperimentInfo, RawDataSettings, ExperimentType, MeaPlate)
        from _3Brain.Common import (MeaPlateModel, MeaChipRoi, MeaDataType)

        consumer = object()

        data = BrwFile.Open(brw_path)
        info = data.get_MeaExperimentInfo()
        Info = get_brw_experiment_setting(brw_path)
        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = get_file_frame_start_end(Info, t_start, t_end, frame_chunk)
        # get required number of frame chunk from frame_start to frame_end depending on frame_chunk size
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))

        data_chunk = []
        traces_list = []
        for chunk in range(nb_frame_chunk):
            # if this is the last chunk, needs to reduce the chunk size to read, to avoid reading beyond the end of the BRW-file stream
            if chunk == nb_frame_chunk-1:
                last_chunk = frame_end - int(frame_start + chunk * frame_chunk)
                data_chunk = data.ReadRawData(int(frame_start + chunk * frame_chunk), last_chunk, data.get_SourceChannels(), consumer)
            # ReadRawData returns a 3D array. first index is the well number (index 0 if single well),
            # second index is the channel, third index the time frame
            else:
                data_chunk = data.ReadRawData(int(frame_start + chunk * frame_chunk), frame_chunk, data.get_SourceChannels(), consumer)

            chunk_ch_data = []
            for ch_id in range(len(ch_to_extract)):
                # convert to voltage and add this chunk data at the end of this ch data array
                chunk_ch_data.append(np.fromiter(info.DigitalToAnalog(data_chunk[0][ch_to_extract[ch_id]]), float))

            # format data to match spikeinterface structure: [ [f0ch0, f0ch1, ...], [f1ch0, f1ch1, ...], ... ]
            # for each frame in this data chunk
            for frame_nb in range(len(chunk_ch_data[0])):
                # frame_data will contain each ch value for a single frame
                frame_data = []
                # for each ch to extract
                for ch_id in range(len(ch_to_extract)):
                    # add this ch value for this frame in frame_data list
                    frame_data.append(chunk_ch_data[ch_id][frame_nb])
                # add this frame_data to traces_list
                traces_list.append(frame_data)

        # Close Files
        data.Close()
        # create the NumpyRecording
        NR = se.NumpyRecording(traces_list=np.array(traces_list), sampling_frequency=Info.get_sampling_rate(), channel_ids=ch_to_extract)

        return NR


    def read_raw_data(self, brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk):
        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = get_file_frame_start_end(Info, t_start, t_end, frame_chunk)
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * Info.get_nb_channel()
        first_frame = frame_start * Info.get_nb_channel()
        last_frame = first_frame + id_frame_chunk

        # open hdf5 file
        hdf_file = h5py.File(brw_path,'r')
        # calculate the invariant term for digital to analog conversion
        converter_x = (Info.max_analog_value - Info.min_analog_value) / (Info.max_digital_value - Info.min_digital_value)

        traces_list = []
        # for each chunk needed to read the recording from t_start to t_end
        for chunk in range(0, nb_frame_chunk):
            # if this is the last chunk of data, need to update last_frame to be the index of last needed frame (avoid out of bound)
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * Info.get_nb_channel()

            # read data from first_frame to last_frame of this data chunk
            data_chunk = hdf_file.get("Well_A1").get("Raw")[first_frame:last_frame+Info.get_nb_channel()]

            # update first_frame and last_frame position for next data chunk
            first_frame += id_frame_chunk + Info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            # for each frame in this data chunk
            for frame_nb in range(0, int(len(data_chunk)/Info.get_nb_channel())):
                frame_data = []
                # get the position of the current frame
                frame_start_id = frame_nb*Info.get_nb_channel()

                # for each channel to extract
                for ch_id in range(0, len(ch_to_extract)):
                    # get the channel number
                    ch = ch_to_extract[ch_id]
                    # convert digital value to analog and add this frame for this channel to corresponding position in list
                    frame_data.append(convert_digital_to_analog(Info.min_analog_value, data_chunk[frame_start_id + ch - 1], converter_x))
                traces_list.append(frame_data)

        hdf_file.close()
        del(hdf_file)

        # convert frame_data list to numpy array and create a SpikeInterface NumpyRecording object using frame_data as data
        NR = se.NumpyRecording(traces_list=np.array(traces_list), sampling_frequency=Info.get_sampling_rate(), channel_ids=ch_to_extract)

        return NR


    def read_raw_compressed_data(self, brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk):
        # TODO: create spikeinterface NumpyRecording object
        #       problem with recording not of the same length
        #           solution using RecordingSegment? a RecordingExtractor segment for each snippet

        # data chunk [start-end[ in number of frame
        toc = self.data.get("TOC")
        # data chunk start in number of element in EventsBasedSparseRaw list (EventsBasedSparseRaw[id])
        event_sparse_raw_toc = self.data.get("Well_A1").get("EventsBasedSparseRawTOC")
        frame_start, frame_end = get_file_frame_start_end(Info, t_start, t_end)

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


    def read(self, brw_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe, use_dll, dll_path):
        # get Brw_Experiment_Settings info for this brw file
        Info = get_brw_experiment_setting(brw_path)
        # t_end is "all", set t_end to recording length
        if t_end == "all": t_end = Info.get_recording_length_sec()
        # if all channels are to be extracted
        if len(ch_to_extract) == 0 or ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)

        # check recording type and use the relevant method to read data
        if Info.get_recording_type() == "RawDataSettings":
            if use_dll:
                NR = self.read_raw_data_dll(brw_path, t_start, t_end, ch_to_extract, frame_chunk, dll_path)
            else:
                NR = self.read_raw_data(brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk)
        if Info.get_recording_type() == "NoiseBlankingCompressionSettings":
            NR = self.read_raw_compressed_data(brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk)

        # if true, create a SpikeInterface probe representing the MEA geometry (electrodes position)
        if attach_probe:
            geom = []
            for ch_nb in ch_to_extract:
                ch_coord = get_ch_coord(ch_nb)
                # electrodes are 60um away, so x coord * 60 and y coord * 60 to get their position is space
                geom.append([ch_coord[0]*60, ch_coord[1]*60])

            # create and attach probe using previously defined geom
            probe = Probe(ndim=2, si_units='um')
            probe.set_contacts(positions=geom, shapes='square', shape_params={'width': 21})
            square_contour = [(-60, -60), (3900, -60), (3900, 3900), (-60, 3900)]
            probe.set_planar_contour(square_contour)
            # WARNING: device_channel_indices does not match channel number
            probe.set_device_channel_indices(range(len(ch_to_extract)))
            NR = NR.set_probe(probe)

        return NR


# ---------------------------------------------------------------- #
def read_brw_SpikeInterface(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000,
                            attach_probe=True, use_dll=False, dll_path="C:\\Program Files\\3Brain\\BrainWave 5"):
    """
    Extract data from a brw file and return a SpikeInterface NumpyRecording object.

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
    attach_probe: Bool
        if set to true, create a SpikeInterface probe representing the MEA geometry
    use_dll: Bool
        if set to True, will use 3Brain dll during read operation
    dll_path: String
        set the path of 3Brain dll. Default value, is "C:\\Program Files\\3Brain\\BrainWave 5"
    """
    BNR = Brw_SpikeInterface()
    NR = BNR.read(file_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe, use_dll, dll_path)
    return NR


def get_spikeinterface_recording(Brw_Rec, t_start=0, t_end="all", ch_to_extract="all"):
    """
    Convert a Brw_Recording to a SpikeInterface NumpyRecording object.

    Parameters
    ----------
    Brw_Rec: Brw_Recording class
    t_start: float
        first time point, in seconds
    t_end: float
        last time point, in seconds
    ch_to_extract: list of int
        channels to extract
    """
    # WARNING: potentlal memory issues. Brw_Rec and NR object exist at the same time + traces_list
    #          Brw_Rec is only read, not modified, so should be passed as a reference and not as a copy. need to check
    # TODO: option to create a RecordingExtractor segment for each snippet in NoiseBlankingCompressionSettings
    #       ie, create NR with non reconstructed data, but continuous raw_compressed data
    #       using NumpyRecordingSegment(traces, sampling_frequency, t_start)?

    # if all channels are to be extracted
    if ch_to_extract == "all":
        ch_to_extract = [Brw_Rec.recording[ch_id][0] for ch_id in range(0, len(Brw_Rec.recording))]

    if t_end == "all":
        t_end = Brw_Rec.Info.get_recording_length_sec()

    if Brw_Rec.Info.recording_type == "RawDataSettings":
        # get first and last frame corresponding to t_start and t_end
        frame_start, frame_end = Brw_Rec.get_frame_start_end(t_start, t_end, ch_to_extract)

    # store raw data
    traces_list = []
    # SpikeInterface Probe geometry (eletrodes position)
    geom = []
    # for each channel to extract
    for ch_nb in ch_to_extract:
        ch_id = 0
        for idx in range(0, len(Brw_Rec.recording)):
            if Brw_Rec.recording[idx][0] == ch_nb:
                ch_id = idx
                break
        # get this channel x y coordinates from its number
        ch_coord = get_ch_coord(Brw_Rec.recording[ch_id][0])

        if Brw_Rec.Info.recording_type == "RawDataSettings":
            traces_list.append(Brw_Rec.recording[ch_id][1][frame_start:frame_end])

        # TODO: use RecordingSegment for NoiseBlankingCompressionSettings recordings
        if Brw_Rec.Info.recording_type == "NoiseBlankingCompressionSettings":
            # continuous raw_compressed data
            # WARNING: problem with recording not of the same length
            if len(ch_to_extract) > 1:
                print("NumpyRecording object is not supported yet for more than one channel")
                return
            snip_stop = 0
            temps = []
            frame_end = t_end * Brw_Rec.Info.get_sampling_rate()
            frame_start = t_start * Brw_Rec.Info.get_sampling_rate()
            for snip_id in range(0, len(Brw_Rec.recording[ch_id][2])):
                if Brw_Rec.recording[ch_id][2][snip_id][1] < frame_end and Brw_Rec.recording[ch_id][2][snip_id][0] > frame_start:
                    snip_start = snip_stop
                    snip_stop = snip_start + Brw_Rec.recording[ch_id][2][snip_id][1] - Brw_Rec.recording[ch_id][2][snip_id][0]
                    temps += Brw_Rec.recording[ch_id][1][snip_start:snip_stop]
            traces_list.append(temps)

        # electrodes are 60um away, so x coord * 60 and y coord * 60 to get their position is space
        geom.append([ch_coord[0]*60, ch_coord[1]*60])

    # transpose traces_list list to match NumpyRecording required shape and create a SpikeInterface NumpyRecording
    NR = se.NumpyRecording(traces_list=np.transpose(traces_list), sampling_frequency=Brw_Rec.Info.get_sampling_rate(), channel_ids=ch_to_extract)

    # create and attach probe using previously defined geom
    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=geom, shapes='square', shape_params={'width': 21})
    square_contour = [(-60, -60), (3900, -60), (3900, 3900), (-60, 3900)]
    probe.set_planar_contour(square_contour)
    # WARNING: device_channel_indices does not match channel number
    probe.set_device_channel_indices(range(len(ch_to_extract)))
    NR = NR.set_probe(probe)

    return NR
