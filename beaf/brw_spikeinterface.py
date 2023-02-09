import os, sys, h5py
import spikeinterface.extractors as se
from probeinterface import *

from .brw_recording import *

# TODO: inheritance for Brw_SpikeInterface and Brw_Recording to avoid code duplication?
# ---------------------------------------------------------------- #
class Brw_SpikeInterface:
    """
    TODO: description
    """
    def read_raw_data_recording(self, brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk):
        frame_start, frame_end = get_file_frame_start_end(Info, t_start, t_end, frame_chunk)
        nb_frame_chunk = int(np.ceil((frame_end - frame_start) / frame_chunk))
        id_frame_chunk = frame_chunk * Info.get_nb_channel()
        first_frame = frame_start * Info.get_nb_channel()
        last_frame = first_frame + id_frame_chunk

        hdf_file = h5py.File(brw_path,'r')
        converter_x = (Info.max_analog_value - Info.min_analog_value) / (Info.max_digital_value - Info.min_digital_value)

        traces_list = []
        for chunk in range(0, nb_frame_chunk):
            if chunk == nb_frame_chunk-1:
                last_frame = frame_end * Info.get_nb_channel()

            data_chunk = hdf_file.get("Well_A1").get("Raw")[first_frame:last_frame+Info.get_nb_channel()]

            first_frame += id_frame_chunk + Info.get_nb_channel()
            last_frame = first_frame + id_frame_chunk

            for frame_nb in range(0, int(len(data_chunk)/Info.get_nb_channel())):
                frame_data = []
                frame_start_id = frame_nb*Info.get_nb_channel()

                for ch_id in range(0, len(ch_to_extract)):
                    ch = ch_to_extract[ch_id]
                    frame_data.append(convert_digital_to_analog(Info.min_analog_value, data_chunk[frame_start_id + ch - 1], converter_x))
                traces_list.append(frame_data)

        hdf_file.close()

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


    def read(self, brw_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe):
        Info = get_brw_experiment_setting(brw_path)

        if t_end == "all": t_end = Info.get_recording_length_sec()
        if ch_to_extract == "all":
            ch_to_extract = []
            for ch in range (0, 4096):
                ch_to_extract.append(ch)

        if Info.get_recording_type() == "RawDataSettings":
            NR = self.read_raw_data_recording(brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk)
        if Info.get_recording_type() == "NoiseBlankingCompressionSettings":
            NR = self.read_raw_compressed_data(brw_path, Info, t_start, t_end, ch_to_extract, frame_chunk)

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


# ---------------------------------------------------------------- #
def read_brw_SpikeInterface(file_path, t_start = 0, t_end = 60, ch_to_extract = [], frame_chunk = 100000, attach_probe=True):
    BNR = Brw_SpikeInterface()
    NR = BNR.read(file_path, t_start, t_end, ch_to_extract, frame_chunk, attach_probe)
    return NR


def get_spikeinterface_recording(Brw_Rec, t_start=0, t_end="all", ch_to_extract="all"):
    # WARNING: potentlal memory issues. Brw_Rec and NR object exist at the same time + traces_list
    #          Brw_Rec is only read, not modified, so should be passed as a reference and not as a copy. need to check
    # TODO: option to create a RecordingExtractor segment for each snippet in NoiseBlankingCompressionSettings
    #       ie, create NR with non reconstructed data, but continuous raw_compressed data
    #       using NumpyRecordingSegment(traces, sampling_frequency, t_start)?
    if ch_to_extract == "all":
        ch_to_extract = [Brw_Rec.recording[ch_id][0] for ch_id in range(0, len(Brw_Rec.recording))]

    if Brw_Rec.Info.recording_type == "RawDataSettings":
        frame_start, frame_end = Brw_Rec.get_frame_start_end(t_start, t_end, ch_to_extract)

    traces_list = []
    geom = []
    for ch_nb in ch_to_extract:
        ch_id = 0
        for idx in range(0, len(Brw_Rec.recording)):
            if Brw_Rec.recording[idx][0] == ch_nb:
                ch_id = idx
                break

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

        geom.append([ch_coord[0]*60, ch_coord[1]*60])

    NR = se.NumpyRecording(traces_list=np.transpose(traces_list), sampling_frequency=Brw_Rec.Info.get_sampling_rate(), channel_ids=ch_to_extract)

    # create and attach probe
    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=geom, shapes='square', shape_params={'width': 21})
    square_contour = [(-60, -60), (3900, -60), (3900, 3900), (-60, 3900)]
    probe.set_planar_contour(square_contour)
    # WARNING: device_channel_indices does not match channel number
    probe.set_device_channel_indices(range(len(ch_to_extract)))
    NR = NR.set_probe(probe)

    return NR
