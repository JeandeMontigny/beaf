import numpy as np
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from probeinterface import Probe


# ---------------------------------------------------------------- #
def get_ch_number(x, y):
    ch_nb = x * 64 + y % 64
    return ch_nb


def get_ch_coord(ch_nb):
    x = ch_nb // 64
    y = ch_nb % 64

    return x, y


def get_frame_start_end(File, t_start, t_end, ch_list="all"):
    frame_start = 0
    frame_end = 0
    if File.info.recording_type == "RawDataSettings":
        rec_frame_start = File.recording[0][2][0][0]
        rec_frame_end = File.recording[0][2][0][1]

        frame_start = t_start * File.info.get_sampling_rate() - rec_frame_start
        if t_start * File.info.get_sampling_rate() < rec_frame_start:
            frame_start = 0
        if frame_start > len(File.recording[0][1]):
            raise SystemExit("Requested start time of recording to display is higher than the recording length")

        if t_end == "all" or t_end * File.info.get_sampling_rate() > rec_frame_end:
            frame_end = rec_frame_end - rec_frame_start
        else:
            frame_end = t_end * File.info.get_sampling_rate() - rec_frame_start

    elif File.info.recording_type == "NoiseBlankingCompressionSettings":
        frame_start = t_start * File.info.get_sampling_rate()
        if t_end == "all":
            if ch_list == "all":
                ch_list = [i for i in range(0, 4096)]
            t_last_event = 0
            for idx in range(0, len(File.recording)):
                if len(File.recording[idx][2]) != 0 and t_last_event < File.recording[idx][2][len(File.recording[idx][2])-1][1] and File.recording[idx][0] in ch_list:
                    t_last_event = File.recording[idx][2][len(File.recording[idx][2])-1][1]
            frame_end = t_last_event
        else:
            frame_end = t_end * File.info.get_sampling_rate()
    return int(frame_start), int(frame_end)


def get_reconstructed_raw_compressed(File, t_start, t_end, ch_to_reconstruct, artificial_noise=False):
    reconst_raw_comp = []

    for ch in ch_to_reconstruct:
        ch_raw_reconst = get_reconstructed_ch_raw_compressed(File, t_start, t_end, ch, artificial_noise)
        reconst_raw_comp.append([[ch], ch_raw_reconst, [t_start, t_end]])

    return reconst_raw_comp


def get_reconstructed_ch_raw_compressed(File, t_start, t_end, ch_nb, artificial_noise, n_std, seed=0):
    np.random.seed(seed)
    ch_id = 0
    for idx in range(0, len(File.recording)):
        if File.recording[idx][0] == ch_nb:
            ch_id = idx
            break

    ch_raw_reconst = []
    frame_end = int(t_start*File.info.get_sampling_rate())
    snip_stop = 0
    for snip_id in range(0, len(File.recording[ch_id][2])):
        frame_start = File.recording[ch_id][2][snip_id][0]
        snip_start = snip_stop
        snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]

        if File.recording[ch_id][2][snip_id][1] < t_end * File.info.get_sampling_rate() and frame_start > t_start * File.info.get_sampling_rate():
            if artificial_noise:
                # add artificial noise between recordings
                ch_raw_reconst += [np.random.normal(0, n_std) for i in range(frame_start - frame_end)]
            else:
                # add 0 values between recordings
                ch_raw_reconst += [0 for i in range(frame_start - frame_end)]
            #  add snippet
            ch_raw_reconst += File.recording[ch_id][1][snip_start:snip_stop]
            frame_end = File.recording[ch_id][2][snip_id][1]

    if artificial_noise:
        # add artificial noise from last snippet to t_end
        ch_raw_reconst += [np.random.normal(0, n_std) for i in range(frame_end, int(t_end*File.info.get_sampling_rate()))]
    else:
        # add 0 data from last snippet to t_end
        ch_raw_reconst += [0 for i in range(frame_end, int(t_end*File.info.get_sampling_rate()))]

    return ch_raw_reconst


def get_spikeinterface_struct(File, t_start=0, t_end="all", ch_to_extract="all", reconstruct=True, artificial_noise=True, n_std=15, seed=0):
    # WARNING: potentlal memory issues. File and NR object exist at the same time (+ traces_list)
    # TODO: option to create a RecordingExtractor segment for each snippet in NoiseBlankingCompressionSettings
    #       ie, create NR with non reconstructed data, but continuous raw_compressed data
    #       using NumpyRecordingSegment(traces, sampling_frequency, t_start)?
    if ch_to_extract == "all":
        ch_to_extract = [File.recording[ch_id][0] for ch_id in range(0, len(File.recording))]

    if File.info.recording_type == "RawDataSettings":
        frame_start, frame_end = get_frame_start_end(File, t_start, t_end, ch_to_extract)

    traces_list = []
    geom = []
    for ch_nb in ch_to_extract:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0] == ch_nb:
                ch_id = idx
                break

        ch_coord = get_ch_coord(File.recording[ch_id][0])
        if File.info.recording_type == "RawDataSettings":
            traces_list.append(File.recording[ch_id][1][frame_start:frame_end])
        if File.info.recording_type == "NoiseBlankingCompressionSettings":
            if reconstruct:
                traces_list.append(get_reconstructed_ch_raw_compressed(File, t_start, t_end, ch_nb, artificial_noise, n_std))
            else:
                # continuous raw_compressed data
                # WARNING: problem with recording not of the same length
                # TODO: solution using RecordingSegment?
                if len(ch_to_extract) > 1:
                    print("NumpyRecording object is not supported yet for more than one channel without signal reconstruction")
                    return
                snip_stop = 0
                temps = []
                frame_end = t_end * File.info.get_sampling_rate()
                frame_start = t_start * File.info.get_sampling_rate()
                for snip_id in range(0, len(File.recording[ch_id][2])):
                    if File.recording[ch_id][2][snip_id][1] < frame_end and File.recording[ch_id][2][snip_id][0] > frame_start:
                        snip_start = snip_stop
                        snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]
                        temps += File.recording[ch_id][1][snip_start:snip_stop]
                traces_list.append(temps)

        geom.append([ch_coord[0]*60, ch_coord[1]*60])

    NR = se.NumpyRecording(traces_list=np.transpose(traces_list), sampling_frequency=File.info.get_sampling_rate(), channel_ids=ch_to_extract)

    # create and attach probe
    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=geom, shapes='square', shape_params={'width': 21})
    square_contour = [(-60, -60), (3900, -60), (3900, 3900), (-60, 3900)]
    probe.set_planar_contour(square_contour)
    # WARNING: device_channel_indices does not match channel number
    probe.set_device_channel_indices(range(len(ch_to_extract)))
    NR = NR.set_probe(probe)

    return NR
