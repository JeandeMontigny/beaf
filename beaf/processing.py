import numpy as np
from scipy import signal
from .utils import *

# ---------------------------------------------------------------- #
# TODO
def low_pass_filter(File, ch_to_process, highcut):
    if ch_to_process == "all":
        ch_to_process = [File.recording[ch_id][0] for ch_id in range(0, len(File.recording))]

    b, a = scipy.signal.butter(order, Wn, btype='lowpass')

    for ch_nb in ch_to_process:
        filtered_data = scipy.signal.filtfilt(b, a, signal)

def high_pass_filter(File, ch_to_process, lowcut):
    return

def band_pass_filter():
    return

def down_sample(File, freq, ch_to_process, t_start=0, t_end="all", overwrite=False):
    if freq > File.info.get_sampling_rate():
        print("required resampling rate is higher than the initial sampling rate")
    if ch_to_process == "all":
        ch_to_process = [File.recording[ch_id][0] for ch_id in range(0, len(File.recording))]

    frame_start, frame_end = get_frame_start_end(File, t_start, t_end, ch_to_process)
    # number of sample to get from the initial data
    samps = int((frame_end/File.info.get_sampling_rate() - frame_start/File.info.get_sampling_rate()) * freq)

    if File.info.recording_type == "RawDataSettings":
        down_sampled_data = []
        for ch_nb in ch_to_process:
            for ch_id in range(0, len(File.recording)):
                if File.recording[ch_id][0] == ch_nb:
                    break
            # re sample this channel
            down_sampled_ch_data = signal.resample(File.recording[ch_id][1][frame_start:frame_end], samps)
            if overwrite:
                File.recording[ch_id][1] = down_sampled_ch_data
                for rec_segment in range(0, len(File.recording[ch_id][2])):
                    File.recording[ch_id][2][rec_segment][0] = np.floor(File.recording[ch_id][2][rec_segment][0] / (File.info.get_sampling_rate()/ freq))
                    File.recording[ch_id][2][rec_segment][1] = np.floor(File.recording[ch_id][2][rec_segment][1] / (File.info.get_sampling_rate()/ freq))
                # TODO: protected attributes, need set_sampling_rate()
                File.info.sampling_rate = freq
            else:
                down_sampled_data.append([ch_nb, down_sampled_ch_data, [int(np.floor(frame_start/File.info.get_sampling_rate() * freq)), int(np.floor(frame_end/File.info.get_sampling_rate() * freq))]])

        return down_sampled_data

    if File.info.recording_type == "NoiseBlankingCompressionSettings":
        # TODO
        print("not implemented yet.")
        return

# def sorting_extractor(unit_ids):
#     # TODO
#     ss.NumpySorting(unit_ids)
