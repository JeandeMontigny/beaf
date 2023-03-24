import inspect, psutil
import numpy as np

# ---------------------------------------------------------------- #
def get_ch_number(x, y, one_one_origin=True):
    if one_one_origin:
        if y == 64:
            return (x-1) * 64 + y % 64 + 64
        else:
            return (x-1) * 64 + y % 64
    else :
        return x * 64 + y % 64


def get_ch_coord(ch_nb, one_one_origin=True):
    x = ch_nb // 64
    y = ch_nb % 64
    if one_one_origin:
        x = x + 1
    return x, y


def get_file_frame_start_end(Info, t_start, t_end, frame_chunk=100000):
    frame_start =  int(np.floor(t_start * Info.get_sampling_rate()))
    frame_end = int(np.floor(t_end * Info.get_sampling_rate()))

    if Info.get_recording_type() == "RawDataSettings":
        # resize frame_chunk if it is larger than the recording length to extract
        if frame_chunk > Info.get_recording_length():
            frame_chunk = frame_end - frame_start

        if frame_start > Info.get_recording_length():
            raise SystemExit("Requested start time of recording to extract is higher than the recording length")
        # comparison in bit
        if  frame_chunk * Info.get_nb_channel() * 2 > psutil.virtual_memory().available:
            raise SystemExit("Memory size of the recording chunk to extract is bigger than your available system memory. Try again using a smaller frame_chunk value")

    if Info.get_recording_type == "NoiseBlankingCompressionSettings":
        hdf_file = h5py.File(Info.file_path,'r')
        toc = hdf_file.get("TOC")
        if frame_start > toc[len(toc)-1][1]:
            raise SystemExit("Requested start time of recording to extract is higher that the recording length")
        if frame_end > toc[len(toc)-1][1]:
            frame_end = self.Info.get_recording_length()

        hdf_file.close()

    return frame_start, frame_end


def convert_digital_to_analog(min_analog_value, value, x):
    digital_value = min_analog_value + value * x
    # clean saturated values
    if digital_value > 4095 or digital_value < -4095:
        digital_value = 0
    return digital_value


def check_ch_to_display(Info, ch_to_display):
    if ch_to_display=="all":
        ch_to_display = []
        for idx in Info.channel_idx:
            ch_to_display.append(idx)
    if type(ch_to_display) == int:
        raise SystemExit("Error in \'" + inspect.stack()[1].function + "\' function: ch_to_display is an integer. Please, enter the channel(s) to display as a list")

    return ch_to_display


def ch_rec_min_max(rec, method):
    min = 0
    max = 0
    if len(rec) == 0:
        return 0
    for val_id in range(0, len(rec)):
        if rec[val_id] < min:
            min = rec[val_id]
        if rec[val_id] > max:
            max = rec[val_id]
    if method == "min":
        val = min
    elif method == "max":
        val = max
    else:
        val = max - min
    return val


def ch_rec_std(rec):
    if len(rec) == 0:
        return 0
    std = np.std(rec)
    return std


def ch_rec_spike_nb(rec, threshold):
    # TODO: simple spike detection
    if len(rec) == 0:
        return 0
    nb_spike = 0
    return nb_spike
