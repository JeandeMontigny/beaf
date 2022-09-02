

def get_ch_number(x, y):
    ch_nb = x * 64 + y % 64
    return ch_nb


def get_ch_coord(ch_nb):
    x = ch_nb // 64
    y = ch_nb % 64
    if y == 0:
        y = 64

    return x, y


def get_frame_start_end(File, t_start, t_end, ch_to_display="all"):
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
            if ch_to_display == "all":
                ch_to_display = [i for i in range(0, 4096)]
            t_last_event = 0
            for idx in range(0, len(File.recording)):
                if File.recording[idx][0] in ch_to_display and len(File.recording[idx][2]) != 0 and t_last_event < File.recording[idx][2][len(File.recording[idx][2])-1][1]:
                    t_last_event = File.recording[idx][2][len(File.recording[idx][2])-1][1]
            frame_end = t_last_event
        else:
            frame_end = t_end * File.info.get_sampling_rate()
    return frame_start, frame_end


def get_reconstructed_raw_compressed(File, ch_to_reconstruct):
    # TODO: reconstruct data from raw_compressed format recording
    # TODO: fake background noise option
    return
