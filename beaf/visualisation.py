import matplotlib.pyplot as plt
import inspect
from .read_file import *
from .utils import *

# ---------------------------------------------------------------- #
def plot_raw(File, ch_to_display, t_start=0, t_end="all"):
# NOTE: t_start/t_end here are depending on File.recording length (the recording snippet that has been extracted using read_brw_file), not on the brw file recording length
# TODO: plot in lines or in MEA shape
    ch_to_display = check_ch_to_display(File.recording, ch_to_display)

    if t_start * File.info.get_sampling_rate() > len(File.recording[0][1]):
        raise SystemExit("Requested start time of recording to display is higher than the recording length")
    if t_end == "all":
        t_end = len(File.recording[0][1])/File.info.get_sampling_rate()
    if t_end * File.info.get_sampling_rate() > len(File.recording[0][1]):
        t_end = len(File.recording[0][1])/File.info.get_sampling_rate()

    frame_start = t_start * File.info.get_sampling_rate()
    frame_end = t_end * File.info.get_sampling_rate()

    fig = plt.figure()

    fig_nb = 1
    for ch in ch_to_display:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0] == ch:
                ch_id = idx
                break

        ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)
        plt.plot([x/File.info.get_sampling_rate() for x in range(int(frame_start), int(frame_end))], File.recording[ch_id][1][int(frame_start):int(frame_end)], c='black')

        plt.xlabel("sec")
        plt.ylabel("µV")
        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()


def plot_raw_compressed(File, ch_to_display, visualisation="reconstructed", t_start=0, t_end="all"):
# TODO: display for selected t_start, t_end
#       plot in lines or in MEA shape
    ch_to_display = check_ch_to_display(File, ch_to_display)

    t_last_event = 0
    if t_end == "all":
        # get the latest event from all channels to display
        for ch in ch_to_display:
            for idx in range(0, len(File.recording)):
                if File.recording[idx][0] == ch:
                    if t_last_event < File.recording[idx][2][len(File.recording[idx][2])-1][1]:
                        t_last_event = File.recording[idx][2][len(File.recording[idx][2])-1][1]
                    break
        t_end = (t_last_event + 0.1 * File.info.get_sampling_rate() ) / File.info.get_sampling_rate()

    fig = plt.figure()

    fig_nb = 1
    for ch in ch_to_display:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0] == ch:
                ch_id = idx
                break
        # create new subplot
        ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)

        if visualisation == "reconstructed":
            plot_raw_compressed_r(File, ch_to_display, t_start, t_end, ch_id)
        if visualisation == "continuous" or visualisation == "superimposed":
            plot_raw_compressed_c_s(File, ch_to_display, visualisation, t_start, t_end, ch_id)

        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()


def plot_raw_compressed_r(File, ch_to_display, t_start, t_end, ch_id):
    # plot raw_compressed data in sec with 0 between snippets
    temps = []
    frame_end = int(t_start*File.info.get_sampling_rate())
    snip_stop = 0
    for snip_id in range(0, len(File.recording[ch_id][2])):
        frame_start = File.recording[ch_id][2][snip_id][0]
        snip_start = snip_stop
        snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]

        if File.recording[ch_id][2][snip_id][1] < t_end * File.info.get_sampling_rate() and frame_start > t_start * File.info.get_sampling_rate():
            # add 0 values between recordings
            temps += [0 for i in range(frame_start - frame_end)]
            #  add snippet
            temps += File.recording[ch_id][1][snip_start:snip_stop]
            frame_end = File.recording[ch_id][2][snip_id][1]

    # add 0 data from last snippet to t_end
    temps += [0 for i in range(frame_end, int(t_end*File.info.get_sampling_rate()))]

    plt.plot([x/File.info.get_sampling_rate() + t_start for x in range(0, len(temps))], temps, c='black')
    plt.xlabel("sec")
    plt.ylabel("µV")


def plot_raw_compressed_c_s(File, ch_to_display, visualisation, t_start, t_end, ch_id):
    snip_stop = 0
    temps=[]
    for snip_id in range(0, len(File.recording[ch_id][2])):
        if visualisation == "superimposed":
            snip_start = snip_stop
            snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]
        if File.recording[ch_id][2][snip_id][1] < t_end * File.info.get_sampling_rate() and File.recording[ch_id][2][snip_id][0] > t_start * File.info.get_sampling_rate():
            if visualisation == "superimposed":
                # plot superimposed snippets
                plt.plot(File.recording[ch_id][1][snip_start:snip_stop], label="spike "+ str(snip_id), c='black')
            else:
                snip_start = snip_stop
                snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]
                temps += File.recording[ch_id][1][snip_start:snip_stop]
                # # plot line between snippets for continuous visu
                plt.axvline(snip_stop, c='grey')

    if visualisation == "continuous":
        plt.plot(temps, c='black')

    plt.xlabel("frame")
    plt.ylabel("µV")


def plot_mea(File, ch_to_display="all", label=[], background=False):
# TODO: Rotate figure to have 0,0 top left?
    ch_to_display = check_ch_to_display(File, ch_to_display)

    if background:
        x_coords = []
        y_coords = []
        for i in range(0, 64):
            for j in range(0, 64):
                x_coords.append(i)
                y_coords.append(j)
        plt.plot(x_coords, y_coords,'.', markersize=1, c="gray")

    ch_list = []
    for ch in range(0, len(File.recording)):
        if ch_to_display=="all" or File.recording[ch][0] in ch_to_display:
            ch_list.append(File.recording[ch][0])
    if len(ch_list) == 0:
        print("No channel to display.")
        return

    for ch_id in ch_list:
        ch_coord = get_ch_coord(ch_id)
        plt.plot(ch_coord[0], ch_coord[1], '.', c='darkred')
        if ch_id in label:
            plt.text(ch_coord[0], ch_coord[1], ch_id)

    plt.gca().set_aspect('equal')
    plt.xlim(0,64)
    plt.ylim(0,64)
    plt.show()


def plot_activity_map():
# TODO: _activity map for specified time windows, electrodes and frequency
    return


# ---------------------------------------------------------------- #
def check_ch_to_display(rec, ch_to_display):
    if ch_to_display=="all":
        ch_to_display = []
        for idx in range(0, len(rec)):
            ch_to_display.append(rec[idx][0])
    if type(ch_to_display) == int:
        raise SystemExit("Error in \'" + inspect.stack()[1].function + "\' function: ch_to_display is an integer. Please, enter the channel(s) to display as a list")

    return ch_to_display
