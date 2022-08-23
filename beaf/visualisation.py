import matplotlib.pyplot as plt
from .read_file import *
from .utils import *

# ---------------------------------------------------------------- #
def plot_raw():
# TODO: plot for specified time windows and electrodes
#       plot in lines or in MEA shape
    return


def plot_raw_compressed(File, ch_to_display, visualisation="reconstructed", t_start=0, t_stop="all"):
# TODO: display for selected t_start, t_stop
#       plot in lines or in MEA shape
    if ch_to_display=="all":
        ch_to_display = []
        for idx in range(0, len(File.recording)):
            ch_to_display.append(File.recording[idx][0])
    if type(ch_to_display) == int:
        raise SystemExit("Error in `plot_raw_compressed` function: ch_to_display is an integer. Please, enter the channel(s) to display as a list")

    # if t_stop == "all":# or t_stop > `rec length`
    #     t_stop == `rec length`

    fig = plt.figure()

    fig_nb = 1
    for ch in ch_to_display:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0]== ch:
                ch_id = idx
                break
        # create new subplot
        ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)

        if visualisation == "reconstructed":
            # plot raw_compressed data in sec with 0 between snippets
            temps = []
            frame_end = int(t_start*File.info.get_sampling_rate())
            snip_stop = 0
            for snip in range(0, len(File.recording[ch_id][2])):
                frame_start = File.recording[ch_id][2][snip][0]
                snip_start = snip_stop
                snip_stop = snip_start + File.recording[ch_id][2][snip][1] - File.recording[ch_id][2][snip][0]
                temps += [0 for i in range(frame_start - frame_end)]
                temps += File.recording[ch_id][1][snip_start:snip_stop]
                frame_end = File.recording[ch_id][2][snip][1]
            # add 0 data from last snippet to t_stop
            temps += [0 for i in range(int(t_stop*File.info.get_sampling_rate()) - frame_end)]
            plt.plot([x/File.info.get_sampling_rate() for x in range(0, len(temps))], temps, c='black')
            plt.xlabel("sec")
            plt.ylabel("µV")

        else:
            if visualisation == "continuous":
                # plot raw_compressed continuously
                # TODO: display actual time on x axis
                plt.plot(File.recording[ch_id][1], c='black')

            snip_stop = 0
            for snip in range(0, len(File.recording[ch_id][2])):
                snip_start = snip_stop
                snip_stop = snip_start + File.recording[ch_id][2][snip][1] - File.recording[ch_id][2][snip][0]
                if visualisation == "continuous" and snip < len(File.recording[ch_id][2])-1:
                    # plot line between snippets for continuous visu
                    plt.axvline(snip_stop, c='grey')
                if visualisation == "superimposed":
                    # plot superimposed snippets
                    plt.plot(File.recording[ch_id][1][snip_start:snip_stop], label="spike "+ str(snip), c='black')
            plt.xlabel("frame")
            plt.ylabel("µV")

        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()

    return


def plot_mea(File, ch_to_display="all", label=[], background=False):
# TODO: Rotate figure to have 0,0 top left?
    if type(ch_to_display) == int:
        raise SystemExit("Error in `plot_raw_compressed` function: ch_to_display is an integer. Please, enter the channel(s) to display as a list")

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
