import matplotlib.pyplot as plt
import inspect
from .read_file import *
from .utils import *

# ---------------------------------------------------------------- #
def plot_raw(File, ch_to_display, t_start=0, t_end="all", visualisation="reconstructed"):
    # distribtue to sub functions plot_raw_format or plot_raw_compressed depending on recording format
    if File.info.recording_type == "RawDataSettings":
        plot_raw_format(File, ch_to_display, t_start, t_end)
    if File.info.recording_type == "NoiseBlankingCompressionSettings":
        plot_raw_compressed(File, ch_to_display, t_start, t_end, visualisation)


def plot_raw_format(File, ch_to_display, t_start=0, t_end="all"):
# TODO: plot in lines or in MEA shape
    ch_to_display = check_ch_to_display(File.recording, ch_to_display)
    plt.rcdefaults()

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

    fig = plt.figure()

    fig_nb = 1
    for ch in ch_to_display:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0] == ch:
                ch_id = idx
                break

        ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)
        plt.plot([x/File.info.get_sampling_rate() for x in range(int(frame_start+rec_frame_start), int(frame_end+rec_frame_start))], File.recording[ch_id][1][int(frame_start):int(frame_end)], c='black')

        plt.xlabel("sec")
        plt.ylabel("µV")
        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()


def plot_raw_compressed(File, ch_to_display, t_start=0, t_end="all", visualisation="reconstructed"):
    # TODO: plot in lines or in MEA shape
    ch_to_display = check_ch_to_display(File, ch_to_display)
    plt.rcdefaults()

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
    plt.rcdefaults()

    if background:
        x_coords = []
        y_coords = []
        for i in range(0, 64):
            for j in range(0, 64):
                x_coords.append(i)
                y_coords.append(j)
        plt.scatter(x_coords, y_coords, marker="s", s=1, c="silver")

    ch_list = []
    for ch in range(0, len(File.recording)):
        if ch_to_display=="all" or File.recording[ch][0] in ch_to_display:
            ch_list.append(File.recording[ch][0])
    if len(ch_list) == 0:
        print("No channel to display.")
        return

    for ch_id in ch_list:
        ch_coord = get_ch_coord(ch_id)
        plt.scatter(ch_coord[0], ch_coord[1], marker="s", s=1, c='red')
        if ch_id in label:
            plt.text(ch_coord[0], ch_coord[1], ch_id)

    plt.gca().set_aspect('equal')
    plt.xlim(0,64)
    plt.ylim(0,64)
    plt.show()


def plot_activity_map(File, label=[], t_start=0, t_end="all", method="std", min_range=False, max_range=False, cmap='viridis'):
    # activity map for specified time windows
    # TODO: display for selected t_start, t_end (within ch_rec_* functions)
    # TODO: more methods for activity map
    plt.rcdefaults()

    x_list = []
    y_list = []
    intensity_list = []
    for ch_id in range(0, len(File.recording)):
        val = 0
        if method == "min" or method == "max" or method == "min-max":
            val = ch_rec_min_max(File.recording[ch_id][1], method, t_start, t_end, min_range, max_range)
        if method == "std":
            val = ch_rec_std(File.recording[ch_id][1], t_start, t_end, min_range, max_range)

        x, y = get_ch_coord(File.recording[ch_id][0])
        x_list.append(x)
        y_list.append(y)
        intensity_list.append(val)

    # cmap colours: viridis, plasma, magma, hot, gray
    plt.scatter(x_list, y_list, c=intensity_list, marker="s", cmap=cmap)
    plt.colorbar(label=method)
    plt.gca().set_aspect('equal')
    plt.xlim(0,64)
    plt.ylim(0,64)

    for ch in label:
        ch_coord = get_ch_coord(ch)
        plt.scatter(ch_coord[0], ch_coord[1], marker='s', s=1, c='red')
        plt.text(ch_coord[0], ch_coord[1], ch, c='red')

    plt.show()

# ---------------------------------------------------------------- #
def check_ch_to_display(rec, ch_to_display):
    if ch_to_display=="all":
        ch_to_display = []
        for idx in range(0, len(rec)):
            ch_to_display.append(rec[idx][0])
    if type(ch_to_display) == int:
        raise SystemExit("Error in \'" + inspect.stack()[1].function + "\' function: ch_to_display is an integer. Please, enter the channel(s) to display as a list")

    return ch_to_display


def ch_rec_min_max(rec, method, t_start, t_end, min_range, max_range):
    # TODO: from t_start to t_end
    min = 0
    max = 0
    if len(rec) == 0:
        if min_range:
            return min_range
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

    if min_range and val < min_range:
        val = min_range
    if max_range and val > max_range:
        val = max_range
    return val


def ch_rec_std(rec, t_start, t_end, min_range, max_range):
    # TODO: from t_start to t_end
    if len(rec) == 0:
        if min_range:
            return min_range
        return 0
    std = np.std(rec)
    if min_range and std < min_range:
        std = min_range
    if max_range and std > max_range:
        std = max_range
    return std
