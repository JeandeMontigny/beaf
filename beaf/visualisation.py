import matplotlib.pyplot as plt
import inspect
from .read_file import *
from .utils import *

# WARNING: is File duplicated when used as an argument?
#          if yes, need to move all visualisation methods to Brw_File class definition
#          and more generally, all functions using Brw_File (recording) informatons

# ---------------------------------------------------------------- #
def plot_raw(File, ch_to_display, t_start=0, t_end="all", y_min=None, y_max=None, visualisation="aligned", artificial_noise=False, n_std=15, seed=0):
    # distribtue to sub functions plot_raw_format or plot_raw_compressed depending on recording format
    if File.info.recording_type == "RawDataSettings":
        plot_raw_format(File, ch_to_display, t_start, t_end, y_min, y_max)
    if File.info.recording_type == "NoiseBlankingCompressionSettings":
        plot_raw_compressed(File, ch_to_display, t_start, t_end, y_min, y_max, visualisation, artificial_noise, n_std, seed)


def plot_raw_format(File, ch_to_display, t_start, t_end, y_min, y_max):
# TODO: plot in lines or in MEA shape
    ch_to_display = check_ch_to_display(File.recording, ch_to_display)
    plt.rcdefaults()

    rec_frame_start = File.recording[0][2][0][0]
    rec_frame_end = File.recording[0][2][0][1]
    frame_start, frame_end = get_frame_start_end(File, t_start, t_end, ch_to_display)

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

        if y_min or y_max:
            plt.ylim(y_min, y_max)
        plt.xlabel("sec")
        plt.ylabel("µV")
        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()
    plt.close()


def plot_raw_compressed(File, ch_to_display, t_start, t_end, y_min, y_max, visualisation, artificial_noise, n_std, seed):
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
    for ch_nb in ch_to_display:
        ch_id = 0
        for idx in range(0, len(File.recording)):
            if File.recording[idx][0] == ch_nb:
                ch_id = idx
                break
        # create new subplot
        ax = fig.add_subplot(len(ch_to_display), 1, fig_nb)

        if visualisation == "aligned":
            plot_raw_compressed_a(File, t_start, t_end, ch_nb)
        if visualisation == "reconstructed":
            plot_raw_compressed_r(File, t_start, t_end, ch_nb, artificial_noise, n_std, seed)
        if visualisation == "continuous" or visualisation == "superimposed":
            plot_raw_compressed_c_s(File, visualisation, t_start, t_end, ch_id)

        if y_min or y_max:
            plt.ylim(y_min, y_max)
        plt.title(File.recording[ch_id][0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_nb += 1

    plt.show()
    plt.close()


def plot_raw_compressed_a(File, t_start, t_end, ch_nb, plot_zeros=False, artificial_noise=False, n_std=15, seed=0):
    ch_id = 0
    for idx in range(0, len(File.recording)):
        if File.recording[idx][0] == ch_nb:
            ch_id = idx
            break

    frame_end = int(t_start*File.info.get_sampling_rate())
    snip_stop = 0
    for snip_id in range(0, len(File.recording[ch_id][2])):
        frame_start = File.recording[ch_id][2][snip_id][0]
        snip_start = snip_stop
        snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]

        if File.recording[ch_id][2][snip_id][1] < t_end * File.info.get_sampling_rate() and frame_start > t_start * File.info.get_sampling_rate():
            if artificial_noise:
                # add artificial noise before this snippet
                plt.plot([x/File.info.get_sampling_rate() for x in range(frame_end, frame_start)], [np.random.normal(0, n_std) for y in range(frame_start - frame_end)], c='black')
            if plot_zeros:
                # add horizontal line at y=0 before this snippet
                plt.hlines(y=0, xmin=frame_end/File.info.get_sampling_rate(), xmax=frame_start/File.info.get_sampling_rate(), color='black')
            # plot snippet
            plt.plot([x/File.info.get_sampling_rate() for x in range(File.recording[ch_id][2][snip_id][0], File.recording[ch_id][2][snip_id][1])], File.recording[ch_id][1][snip_start:snip_stop], c='black')
            frame_end = File.recording[ch_id][2][snip_id][1]

    if artificial_noise:
        # add artificial noise from last snippet to t_end
        plt.plot([x/File.info.get_sampling_rate() for x in range(frame_end, int(t_end*File.info.get_sampling_rate()))], [np.random.normal(0, n_std) for y in range(frame_end, int(t_end*File.info.get_sampling_rate()))], c='black')
    if plot_zeros:
        # add 0 data from last snippet to t_end
        plt.hlines(y=0, xmin=frame_end/File.info.get_sampling_rate(), xmax=t_end, color='black')

    plt.xlim(t_start, t_end)
    plt.xlabel("sec")
    plt.ylabel("µV")


def plot_raw_compressed_r(File, t_start, t_end, ch_nb, artificial_noise, n_std, seed):
    # plot raw_compressed data in sec with 0 or artificial noise between snippets
    if artificial_noise:
        plot_raw_compressed_a(File, t_start, t_end, ch_nb, False, artificial_noise, n_std, seed)
    else:
        plot_raw_compressed_a(File, t_start, t_end, ch_nb, plot_zeros=True)


def plot_raw_compressed_c_s(File, visualisation, t_start, t_end, ch_id):
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
                x_coords.append(i*60)
                y_coords.append(j*60)
        plt.scatter(x_coords, y_coords, marker="s", s=1, c="silver")

    if len(ch_to_display) == 0:
        print("No channel to display.")
        return

    for ch_id in range(0, len(File.recording)):
        ch_nb = File.recording[ch_id][0]
        if ch_to_display=="all" or ch_nb in ch_to_display:
            ch_coord = get_ch_coord(ch_nb)
            plt.scatter(ch_coord[0]*60, ch_coord[1]*60, marker="s", s=1, c='red')
            if ch_nb in label:
                plt.text(ch_coord[0]*60, ch_coord[1]*60, ch_nb)

    plt.gca().set_aspect('equal')
    plt.xlim(0, 3840)
    plt.ylim(0, 3840)
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.show()
    plt.close()


def plot_activity_map(File, label=[], t_start=0, t_end="all", method="std", threshold=-100, min_range=None, max_range=None, cmap='viridis', save_path=False, flip=False):
    # activity map for specified time windows
    # TODO: more methods for activity map
    plt.rcdefaults()

    frame_start, frame_end = get_frame_start_end(File, t_start, t_end)

    x_list = []
    y_list = []
    intensity_list = []
    for ch_id in range(0, len(File.recording)):
        rec = []
        if File.info.recording_type == "RawDataSettings":
            rec = File.recording[ch_id][1][int(frame_start):int(frame_end)]
        if File.info.recording_type == "NoiseBlankingCompressionSettings":
            snip_stop = 0
            for snip_id in range(0, len(File.recording[ch_id][2])):
                if File.recording[ch_id][2][snip_id][1] > frame_end:
                    break
                if File.recording[ch_id][2][snip_id][0] > frame_start:
                    snip_start = snip_stop
                    snip_stop = snip_start + File.recording[ch_id][2][snip_id][1] - File.recording[ch_id][2][snip_id][0]
                    rec += File.recording[ch_id][1][snip_start:snip_stop]

        val = 0
        if method == "min" or method == "max" or method == "min-max":
            val = ch_rec_min_max(rec, method)
        if method == "std":
            val = ch_rec_std(rec)
        if method == "spike_number":
            val = ch_rec_spikenb(rec, threshold)

        x, y = get_ch_coord(File.recording[ch_id][0])
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
    plt.gca().set_aspect('equal')
    plt.xlim(0,64)
    plt.ylim(0,64)
    if flip:
        plt.ylim(-64,0)

    for ch in label:
        ch_coord = get_ch_coord(ch)
        plt.scatter(ch_coord[0], ch_coord[1], marker='s', s=1, c='red')
        plt.text(ch_coord[0], ch_coord[1], ch, c='red')

    if save_path:
        plt.savefig(save_path + ".png")

    plt.show()
    plt.close()


# ---------------------------------------------------------------- #
def check_ch_to_display(rec, ch_to_display):
    if ch_to_display=="all":
        ch_to_display = []
        for idx in range(0, len(rec)):
            ch_to_display.append(rec[idx][0])
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
