

def get_ch_number(x, y):
    ch_nb = x * 64 + y % 64
    return ch_nb


def get_ch_coord(ch_nb):
    x = ch_nb // 64
    y = ch_nb % 64
    if y == 0:
        y = 64

    return x, y
