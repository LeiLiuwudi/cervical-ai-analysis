from entropy import EntroPy


def mse(seq, m, r, tau_list):
    return EntroPy.multiscale_entropy(seq, m, r, tau_list)


if __name__ == '__main__':
    print(mse([119, 12, 13, 24, 5, 1, 8, 5, 6, 7, 8, 9], 0, 0, 0))
