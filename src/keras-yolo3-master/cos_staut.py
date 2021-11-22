import scipy.stats as stats


def cos_staut(x):
    n = len(x)
    xx = x  # 因为需要删除，所以复制一份
    if n % 2 == 1:
        del xx[n // 2]
    c = n // 2

    # 计算正负符号的数量
    n_pos = n_neg = 0  # n_pos=S+  n_neg=S-
    for i in range(c):
        diff = xx[i + c] - x[i]
        if diff > 0:
            n_pos += 1
        elif diff < 0:
            n_neg += 1
        else:
            continue

    num = n_pos + n_neg
    k = min(n_pos, n_neg)  # 求K值
    p_value = 2 * stats.binom.cdf(k, num, 0.5)  # 计算p_value
    print('fall:{}, rise:{}, p-value:{}'.format(n_neg, n_pos, p_value))

    # p_value<0.05,零假设不成立
    if n_pos > n_neg and p_value < 0.05:
        return 0
    elif n_neg > n_pos and p_value < 0.05:
        return 1
    else:
        return 2
