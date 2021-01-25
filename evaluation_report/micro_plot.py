"""
Statistical Evaluation piece of code

This piece of code provides plots per_request#1 and 2. As a user, you just
need to change the input file (micro_remote#1 or 2)

Author: Benyamin Eslami ( www.beneslami.com)

"""

from statistics import mean, variance, stdev
import matplotlib.pyplot as plt
import numpy as np


def main():
    contents = " "
    f = open("micro_remote#2", "r")
    if f.mode == "r":
        contents = f.read()
    data_list = contents.splitlines()
    integer_map: list[int] = []
    dist_dict = {}
    for i in range(len(data_list)):
        integer_map.append(int(data_list[i]))

    integer_map_mean = mean(integer_map)
    integer_map_var = variance(integer_map)
    integer_map_std = stdev(integer_map)

    dist_dict[integer_map[0]] = 1
    for i in range(1, len(integer_map)):
        if dist_dict.get(integer_map[i]) is None:
            dist_dict[integer_map[i]] = 1
        else:
            dist_dict[integer_map[i]] = dist_dict[integer_map[i]] + 1

    m = mean(list(dist_dict.keys()))

    exponential_dict = {}
    # exponential distribution calculation
    own_dict = {}
    for i in dist_dict.keys():
        own_dict[i] = dist_dict[i] / len(integer_map)
    for i in range(0, len(integer_map)):
        exponential_dict[integer_map[i]] = (1 / integer_map_mean) * np.exp(-(1 / integer_map_mean) * integer_map[i])

    # Poisson distribution calculation

    # Plot Section
    plt.figure(figsize=(10, 4), dpi=120)
    plt.subplots_adjust(left=0.1,
                        bottom=0.147,
                        right=0.8,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    # exponential distribution section
    plt.subplot(1, 2, 1)
    plt.plot(own_dict.keys(), own_dict.values(), 'ro', marker='D', label='observed probability')
    plt.plot(exponential_dict.keys(), exponential_dict.values(), 'g*', linewidth=6, label='exponential distribution')
    plt.title("Amount of time per 1 request (Exponential)")
    plt.xlabel("Time epoch (Microsecond)")
    plt.ylabel("probability")
    plt.legend(loc='best')

    # Dispersion section
    plt.subplots_adjust(left=0.1,
                        bottom=0.147,
                        right=0.97,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.subplot(1, 2, 2)
    plt.bar(list(dist_dict.keys()), list(dist_dict.values()), color='g')
    plt.xlim(0, 300)
    plt.title("The dispersion of the amount of \n time it takes to send/receive 1 request")
    plt.xlabel("Time epoch (Microsecond)")
    plt.ylabel("The number of time epochs\nto send/receive 1 request")
    xx = [0, 300]
    yy = [m, m]
    plt.plot(xx, yy, linewidth=6)
    plt.ylim(0, 3000)
    plt.show()


if __name__ == "__main__":
    main()
