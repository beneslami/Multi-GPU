from statistics import mean, variance, stdev
import matplotlib.pyplot as plt
import numpy as np


def main():
    contents = " "
    f = open("micro_remote#1", "r")
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

    exponential_dict = {}
    normal_dict = {}
    own_dict = {}
    for i in dist_dict.keys():
        own_dict[i] = dist_dict[i]
    for i in range(0, len(integer_map)):
        exponential_dict[integer_map[i]] = (1 / integer_map_mean) * np.exp(-(1 / integer_map_mean) * integer_map[i])
        normal_dict[integer_map[i]] = ((1/np.sqrt(2*np.pi))*integer_map_std)*np.exp(-(integer_map[i]-integer_map_mean)**2 / 2*integer_map_std)


    plt.plot(exponential_dict.keys(), exponential_dict.values(), 'g*')
    plt.plot(own_dict.keys(), own_dict.values(), 'r^')
    #plt.plot(normal_dict.keys(), normal_dict.values(), 'r-', data=None)
    plt.show()


if __name__ == "__main__":
    main()
