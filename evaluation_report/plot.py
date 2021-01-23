"""
Statistical Evaluation piece of code

This piece of code provides plots request_number_per_sec1,2 and 3. As a user, you just
need to change the input file (remote#1,2 or 3)

Author: Benyamin Eslami ( www.beneslami.com)
"""

from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import poisson


def main():
    contents = " "
    xy_value = {}
    key = 0
    f = open("remote#3", "r")
    if f.mode == "r":
        contents = f.read()
    list1 = contents.splitlines()

    previous_time_slot = list1[0].split(" --> ")[1].split(":")[2]
    xy_value[key] = int(list1[0].split(" --> ")[0])

    for i in range(1, len(list1)):
        info = list1[i].split(" --> ")
        if len(info) != 0 and len(info[0]) != 0 and len(info[1]) != 0:
            count = int(info[0])
            time = info[1].split(":")[2]
            if time == previous_time_slot:
                xy_value[key] += count
            else:
                key = key + 1
                previous_time_slot = time
                xy_value[key] = count

    own_sample_space = {xy_value[0]: 1}
    sum_ = 0
    for i in range(1, len(xy_value)):
        sum_ += xy_value[i]
        if xy_value[i] not in own_sample_space.keys():
            own_sample_space[xy_value[i]] = 1
        else:
            own_sample_space[xy_value[i]] += 1
    sorted_own_sample_space = sorted(own_sample_space.items(), key=lambda x: x[0])

    data_mean = mean(xy_value.values())
    own_probability = {}
    poisson_probability = {}

    for i in range(len(sorted_own_sample_space)):
        own_probability[sorted_own_sample_space[i][0]] = sorted_own_sample_space[i][1] / sum_
        poisson_probability[sorted_own_sample_space[i][0]] = poisson.pmf(sorted_own_sample_space[i][1], data_mean)

    plt.figure(figsize=(10, 4), dpi=120)
# line chart of poisson distribution
    plt.subplot(1, 2, 1)
    plt.plot(own_probability.keys(), own_probability.values(), 'r-', label='observed probability')
    plt.plot(poisson_probability.keys(), poisson_probability.values(), 'b--', label='Poisson probability')
    plt.title("Number of requests per second (Poisson)")
    plt.xlabel("number of requests")
    plt.ylabel("probability")
    plt.legend(loc='best')
    plt.xlim(-10, 240)
    plt.ylim(-0.001, 0.025)
    plt.grid(linestyle='--', linewidth=1, alpha=0.15)
    #plt.show()
# Scatter chart
    plt.subplot(1, 2, 2)
    plt.plot(xy_value.keys(), xy_value.values(), "go")
    plt.title('Remote request dispersion over the execution time')
    plt.xlabel('time')
    plt.ylabel('number of request')

# show all plots
    plt.show()


if __name__ == "__main__":
    main()
