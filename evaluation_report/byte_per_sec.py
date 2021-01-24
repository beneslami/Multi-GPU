"""
Statistical Evaluation piece of code

This piece of code provides plots bytes_per_sec1,2 and 3. As a user, you just
need to change the input file (remote#1,2 or 3)

python byte_per_sec.py <INPUT_NAME>

Author: Benyamin Eslami ( www.beneslami.com)
"""
import sys
import matplotlib.pyplot as plt
from numpy import mean
import decimal


def per_micro_second(input_str):
    contents = " "
    f = open(input_str, "r")
    if f.mode == "r":
        contents = f.read()
    data_list = contents.splitlines()
    integer_map = []
    for i in range(len(data_list)):
        integer_map.append(int(data_list[i]))
    bytes_per_sec = []
    for i in range(len(integer_map)):
        b = decimal.Decimal(128) / decimal.Decimal(integer_map[i])
        bytes_per_sec.append(float(b))
    print(bytes_per_sec)


def per_second(input_str):
    contents = " "
    xy_value = {}
    key = 0
    f = open(input_str, "r")
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

    byte_per_sec = {}
    total_byte = 0
    for i in range(len(xy_value)):
        byte_per_sec[i] = (xy_value[i] * 128) / 1000
        total_byte += byte_per_sec[i]
    total_byte_in_KB = total_byte
    print(mean(list(byte_per_sec.values())))
    plt.bar(byte_per_sec.keys(), byte_per_sec.values(), color='0c', label='bytes')
    x = [0, 750]
    y = [mean(list(byte_per_sec.values())), mean(list(byte_per_sec.values()))]
    plt.plot(x, y, 'r--', linewidth=6, label='mean')
    plt.title("Amount of bytes per second")
    plt.xlabel("Second (s)")
    plt.ylabel("Bytes (KB)")
    plt.legend(loc='best')
    plt.show()


def main():
    if "micro_remote#" in str(sys.argv[1]):
        per_micro_second(str(sys.argv[1]))
    else:
        per_second(str(sys.argv[1]))


if __name__ == "__main__":
    main()
