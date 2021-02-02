from statistics import mean, variance, stdev
import matplotlib.pyplot as plt


def size_count(_list):
    sec_t = int(_list[1][len(_list[1])-1].split(":")[1])
    byte_per_sec_overall = {}
    byte_per_sec_request = {}
    byte_per_sec_reply = {}
    j = j1 = j2 = temp = temp1 = temp2 = 0
    byte_per_sec_overall[j] = int(_list[1][3])
    for i in range(2, len(_list)):
        if int(_list[i][len(_list[i])-1].split(":")[1]) == sec_t:
            temp += int(_list[i][3])
            if _list[i][4] == "READ_REQUEST":
                temp1 += int(_list[i][3])
            elif _list[i][4] == "READ_REPLY":
                temp2 += int(_list[i][3])
        else:
            byte_per_sec_overall[j] = temp
            byte_per_sec_request[j] = temp1
            byte_per_sec_reply[j] = temp2
            j += 1
            sec_t = int(_list[i][len(_list[i])-1].split(":")[1])
            temp = temp1 = temp2 = 0

    plt.plot(byte_per_sec_overall.keys(), byte_per_sec_overall.values(), label="overall")
    plt.plot(byte_per_sec_request.keys(), byte_per_sec_request.values(), label="requests")
    plt.plot(byte_per_sec_reply.keys(), byte_per_sec_reply.values(), label="replies")
    plt.legend(loc="best")
    plt.show()


def cycle_count(_list):
    read_req = {}
    read_reply = {}
    write_req = {}
    write_reply = {}
    packet = {}
    j1 = j2 = j3 = j4 = 0
    for i in range(len(_list)):
        if _list[i][4] == "READ_REQUEST":
            read_req[j1] = int(_list[i][8])
            packet[i] = int(_list[i][8])
            j1 +=1
        elif _list[i][4] == "READ_REPLY":
            read_reply[j2] = int(_list[i][8])
            packet[i] = int(_list[i][8])
            j2 += 1
        elif _list[i][4] == "WRITE_REQUESTّّ":
            write_req[j3] = int(_list[i][8])
            packet[i] = int(_list[i][8])
            j3 += 1
            print(write_req[j3])
        elif _list[i][4] == "WRITE_REPLY":
            write_reply[j4] = int(_list[i][8])
            packet[i] = int(_list[i][8])
            j4 += 1
            print(write_reply[j4])

    read_req_mean = mean(read_req.values())
    read_req_var = variance(read_req.values())
    read_req_stdev = stdev(read_req.values())
    read_reply_mean = mean(read_reply.values())
    read_reply_var = variance(read_reply.values())
    read_reply_stdev = stdev(read_reply.values())

    plt.figure(figsize=(10, 4), dpi=120)
    plt.subplots_adjust(left=0.09,
                        bottom=0.147,
                        right=0.99,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.3)
    plt.subplot(1, 2, 1)
    plt.title("Read Requests")
    plt.xlabel("i-th packet")
    plt.ylabel("Number of CPU cycles")
    mean_x = [0, len(read_req.keys())]
    mean_y = [read_req_mean, read_req_mean]
    stdev_x = [0, len(read_req.keys())]
    stdev__y = [read_req_mean - read_req_stdev, read_req_mean - read_req_stdev]
    stdev_y =[read_req_mean + read_req_stdev, read_req_mean + read_req_stdev]
    plt.plot(read_req.keys(), read_req.values(), label="requests")
    plt.plot(mean_x, mean_y, "r", label= "mean")
    plt.plot(stdev_x, stdev__y, "r--", label="stdev")
    plt.plot(stdev_x, stdev_y, "r--")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.title("Read Replies")
    plt.xlabel("i-th packet")
    plt.ylabel("Number of CPU cycles")
    mean_xx = [0, len(read_reply.keys())]
    mean_yy = [read_reply_mean, read_reply_mean]
    stdev_xx = [0, len(read_reply.keys())]
    stdev__yy = [read_reply_mean - read_reply_stdev, read_reply_mean - read_reply_stdev]
    stdev_yy =[read_reply_mean + read_reply_stdev, read_reply_mean + read_reply_stdev]
    plt.plot(read_reply.keys(), read_reply.values(), label="requests")
    plt.plot(mean_xx, mean_yy, "r", label= "mean")
    plt.plot(stdev_xx, stdev__yy, "r--", label="stdev")
    plt.plot(stdev_xx, stdev_yy, "r--")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    file = open("out1", "r")
    raw_content = ""

    if file.mode == "r":
        raw_content = file.readlines()

    lined_content = []
    lined_list = []
    for line in raw_content:
        lined_content.append(line)

    for i in lined_content:
        item = [x for x in i.split("\t") if x not in ['', '\t']]
        lined_list.append(item)

    size_count(lined_list)
    cycle_count(lined_list)
