import os
import glob
import re
import sys
import datetime


def main():
    dir_path = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-chicago"
    # dir_path = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2019/equinix-chicago"

    rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-chicago/"
    rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-sanjose/"

    quarter_dir = ["20140320-130000.UTC",
                   "20140619-130000.UTC",
                   "20140918-130000.UTC",
                   "20141218-130000.UTC"]
    # rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2019/equinix-nyc/"
    # quarter_dir = ["20190117-130000.UTC"]


    first_ts = 0
    last_ts = 0
    for qdir in quarter_dir:
        print(qdir)
        count_ipv4 = 0
        count_ipv6 = 0
        for filename in glob.iglob(rootdir + qdir + '/*.stats', recursive=True):
            """
            First timestamp:                     1418911260.000002000
            Last timestamp:                      1418911271.640382000
            """
            with open(filename, 'r') as f:
                for line in f:
                    if re.search("First timestamp", line):
                        first_ts = float((line.strip().replace(" ", '').split(":")[-1]))
                    if re.search("Last timestamp", line):
                        last_ts += float((line.strip().replace(" ", '').split(":")[-1]))

                    temp = datetime.datetime.fromtimestamp((last_ts - first_ts) / 1000).strftime('%H:%M:%S')
                    # print(temp)

                    if re.search("IPv4 pkt", line):
                        curr_count_ipv4 = int(line.strip().replace(" ", '').split(":")[-1])
                        count_ipv4 += curr_count_ipv4
                        print("count_ipv4: {0}".format("{:,}".format(curr_count_ipv4)))
                    if re.search("IPv6 pkts", line):
                        curr_count_ipv6 = int(line.strip().replace(" ", '').split(":")[-1])
                        count_ipv6 += curr_count_ipv6

                    # if "00:3" == temp[:4]:
                print(temp)
                    # print("count_ipv4: {0}".format("{:,}".format(curr_count_ipv4)))
                    # print("count_ipv6: {0}".format("{:,}".format(curr_count_ipv6)))


        print("Q {1}: count_ipv4: {0}".format("{:,}".format(count_ipv4), qdir))
        print("Q {1} :count_ipv6: {0}".format("{:,}".format(count_ipv6), qdir))

    # for file in os.listdir(dir_path):
    #     print("s")
    # 14,719,954,953


if __name__ == "__main__":
    main()
