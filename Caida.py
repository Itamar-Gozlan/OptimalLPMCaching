import json
import os
import glob
import re
import sys
import datetime
from csv import reader, DictReader
import ipaddress
import Utils
import numpy as np


def create_LPM_trace():
    csv_path = sys.argv[1]
    row_size = 8
    ip_dst_idx = 4
    ip_src_idx = 3

    tcp_dstport_idx = 6
    udp_dstport_idx = 8
    type_idx = 11
    idx_to_name = {0: "time_idx",
                   3: "src_ip",
                   4: "dst_ip",
                   5: "tcp_srcport",
                   6: "tcp_dstport",
                   7: "udp_srcport",
                   8: "udp_dstport"}
    with open(csv_path, 'r') as read_obj:
        """
        Transform to /20
        for every rule in policy:
            if UDP -> replace with rule
                rule = rule + /22
                if TCP -> replace with /22 if src ip bit lsb is 0
                I
        """

        mask = 20
        tpt_type = "UDP"
        policy_20 = set()
        dst_port_idx = udp_dstport_idx
        for tpt_type, dst_port_idx in zip(["UDP", "TCP"], [udp_dstport_idx, tcp_dstport_idx]):
            csv_reader = reader(read_obj)
            binary_policy = []
            prefix_weight = {}
            for row in csv_reader:
                if len(row) < type_idx:
                    continue
                dst_ip = row[ip_dst_idx]
                try:
                    masked_binary = "{:032b}".format(int(ipaddress.IPv4Address(dst_ip)))[:int(mask)]  # /20
                except ipaddress.AddressValueError:
                    continue
                policy_20.add(masked_binary)
                try:
                    if row[type_idx] == tpt_type:
                        masked_binary = masked_binary + "00"  # /22
                        if bin(int(row[dst_port_idx]))[-1] == '0':
                            masked_binary = masked_binary + "11"  # /24
                            if bin(int(row[dst_port_idx]))[-2] == '0':
                                masked_binary = masked_binary + "00"  # /26
                                if "{:032b}".format(int(ipaddress.IPv4Address(row[ip_src_idx])))[-1] == '0':
                                    masked_binary = masked_binary + "11"  # /28
                                    if "{:032b}".format(int(ipaddress.IPv4Address(row[ip_src_idx])))[-1] == '0':
                                        masked_binary = masked_binary + "00"  # /30
                except IndexError:
                    continue

                prefix_weight[Utils.binary_lpm_to_str(masked_binary)] = 1 + prefix_weight.get(
                    Utils.binary_lpm_to_str(masked_binary), 0)
                binary_policy.append(Utils.binary_lpm_to_str(masked_binary))
            policy = list(set(binary_policy))
            print("len(policy_20) = {0}".format(len(policy_20)))
            print("len(policy) : {0}".format(len(policy)))
            base_path = '/'.join(csv_path.split('/')[:-1])
            # with open(base_path + '/caida_trace{0}_packet_array.json'.format(tpt_type), 'w') as f:
            #     json.dump(binary_policy, f)

            print(base_path + 'caida_trace{0}_prefix_weight.json'.format(tpt_type))
            with open(base_path + '/caida_trace{0}_prefix_weight.json'.format(tpt_type), 'w') as f2:
                json.dump(prefix_weight, f2)


def analyze_data():
    # dir_path = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-chicago"
    dir_path = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2019/equinix-chicago"

    # rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-chicago/"
    # rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2014/equinix-sanjose/"

    # quarter_dir = ["20140320-130000.UTC",
    #                "20140619-130000.UTC",
    #                "20140918-130000.UTC",
    #                "20141218-130000.UTC"]
    rootdir = "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/data.caida.org/datasets/passive-2019/equinix-nyc/"
    quarter_dir = ["20190117-130000.UTC"]
    data = []
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
                        # print("count_ipv4: {0}".format("{:,}".format(curr_count_ipv4)))
                    if re.search("IPv6 pkts", line):
                        curr_count_ipv6 = int(line.strip().replace(" ", '').split(":")[-1])
                        count_ipv6 += curr_count_ipv6

                    if re.search("Unique IPv4 destination addresses", line):
                        # print(line)
                        data.append((line.strip().split(' ')[-1], filename))

                    # if "00:3" == temp[:4]:
                # print("count_ipv4: {0}".format("{:,}".format(curr_count_ipv4)))
                # print("count_ipv6: {0}".format("{:,}".format(curr_count_ipv6)))

        # print("Q {1}: count_ipv4: {0}".format("{:,}".format(count_ipv4), qdir))
        # print("Q {1} :count_ipv6: {0}".format("{:,}".format(count_ipv6), qdir))

    # for file in os.listdir(dir_path):
    #     print("s")
    # 14,719,954,953
    for v, p in sorted(data, key=lambda x: np.int64(x[0])):
        print(v)
        print(p)
        print(" ")


def main():
    create_LPM_trace()
    # analyze_data()


if __name__ == "__main__":
    main()
