import json
import sys
import time

from Utils import *
from TimeSeriesLogger import *
from Algorithm import *

time_slice = 1.0
base = ''

class Controller:
    def __init__(self, epoch, cache_size, Algorithm):
        self.epoch = epoch
        self.cache = set()
        self.cache_item_counter = {}
        self.cache_item_filter = set()

        self.last_timestamp = 0
        self.elapsed_time = 0
        self.cache_size = cache_size
        self.Algorithm = Algorithm
        self.epoch_count = 0

    def miss_handler(self, cache_item, timestamp):
        if cache_item in self.cache_item_filter:
            self.cache_item_counter[cache_item] = 1 + self.cache_item_counter.get(cache_item, 0)
        else:
            self.cache_item_filter.add(cache_item)
        self.elapsed_time += timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        if self.elapsed_time > self.epoch:  # trigger algorithm
            print("calling algorithm {0}".format(timestamp))
            # print("len(self.cache_item_counter : {0}".format(len(self.cache_item_counter)))
            self.elapsed_time = 0
            self.cache_item_filter = set()
            print("len(self.cache_item_counter) : {0}".format(len(self.cache_item_counter)))
            self.epoch_count += 1
            if self.epoch_count > 10:
                self.epoch_count = 0
                for rule, count in list(self.cache_item_counter.items()):
                    count = int(count / 4)
                    if count == 0:
                        del self.cache_item_counter[rule]
                        continue
                    self.cache_item_counter[rule] = count
            print("len(self.cache_item_counter) : {0}".format(len(self.cache_item_counter)))
            self.cache = self.Algorithm.get_cache(self.cache_item_counter)
            return True

        return False


class Switch:
    def __init__(self, epoch, cache_size, algorithm):
        self.cache = set()
        self.controller = Controller(epoch, cache_size, algorithm)
        self.logger = TimeSeriesLogger.init_time_series_time_slice(time_slice)

    def send_to_switch(self, cache_item, timestamp):
        if cache_item in self.cache:
            self.logger.log_event(EventType.switch_bw, timestamp, 1)
        else:
            self.logger.log_event(EventType.controller_bw, timestamp, 1)
            if self.controller.miss_handler(cache_item, timestamp):
                self.cache = self.controller.cache

    def __str__(self):
        return Switch.get_switch_str(self.logger)

    @staticmethod
    def get_switch_str(logger):
        total_accesses = logger.sum_all_events(EventType.switch_bw) + logger.sum_all_events(
            EventType.controller_bw)
        total_hits = logger.sum_all_events(EventType.switch_bw)
        total_misses = logger.sum_all_events(EventType.controller_bw)
        ret_str = "Total Accesses: " + str(total_accesses) + ","
        ret_str += "Total Hits Count: " + str(total_hits) + ", Total Hits Percent :" + str(
            round(float(total_hits / total_accesses) * 100,
                  2))
        ret_str += ", Total Misses: " + str(total_misses) + ", Total Misses Percent: " + str(
            round(float(total_misses / total_accesses) * 100,
                  2))
        return ret_str

def online_simulator():
    epoch = float(sys.argv[1])
    cache_size_percentage = float(sys.argv[2])
    dependency_splice = eval(sys.argv[3])
    # with open("../Zipf/prefix_only.txt", 'r') as f:
    #     policy = f.readlines()

    with open("../Zipf/prefix_only.txt", 'r') as f:
        policy = f.readlines()

    cache_size = int(len(policy) * (cache_size_percentage / 100))

    print("=== Epoch {0}, Cache Size % {1}, Cache Size: {2} Dependency Splice {3} ===".format(epoch,
                                                                                              cache_size_percentage,
                                                                                              cache_size,
                                                                                              dependency_splice))
    json_path = "simulator_epoch{0}_cachesizep{1}_cachesize{2}_dependency_splice{3}".format(epoch,
                                                                                            cache_size_percentage,
                                                                                            cache_size,
                                                                                            dependency_splice) + '.json'
    algorithm = OptimalLPMCache(cache_size, policy, dependency_splice=dependency_splice)  # 5%
    # algorithm = PowerOfKChoices(cache_size)
    switch = Switch(epoch, cache_size, algorithm)
    clock = 0
    with open("../Zipf/sorted_zipf_packet_trace.json", 'r') as f:
        packet_array = json.load(f)
    for idx, prefix in enumerate(packet_array):
        switch.send_to_switch(prefix, clock)
        clock += np.random.exponential(1 / 100000)
        if idx % 1000000 == 0:
            print("idx : {0}".format(idx))
            print(switch)

    print(" ")
    print(switch)

    with open('../log/' + json_path, 'w') as f:
        json.dump(switch.logger.event_logger, f)


def offline_simulator():
    global base
    prefix_weight_json_path = sys.argv[1]
    packet_trace_json_path = sys.argv[2]
    cache_size = int(sys.argv[3])

    # prefix_weight_json_path = base + "Zipf/traces/zipf_trace_10_50_prefix2weight.json"
    # packet_trace_json_path = base + "/Zipf/traces/zipf_trace_10_50_packet_array.json"
    with open(prefix_weight_json_path, 'r') as f:
        prefix_weight = json.load(f)
    threshold = 15
    shorter_prefix_weight = {k: np.int64(v) for k, v in prefix_weight.items() if np.int64(v) > threshold}
    print(len(shorter_prefix_weight))
    t0 = time.time()
    opt_cache_algorithm = OptimalLPMCache(cache_size=cache_size,
                                    policy=list(prefix_weight.keys()),
                                    dependency_splice=True)
    print(time.time() - t0)
    t0 = time.time()
    optimal_offline_cache = opt_cache_algorithm.get_cache(shorter_prefix_weight)
    print(time.time() - t0)

    with open(packet_trace_json_path, 'r') as f:
        packet_trace = json.load(f)

    hit = 0
    t0 = time.time()
    for idx, packet in enumerate(packet_trace):
        if packet in optimal_offline_cache:
           hit += 1
        if idx % 100000 == 0:
            print("idx: {0} hit-count :{1}".format(idx, hit))
    print(time.time() - t0)

    print("Hit rate: {0}".format(hit*100/len(packet_trace)))






def main():
    global base
    base = "/home/itamar/PycharmProjects/OptimalLPMCaching/"
    # online_simulator()
    offline_simulator()



if __name__ == "__main__":
    main()