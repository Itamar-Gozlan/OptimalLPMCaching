import json
import sys
import time

import Utils
from Utils import *
from TimeSeriesLogger import *
from Algorithm import *
from process_results import RunCheck

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
    policy = sys.argv[1]
    packet_trace_json_path = sys.argv[2]
    cache_size = int(sys.argv[3])
    dependency_splice = eval(sys.argv[4])
    epoch = float(sys.argv[5])

    with open(policy, 'r') as f:
        policy = json.load(f)

    print("=== Epoch {0}, Cache Size % {1}, Cache Size: {2} Dependency Splice {3} ===".format(epoch,
                                                                                              cache_size / len(policy),
                                                                                              cache_size,
                                                                                              dependency_splice))
    # json_path = "simulator_epoch{0}_cachesizep{1}_cachesize{2}_dependency_splice{3}".format(epoch, cache_size /
    # len(policy), cache_size, dependency_splice) + '.json'
    algorithm = HeuristicLPMCache(cache_size, policy, dependency_splice=dependency_splice)  # 5%
    switch = Switch(epoch, cache_size, algorithm)
    clock = 0
    with open(packet_trace_json_path, 'r') as f:
        packet_array = json.load(f)
    for idx, prefix in enumerate(packet_array):
        switch.send_to_switch(prefix, clock)
        clock += np.random.exponential(1 / 100000)
        if idx % 1000000 == 0:
            print("idx : {0}".format(idx))
            print(switch)

    print(" ")
    print(switch)

    # with open('../log/' + json_path, 'w') as f:
    #     json.dump(switch.logger.event_logger, f)


def offline_simulator():
    global base
    prefix_weight_json_path = sys.argv[1]
    dir_path = sys.argv[2]

    print("prefix_weight_json_path : {0}".format(prefix_weight_json_path))

    with open(prefix_weight_json_path, 'r') as f:
        prefix_weight = json.load(f)

    threshold = 0 # using filtered prefix2weight
    shorter_prefix_weight = {k: int(v) for k, v in prefix_weight.items() if np.int64(v) > threshold}
    shorter_prefix_weight['0.0.0.0/0'] = 0
    cache_size = 1024
    # cache_size = 64

    print(len(shorter_prefix_weight))
    # optimal_lpm_cache = OptimalLPMCache(policy=list(shorter_prefix_weight.keys()),
    #                                       prefix_weight=shorter_prefix_weight,
    #                                       cache_size=cache_size)

    optimized_lpm_cache = OptimizedOptimalLPMCache(policy=list(shorter_prefix_weight.keys()),
                                          prefix_weight=shorter_prefix_weight,
                                          cache_size=cache_size)

    # RunCheck.draw_policy_tree_from_algorithm(optimal_lpm_cache, cache_size, prefix_weight)

    t0 = time.time()
    optimized_lpm_cache.get_optimal_cache()
    print("optimized_lpm_cache: {0}".format(time.time() - t0))
    # t0 = time.time()
    # optimal_lpm_cache.get_optimal_cache()
    # print("optimal_lpm_cache: {0}".format(time.time() - t0))


    optimized_lpm_cache.to_json(dir_path)


def run_OTC():
    policy_json_path = sys.argv[1]
    packet_trace_json_path = sys.argv[2]
    cache_size = int(sys.argv[3])

    with open(policy_json_path, 'r') as f:
        policy = json.load(f)

    with open(packet_trace_json_path, 'r') as f:
        packet_trace = json.load(f)

    OTC = OnlineTreeCache(policy, cache_size)

    hit = 0
    t0 = time.time()
    for idx, packet in enumerate(packet_trace):
        if packet in OTC.cache:
            OTC.cache_hit(packet)
            hit += 1
        else:
            OTC.cache_miss(packet)

        if idx % 100000 == 0:
            print("idx: {0} hit-count :{1}".format(idx, hit))
    print(time.time() - t0)

    print("Hit rate: {0}".format(hit * 100 / len(packet_trace)))


def main():
    # online_simulator()
    offline_simulator()
    # run_OTC()


if __name__ == "__main__":
    main()
