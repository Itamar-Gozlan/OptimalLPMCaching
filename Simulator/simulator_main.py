from Utils import *
from TimeSeriesLogger import *
from Algorithm import *

time_slice = 1.0


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


def main():
    epoch = 1.0
    with open("../Zipf/prefix_only.txt", 'r') as f:
        policy = f.readlines()
    cache_size = int(len(policy) * (0.5 / 100))
    cache_size = 10
    print(cache_size)
    algorithm = OptimalLPMCache(cache_size, policy, dependency_splice=False)  # 5%
    # algorithm = PowerOfKChoices(cache_size)
    switch = Switch(epoch, cache_size, algorithm)
    clock = 0
    with open("../Zipf/random_zipf_trace.json", 'r') as f:
        packet_array = json.load(f)
    for idx, prefix in enumerate(packet_array):
        switch.send_to_switch(prefix, clock)
        clock += np.random.exponential(1 / 100000)
        if idx % 1000000 == 0:
            print("idx : {0}".format(idx))
            print(switch)

    print(" ")
    print(switch)


if __name__ == "__main__":
    main()
    """
    TODO
    1. Leaf push with power of K choices
    3. Heuristic algorithm
    
    """
