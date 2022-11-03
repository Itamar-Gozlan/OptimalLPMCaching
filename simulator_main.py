import sys
from Utils import *
from Algorithm import *


class RunAlgorithm:
    @staticmethod
    def measure_algorithm_time(algorithm, name, prefix_weight, dir_path):
        result_df = pd.DataFrame(columns=['Cache Size', 'Hit Rate', 'Elapsed Time'])
        for cache_size in [64, 128, 256, 512, 1024]:
            t0 = time.time()
            algorithm.compute_cache(cache_size, prefix_weight)
            elapsed_time = time.time() - t0
            hit_rate = algorithm.get_hit_rate()
            row = {'Cache Size': cache_size,
                   'Hit Rate': hit_rate,
                   'Elapsed Time': elapsed_time}
            print("cache_size: {0}, elapsed_time: {1}".format(cache_size, elapsed_time))
            result_df = result_df.append(row, ignore_index=True)

        result_df.to_csv('{0}/{1}.csv'.format(dir_path, name))

    @staticmethod
    def run():
        flag = int(sys.argv[1])
        prefix_weight_json_path = sys.argv[2]
        dir_path = sys.argv[3]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(prefix_weight_json_path, 'r') as f:
            prefix_weight = json.load(f)

        prefix_weight = {k: int(v) for k, v in prefix_weight.items()}

        trace_name = prefix_weight_json_path.split('/')[-1].replace('.json', '') + '_'
        if flag == 0:
            OptSplice = OptimizedOptimalLPMCache(policy=list(prefix_weight.keys()),
                                                 prefix_weight=prefix_weight,
                                                 cache_size=0)
            return RunAlgorithm.measure_algorithm_time(OptSplice, trace_name + "OptSplice", prefix_weight, dir_path)
        if flag == 1:
            GreedySplice = Algorithm.HeuristicLPMCache(0, prefix_weight.keys(), True)
            return RunAlgorithm.measure_algorithm_time(GreedySplice, trace_name + "GreedySplice", prefix_weight,
                                                       dir_path)

        if flag == 2:
            MixedSet = CacheFlow(prefix_weight.keys(), prefix_weight)
            return RunAlgorithm.measure_algorithm_time(MixedSet, trace_name + "MixedSet", prefix_weight, dir_path)

        if flag == 3:
            OptLocal = Algorithm.HeuristicLPMCache(0, prefix_weight.keys(), False)
            return RunAlgorithm.measure_algorithm_time(OptLocal, trace_name + "OptLocal", prefix_weight, dir_path)


def main():
    RunAlgorithm.run()


if __name__ == "__main__":
    main()
