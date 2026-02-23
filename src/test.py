import os
import yaml
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.recovery.benchmark import Benchmark
import datetime
import time
import math
import matplotlib.pyplot as plt
bench = Benchmark()
dir = "../data/04_working_files/"
with open(dir + "file_list.yaml", 'r') as file:
    datafiles = yaml.safe_load(file)
datafiles = ["../" + d for d in datafiles]
with open(dir + "file_list_missing.yaml", 'r') as file:
    datafiles_missing = yaml.safe_load(file)
datafiles_missing = ["../" + d for d in datafiles_missing]
ts = TimeSeries()

bench = Benchmark()
algorithms=ts.algorithms
datasets=datafiles
ts_missing=datafiles_missing
patterns=["mcar"]
x_axis=[0.1]
optimizers=["ray_tune"]
metrics=["*"]
normalizer=""
nbr_series=11
nbr_vals=100000
dl_ratio=0.9
verbose=True
save_dir="./imputegap_assets/benchmark"
runs=1
x=0.2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

run_storage = []
not_optimized = ["none"]
mean_group = ["mean", "MeanImpute", "min", "MinImpute", "zero", "ZeroImpute", "MeanImputeBySeries", "meanimpute", "minimpute", "zeroimpute", "meanimputebyseries"]

if not isinstance(algorithms, list):
    raise TypeError(f"'algorithms' must be a list, but got {type(algorithms).__name__}")
if not isinstance(datasets, list):
    raise TypeError(f"'datasets' must be a list, but got {type(datasets).__name__}")
if not isinstance(patterns, list):
    raise TypeError(f"'patterns' must be a list, but got {type(patterns).__name__}")
if not isinstance(x_axis, list):
    raise TypeError(f"'x_axis' must be a list, but got {type(x_axis).__name__}")

if "*" in metrics or "all" in metrics:
    metrics = utils.list_of_metrics()
if "*" in metrics or "all" in algorithms:
    all_algs = utils.list_of_algorithms()
    algorithms = [item for item in all_algs if item.upper() != "MPIN"]

directory_now = datetime.datetime.now()
directory_time = directory_now.strftime("%y_%m_%d_%H_%M_%S")
save_dir = save_dir + "/" + "bench_" + directory_time

benchmark_time = time.time()
for i_run in range(0, abs(runs)):
    for i, dataset in enumerate(datasets):
        runs_plots_scores = {}
        block_size_mcar = 10
        y_p_size = max(4, len(algorithms)*0.275)

        if verbose:
            print("\n1. evaluation launch for", dataset, "\n")
        ts_test = TimeSeries()
        default_data = TimeSeries()

        header = False
        if dataset == "eeg-reading" or dataset == "eegreading":
            header = True

        reshp = False
        default_data.load_series(data=utils.search_path(dataset), header=header, verbose=False)
        Mdef, Ndef = default_data.data.shape

        if Ndef > nbr_vals or Mdef > nbr_series:
            reshp = True
            print(f"\nThe dataset contains a large number of values {default_data.data.shape}, which may be too much for some algorithms to handle efficiently. Consider reducing the number of series or the volume of data.")
        default_data = None

        ts_test.load_series(data=utils.search_path(dataset), nbr_series=nbr_series, nbr_val=nbr_vals, header=header)
        _, N = ts_test.data.shape

        if reshp:
            print(f"Benchmarking module has reduced the shape to {ts_test.data.shape}.\n")

        if N < 250:
            print(f"The block size is too high for the number of values per series, reduce to 2\n")
            block_size_mcar = 2

        if normalizer in utils.list_of_normalizers():
            ts_test.normalize(verbose=verbose)

        for pattern in patterns:
            if verbose:
                print("\n2. contamination of", dataset, "with pattern", pattern, "\n")

            for algorithm in algorithms:
                has_been_optimized = False

                if verbose:
                    print("\n3. algorithm evaluated", algorithm, "with", pattern, "\n")
                else:
                    print(f"{algorithm} is tested with {pattern}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                incomp_data = TimeSeries()
                incomp_data = incomp_data.load_series(data = ts_missing[i]).data

                for optimizer in optimizers:
                    algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm, verbose=verbose)

                    if isinstance(optimizer, dict):
                        optimizer_gt = {"input_data": ts_test.data, **optimizer}
                        optimizer_value = optimizer.get('optimizer')  # or optimizer['optimizer']

                        if not has_been_optimized and algorithm not in mean_group and algorithm not in not_optimized:
                            if verbose:
                                print("\n5. AutoML to set the parameters", optimizer, "\n")
                            i_opti = self._config_optimization(0.20, ts_test, pattern, algorithm, block_size_mcar)

                            if utils.check_family("DeepLearning", algorithm):
                                i_opti.impute(user_def=False, params=optimizer_gt, tr_ratio=0.80)
                            else:
                                i_opti.impute(user_def=False, params=optimizer_gt)

                            utils.save_optimization(optimal_params=i_opti.parameters, algorithm=algorithm, dataset=dataset, optimizer="e")

                            has_been_optimized = True
                        else:
                            if verbose:
                                print("\n5. AutoML already optimized...\n")

                        if algorithm not in mean_group and algorithm not in not_optimized:
                            if i_opti.parameters is None:
                                opti_params = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="e")
                                if verbose:
                                    print("\n6. imputation", algorithm, "with optimal parameters from files", *opti_params)
                            else:
                                opti_params = i_opti.parameters
                                if verbose:
                                    print("\n6. imputation", algorithm, "with optimal parameters from object", *opti_params)
                        else:
                            if verbose:
                                print("\n5. No AutoML launches without optimal params for", algorithm, "\n")
                            opti_params = None
                    else:
                        if verbose:
                            print("\n5. Default parameters have been set the parameters", optimizer, "for", algorithm, "\n")
                        optimizer_value = optimizer
                        opti_params = None

                    start_time_imputation = time.time()

                    if not bench._benchmark_exception(dataset, algorithm, pattern, 0.2):
                        if utils.check_family("DeepLearning", algorithm) or utils.check_family("LLMs", algorithm):
                            if 0.2 > round(1-dl_ratio, 2):
                                algo.recov_data = incomp_data
                            else:
                                algo.impute(params=opti_params, tr_ratio=dl_ratio)
                        else:
                            algo.impute(params=opti_params)
                    else:
                        algo.recov_data = incomp_data

                    end_time_imputation = time.time()

                    algo.score(input_data=ts_test.data, recov_data=algo.recov_data, verbose=False)

                    if "*" not in metrics and "all" not in metrics:
                        algo.metrics = {k: algo.metrics[k] for k in metrics if k in algo.metrics}

                    time_imputation = (end_time_imputation - start_time_imputation) * 1000
                    if time_imputation < 1:
                        time_imputation = 1
                    log_time_imputation = math.log10(time_imputation) if time_imputation > 0 else None

                    algo.metrics["RUNTIME"] = time_imputation
                    algo.metrics["RUNTIME_LOG"] = log_time_imputation

                    dataset_s = dataset
                    if "-" in dataset:
                        dataset_s = dataset.replace("-", "")

                    save_dir_plot = save_dir + "/" + dataset_s + "/" + pattern + "/recovery/"
                    cont_rate = int(0.2*100)
                    ts_test.plot(input_data=ts_test.data, incomp_data=incomp_data, recov_data=algo.recov_data, nbr_series=3, subplot=True, algorithm=algo.algorithm, cont_rate=str(cont_rate), display=False, save_path=save_dir_plot, verbose=False)

                    runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern), {}).setdefault(str(algorithm), {}).setdefault(str(optimizer_value), {})[str(x)] = {"scores": algo.metrics}

                print(f"done!\n\n")
        #save_dir_runs = save_dir + "/_details/run_" + str(i_run) + "/" + dataset
        #if verbose:
        #    print("\nruns saved in : ", save_dir_runs)
        #self.generate_plots(runs_plots_scores=runs_plots_scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_runs, display=False, verbose=verbose)
        #self.generate_plots(runs_plots_scores=runs_plots_scores, ticks=x_axis, metrics=metrics, subplot=False, y_size=y_p_size, save_dir=save_dir_runs, display=False, verbose=verbose)
        #self.generate_reports_txt(runs_plots_scores=runs_plots_scores, save_dir=save_dir_runs, dataset=dataset, metrics=metrics, run=i_run, verbose=verbose)
        #self.generate_reports_excel(runs_plots_scores, save_dir_runs, dataset, i_run, verbose=verbose)
        run_storage.append(runs_plots_scores)

plt.close('all')  # Close all open figures

for x, m in enumerate(reversed(metrics)):
    #tag = True if x == (len(metrics)-1) else False
    scores_list, algos, sets = bench.avg_results(*run_storage, metric=m)
    _ = bench.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m, save_dir=save_dir, display=False)

run_averaged = bench.average_runs_by_names(run_storage)

benchmark_end = time.time()
total_time_benchmark = round(benchmark_end - benchmark_time, 4)
print(f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds\n")

verb = True

for scores in run_averaged:
    all_keys = list(scores.keys())
    dataset_name = str(all_keys[0])
    save_dir_agg_set = save_dir + "/" + dataset_name

    bench.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, dataset=dataset_name, metrics=metrics, rt=total_time_benchmark, run=-1)
    bench.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_agg_set, display=verb)

print("\nThe results are saved in : ", save_dir, "\n")
bench.list_results = run_averaged
bench.aggregate_results = scores_list