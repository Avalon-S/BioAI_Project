import os
import json
import matplotlib.pyplot as plt

def generate_performance_plots(result_folder, dataset_name):
    dataset_folder = os.path.join(result_folder, dataset_name)
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder {dataset_folder} does not exist.")
        return

    data_names = []
    metrics_dict = {}  # 动态记录每个算法的所有指标

    for data_folder in sorted(os.listdir(dataset_folder)):
        folder_path = os.path.join(dataset_folder, data_folder)
        if not os.path.isdir(folder_path) or data_folder == ".ipynb_checkpoints":
            continue

        metrics_file = os.path.join(folder_path, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            continue

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            if not metrics:
                print(f" Empty metrics in: {metrics_file}")
                continue

            data_names.append(data_folder)

            for algo_name, result in metrics.items():
                if algo_name not in metrics_dict:
                    metrics_dict[algo_name] = {
                        "runtime": [],
                        "hypervolume": [],
                        "diversity": []
                    }

                metrics_dict[algo_name]["runtime"].append(result["runtime"])
                metrics_dict[algo_name]["hypervolume"].append(result["hypervolume"])
                metrics_dict[algo_name]["diversity"].append(result["diversity"])

        except Exception as e:
            print(f" Error parsing {metrics_file}: {e}")

    if not data_names:
        print(f" No data parsed for dataset {dataset_name}. Skipping plot generation.")
        return

    output_folder = os.path.join(result_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # === 绘图函数（可复用）
    def plot_and_save(metric_key, ylabel, filename, title):
        plt.figure(figsize=(10, 6))
        for algo_name, algo_data in metrics_dict.items():
            if len(algo_data[metric_key]) == len(data_names):
                plt.plot(data_names, algo_data[metric_key], marker='o', label=algo_name)
        plt.xticks(rotation=45)
        plt.xlabel("Data Name")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    # === 画三个指标图（已去掉 idle_time）
    plot_and_save("runtime", "Runtime (seconds)", "runtime_comparison.png", f"Runtime Comparison for Dataset: {dataset_name}")
    plot_and_save("hypervolume", "Hypervolume", "hypervolume_comparison.png", f"Hypervolume Comparison for Dataset: {dataset_name}")
    plot_and_save("diversity", "Diversity", "diversity_comparison.png", f"Diversity Comparison for Dataset: {dataset_name}")

    print(f" Performance plots for dataset {dataset_name} saved in {output_folder}.")
