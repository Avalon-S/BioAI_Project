import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_schedule(solution, processing_times, num_machines, makespan=None, load_balance=None, save_path=None):
    """
     Generate Job Scheduling Visualization plots for Standard NSGA-II and Advanced NSGA-II.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution)))

    machine_available_time = {i: 0 for i in range(num_machines)}
    job_last_end_time = [0] * len(solution)

    # Time period for recording all tasks, categorized by machine
    machine_intervals = {i: [] for i in range(num_machines)}

    for job_idx, job_schedule in enumerate(solution):
        for op_idx, machine_idx in enumerate(job_schedule):
            op_time = processing_times[job_idx, op_idx, machine_idx]
            if np.isfinite(op_time):
                start_time = max(machine_available_time[machine_idx], job_last_end_time[job_idx])
                end_time = start_time + op_time

                # Draw Gantt bars
                ax.barh(
                    y=machine_idx, width=op_time, left=start_time, height=0.8,
                    color=colors[job_idx],
                    label=f'Job {job_idx + 1}' if op_idx == 0 else ""
                )

                # Add task number label
                ax.text(
                    start_time + op_time / 2, machine_idx, f"{job_idx + 1}-{op_idx + 1}",
                    ha='center', va='center', rotation=90, fontsize=10, color='white'
                )

                # Update Status
                machine_available_time[machine_idx] = end_time
                job_last_end_time[job_idx] = end_time

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.legend(loc="upper right", ncol=2, fontsize="small")

    title = "Job Scheduling Visualization"
    if makespan is not None and load_balance is not None:
        title += f" (Makespan: {makespan:.1f}, Load Balance: {load_balance:.1f})"
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def generate_performance_plots(result_folder, dataset_name):
    """
    Generate line plots comparing Standard NSGA-II and Advanced NSGA-II performance metrics for a given dataset.
    """
    dataset_folder = os.path.join(result_folder, dataset_name)
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder {dataset_folder} does not exist.")
        return

    data_names = []
    standard_runtime = []
    advanced_runtime = []
    standard_hv = []
    advanced_hv = []
    standard_div = []
    advanced_div = []

    for data_folder in sorted(os.listdir(dataset_folder)):
        if data_folder == ".ipynb_checkpoints":
            continue

        metrics_file = os.path.join(dataset_folder, data_folder, "metrics.txt")
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            continue

        try:
            with open(metrics_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
                # Extract metrics from the file
                runtime_std = float(lines[3].split(":")[1].strip()) if len(lines) > 3 else 0.0
                hv_std = float(lines[4].split(":")[1].strip()) if len(lines) > 4 else 0.0
                div_std = float(lines[5].split(":")[1].strip()) if len(lines) > 5 else 0.0
                runtime_adv = float(lines[8].split(":")[1].strip()) if len(lines) > 8 else 0.0
                hv_adv = float(lines[9].split(":")[1].strip()) if len(lines) > 9 else 0.0
                div_adv = float(lines[10].split(":")[1].strip()) if len(lines) > 10 else 0.0

                # Append values to respective lists
                data_names.append(data_folder)
                standard_runtime.append(runtime_std)
                standard_hv.append(hv_std)
                standard_div.append(div_std)
                advanced_runtime.append(runtime_adv)
                advanced_hv.append(hv_adv)
                advanced_div.append(div_adv)

        except (IndexError, ValueError) as e:
            print(f"Error parsing file {metrics_file}: {e}")

    if not (len(data_names) == len(standard_runtime) == len(advanced_runtime)):
        print(f"Error: Data mismatch in dataset {dataset_name}. Skipping plot generation.")
        return

    output_folder = os.path.join(result_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # Runtime Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_names, standard_runtime, marker='o', label='Standard NSGA-II', linestyle='--')
    plt.plot(data_names, advanced_runtime, marker='s', label='Advanced NSGA-II', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Data Name")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime Comparison for Dataset: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "runtime_comparison.png"))
    plt.close()

    # Hypervolume Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_names, standard_hv, marker='o', label='Standard NSGA-II', linestyle='--')
    plt.plot(data_names, advanced_hv, marker='s', label='Advanced NSGA-II', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Data Name")
    plt.ylabel("Hypervolume")
    plt.title(f"Hypervolume Comparison for Dataset: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "hypervolume_comparison.png"))
    plt.close()

    # Diversity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_names, standard_div, marker='o', label='Standard NSGA-II', linestyle='--')
    plt.plot(data_names, advanced_div, marker='s', label='Advanced NSGA-II', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Data Name")
    plt.ylabel("Diversity")
    plt.title(f"Diversity Comparison for Dataset: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "diversity_comparison.png"))
    plt.close()

    print(f"Performance plots for dataset {dataset_name} saved in {output_folder}.")
