import os
import random
import numpy as np
import matplotlib.pyplot as plt
from data_processing import read_and_parse_fjsp_file
from nsga2_algorithms import run_standard_nsga2, advanced_nsga2
from metrics import hypervolume, diversity
from visualization import visualize_schedule

# Global random seed and parameters
SEED = 43
rnd = random.Random(SEED)
POP_SIZE = 50  # Population size for NSGA-II
N_GEN = 100    # Number of generations

def process_all_files(input_folder, output_folder, dataset_name):
    """
    Process all .fjs files in the specified folder, run standard and advanced NSGA-II algorithms, and save results.

    Args:
        input_folder (str): Path to the folder containing .fjs files.
        output_folder (str): Path to save the results.
        dataset_name (str): Name of the dataset being processed.
    """
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".fjs")]
    total_files = len(all_files)
    
    print(f"Starting processing for dataset: {dataset_name}")
    print(f"Dataset contains {total_files} files.\n")
    
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(input_folder, filename)
        output_dir = os.path.join(output_folder, filename[:-4])  # Mk01, Mk02, ...
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{index}/{total_files}] Processing file: {filename}...")
        
        try:
            # Read and parse data
            operation_data, processing_times, num_jobs, num_machines = read_and_parse_fjsp_file(file_path)
            avg_machines_per_operation = processing_times[np.isfinite(processing_times)].size / (num_jobs * processing_times.shape[1])

            # Save log file
            log_file_path = os.path.join(output_dir, "log.txt")
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Dataset: {dataset_name}\n")
                log_file.write(f"Jobs: {num_jobs}\n")
                log_file.write(f"Machines: {num_machines}\n")
                log_file.write(f"Avg Machines Per Operation: {avg_machines_per_operation:.2f}\n")

            # Run standard NSGA-II
            print(f"  Running standard NSGA-II...")
            std_pop, std_archive, std_runtime = run_standard_nsga2(processing_times, POP_SIZE, N_GEN, SEED)
            
            # Run advanced NSGA-II
            print(f"  Running advanced NSGA-II...")
            adv_pop, adv_archive, adv_runtime = advanced_nsga2(processing_times, POP_SIZE, N_GEN, SEED)

            # Calculate metrics
            reference_point = [
                max(max(std_pop[:, 0]), max(adv_pop[:, 0])),
                max(max(std_pop[:, 1]), max(adv_pop[:, 1])),
            ]
            std_hv = hypervolume(std_pop, reference_point)
            adv_hv = hypervolume(adv_pop, reference_point)
            std_div = diversity(std_pop)
            adv_div = diversity(adv_pop)

            # Save performance metrics
            metrics_file_path = os.path.join(output_dir, "metrics.txt")
            with open(metrics_file_path, "w") as metrics_file:
                metrics_file.write(f"Dataset: {dataset_name}\n\n")
                metrics_file.write("Standard NSGA-II:\n")
                metrics_file.write(f"Runtime: {std_runtime}\n")
                metrics_file.write(f"Hypervolume: {std_hv}\n")
                metrics_file.write(f"Diversity: {std_div}\n\n")
                metrics_file.write("Advanced NSGA-II:\n")
                metrics_file.write(f"Runtime: {adv_runtime}\n")
                metrics_file.write(f"Hypervolume: {adv_hv}\n")
                metrics_file.write(f"Diversity: {adv_div}\n")

            # Save comparison plot
            plt.figure(figsize=(10, 6))
            plt.scatter(std_pop[:, 0], std_pop[:, 1], c="red", label="Standard NSGA-II")
            plt.scatter(adv_pop[:, 0], adv_pop[:, 1], c="blue", label="Advanced NSGA-II")
            plt.xlabel("Makespan")
            plt.ylabel("Load Balance")
            plt.legend()
            plt.title(f"Comparison of NSGA-II Variants ({dataset_name})")
            plt_path = os.path.join(output_dir, "comparison_nsga2_variants.png")
            plt.savefig(plt_path)
            plt.close()

            # Save scheduling visualizations
            print(f"  Saving standard NSGA-II scheduling visualizations...")
            if std_archive:
                for i, solution in enumerate(std_archive):
                    visualize_schedule(solution.candidate, processing_times, num_machines)
                    schedule_path = os.path.join(output_dir, f"std_schedule_{i + 1}.png")
                    plt.savefig(schedule_path)
                    plt.close()

            print(f"  Saving advanced NSGA-II scheduling visualizations...")
            if adv_archive:
                for i, solution in enumerate(adv_archive):
                    visualize_schedule(solution.candidate, processing_times, num_machines)
                    schedule_path = os.path.join(output_dir, f"adv_schedule_{i + 1}.png")
                    plt.savefig(schedule_path)
                    plt.close()

            print(f"[{index}/{total_files}] File {filename} processed successfully.\n")

        except Exception as e:
            print(f"[{index}/{total_files}] Failed to process file {filename}: {e}\n")

    print(f"Processing for dataset {dataset_name} completed.")
