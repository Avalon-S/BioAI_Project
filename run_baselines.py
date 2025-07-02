import argparse
import yaml
import os
from processor.batch_processor import process_all_files
from visualization.performance_plots import generate_performance_plots

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run FJSP experiment and visualization from YAML config.")
    parser.add_argument("--config", type=str, default="config/config_baselines.yaml", help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, help="Override output folder from config")
    parser.add_argument("--input", type=str, help="Override input folder from config")
    parser.add_argument("--datasets", nargs="+", help="Override dataset list from config")

    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded config:")
    print(config)

    base_input_folder = args.input if args.input else config["input"]
    base_output_folder = args.output if args.output else config["output"]
    datasets = args.datasets if args.datasets else config["datasets"]
    algorithms = config.get("algorithm", ["Standard NSGA-II", "Advanced NSGA-II"])  # # Default

    
    for dataset_name in datasets:
        input_folder = os.path.join(base_input_folder, dataset_name, "Text")
        output_folder = os.path.join(base_output_folder, dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        # Step 1: Process all files for the dataset
        process_all_files(input_folder, output_folder, dataset_name, algorithms)
        
        # Step 2: Generate performance plots for the dataset
        print(f"Generating performance plots for dataset: {dataset_name}")
        generate_performance_plots(base_output_folder, dataset_name)

if __name__ == "__main__":
    main()
