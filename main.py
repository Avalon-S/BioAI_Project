from batch_processor import process_all_files
from visualization import generate_performance_plots
import os

if __name__ == "__main__":
    # List of dataset names
    datasets = ["Barnes", "Brandimarte", "Dauzere"]
    # datasets = ["Brandimarte"]
    
    # Base paths for input and output folders
    base_input_folder = "BioAI_Project/data"
    base_output_folder = "BioAI_Project/result"

    # Loop through datasets and process each one
    for dataset_name in datasets:
        input_folder = os.path.join(base_input_folder, dataset_name, "Text")
        output_folder = os.path.join(base_output_folder, dataset_name)
        
        # Step 1: Process all files for the dataset
        process_all_files(input_folder, output_folder, dataset_name)
        
        # Step 2: Generate performance plots for the dataset
        print(f"Generating performance plots for dataset: {dataset_name}")
        generate_performance_plots(base_output_folder, dataset_name)
