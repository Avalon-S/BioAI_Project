import os
import shutil

def remove_ipynb_checkpoints(start_dir="."):
    """
    Recursively remove all `.ipynb_checkpoints` folders in the given directory.

    Args:
        start_dir (str): The starting directory to search for `.ipynb_checkpoints`.
    """
    for root, dirs, files in os.walk(start_dir):
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                folder_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed: {folder_path}")
                except Exception as e:
                    print(f"Failed to remove {folder_path}: {e}")


def clear_result_subfolders(result_folder, subfolders):
    """
    Clear all files and subfolders inside specified subfolders in the result folder.

    Args:
        result_folder (str): Path to the result folder.
        subfolders (list): List of subfolder names to clear (e.g., ["Barnes", "Brandimarte", "Dauzere"]).
    """
    for subfolder in subfolders:
        subfolder_path = os.path.join(result_folder, subfolder)
        if os.path.exists(subfolder_path):
            for item in os.listdir(subfolder_path):
                item_path = os.path.join(subfolder_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)  # Remove file or symbolic link
                        print(f"Removed file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)  # Remove subdirectory
                        print(f"Removed folder: {item_path}")
                except Exception as e:
                    print(f"Failed to remove {item_path}: {e}")
        else:
            print(f"Subfolder {subfolder_path} does not exist.")

# Run the function
if __name__ == "__main__":
    # Step 1: Remove all `.ipynb_checkpoints` folders
    remove_ipynb_checkpoints(".")
    
    # Step 2: Clear result subfolders
    result_folder = "BioAI_Project/result"
    subfolders_to_clear = ["Barnes", "Brandimarte", "Dauzere"]
    clear_result_subfolders(result_folder, subfolders_to_clear)
