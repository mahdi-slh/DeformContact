import os
import shutil
import pymeshlab
from tqdm import tqdm

def convert_stl_to_ply_and_copy_json(input_directory, output_directory):
    """
    Converts all .stl files in the specified directory to .ply format using PyMeshLab
    and saves them in a different output directory. Also copies corresponding .json files.
    
    Args:
    - input_directory (str): Path to the directory containing the .stl files.
    - output_directory (str): Path to the directory where .ply files and .json files will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all the files in the directory
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # Filter only .stl files
    stl_files = [f for f in files if f.endswith('.stl')]

    # Loop through each STL file, convert to PLY, and copy JSON with a progress bar
    for stl_file in tqdm(stl_files, desc="Processing files", unit="file"):
        # Load the mesh using PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(input_directory, stl_file))
        
        # Define the output file name
        base_name = os.path.splitext(stl_file)[0]
        ply_file = base_name + '.ply'
        json_file = base_name + '.json'

        # Save the mesh in .ply format
        ms.save_current_mesh(os.path.join(output_directory, ply_file))

        # Check if the corresponding JSON file exists and copy it
        json_path = os.path.join(input_directory, json_file)
        if os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(output_directory, json_file))

if __name__ == "__main__":
    objects_file = '../../datasets/everyday_deform/objects.txt'
    base_input_directory = '../../datasets/everyday_deform/deformations_unity/'
    base_output_directory = '../../datasets/everyday_deform/deformations/'

    # Read the object names from the file
    with open(objects_file, 'r') as file:
        object_names = file.read().splitlines()

    for object_name in object_names:
        input_directory = os.path.join(base_input_directory, object_name)
        output_directory = os.path.join(base_output_directory, object_name)

        # Check if the directory exists before processing
        if os.path.exists(input_directory):
            convert_stl_to_ply_and_copy_json(input_directory, output_directory)
        else:
            print(f"Directory for {object_name} does not exist. Skipping processing.")
