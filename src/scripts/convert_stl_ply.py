import os
import pymeshlab
from tqdm import tqdm

def convert_stl_to_ply(directory):
    """
    Converts all .stl files in the specified directory to .ply format using PyMeshLab.
    
    Args:
    - directory (str): Path to the directory containing the .stl files.
    """
    # List all the files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter only .stl files
    stl_files = [f for f in files if f.endswith('.stl')]

    # Loop through each STL file and convert to PLY with a progress bar
    for stl_file in tqdm(stl_files, desc="Converting STL to PLY", unit="file"):
        # Load the mesh using PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(directory, stl_file))
        
        # Define the output file name
        ply_file = os.path.splitext(stl_file)[0] + '.ply'

        # Save the mesh in .ply format
        ms.save_current_mesh(os.path.join(directory, ply_file))


if __name__ == "__main__":
    # Specify the directory containing your .stl files here
    directory_path = "../../datasets/everyday_deform/deformations/Bottle"
    convert_stl_to_ply(directory_path)
