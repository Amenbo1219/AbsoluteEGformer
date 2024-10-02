import os
import shutil
from PIL import Image

def check_required_files(dir_path):
    required_files = [
        os.path.join(dir_path, 'panorama', 'full', 'depth.png'),
        os.path.join(dir_path, 'panorama', 'full', 'rgb_rawlight.png')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
        
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return False
    
    return True

def clean_directory(root_dir):
    for scene_dir in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene_dir)
        if os.path.isdir(scene_path):
            rendering_dir = os.path.join(scene_path, '2D_rendering')
            if os.path.exists(rendering_dir):
                for render_num in os.listdir(rendering_dir):
                    render_path = os.path.join(rendering_dir, render_num)
                    if os.path.isdir(render_path):
                        if not check_required_files(render_path):
                            print(f"Removing directory: {render_path}")
                            shutil.rmtree(render_path)

if __name__ == "__main__":
    directories = [
        "/workspace/datasets/Structured3D_SRC/train",
        "/workspace/datasets/Structured3D_SRC/test",
        "/workspace/datasets/Structured3D_SRC/val"
    ]

    for directory in directories:
        clean_directory(directory)