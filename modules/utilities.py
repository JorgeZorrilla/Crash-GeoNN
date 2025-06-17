import os
import shutil

def create_clean_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"Cleaning  directory: {path}")
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

def copy_file(src: str, dst: str):
    """
    Copy a file from src to dst
    """
    try:
        shutil.copy(src, dst)
        print(f"File copied from {src} to {dst}")
    except Exception as e:
        print(f"Error copying file: {e}")

# ANSI code colors
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


def print_correct(text : str):
    print(f"{GREEN}{text}{RESET}")

def print_error(text : str):
    print(f"{RED}{text}{RESET}")

def print_info(text : str):
    print(f"{BLUE}{text}{RESET}")