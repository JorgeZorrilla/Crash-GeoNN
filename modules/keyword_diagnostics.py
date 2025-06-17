import os
from modules.utilities import *

class KeywordDiagnostics:
    def __init__(self):
        print("Initializing KeywordDiagnostics...")
    
    def check_keyword_directory(self, keyword_directory: str) -> bool:
        """
        Check if the keyword directory exists and is valid.
        """
        if not os.path.exists(keyword_directory):
            print(f"Error: The keyword directory '{keyword_directory}' does not exist.")
            return False
        
        if not os.path.isdir(keyword_directory):
            print(f"Error: The path '{keyword_directory}' is not a directory.")
            return False
        
        keyword_files = [f for f in os.listdir(keyword_directory) if f.endswith('.k')]
        if not keyword_files:
            print(f"Error: No keyword files found in the directory '{keyword_directory}'.")
            return False
        for keyword_file in keyword_files:
            keyword_path = os.path.join(keyword_directory, keyword_file)
            if not self.check_keyword_file(keyword_path):
                print(f"Error: The keyword file '{keyword_file}' is invalid.")
                return False
        return True

    def check_keyword_file(self, keyword_path: str) -> bool:
        """
        Check if the keyword file exists and is valid.
        """
        if not os.path.exists(keyword_path):
            print_error(f"Error: The keyword file '{keyword_path}' does not exist.")
            return False
        with open(keyword_path, 'r') as f:
            content = f.read()
            if not content.strip():
                print_error(f"Error: The keyword file '{keyword_path}' is empty.")
                return False
            
            if "BOUNDARY_SPC_NODE" not in content:
                print_error(f"Error: The keyword file '{keyword_path}' does not contain 'BOUNDARY_SPC_NODES'.")
                return False
            if "2         2         2         0         0         0         0         0         0" not in content and "2         2         2         0         0         0         0         0" not in content:
                print_error(f"Error: The keyword file '{keyword_path}' does not contain the expected part properties")
                return False
            if "LSHELL" in content:
                print_error(f"Error: The keyword file '{keyword_path}' contains 'LSHELL', so the part name was not changed.")
                return False
        
        # Additional checks can be added here (e.g., file format, content validation)
        
        print_correct(f"Keyword '{keyword_path}' is valid.")
        return True
    
if __name__ == "__main__":
    input_keyword_path = "C:/Users/jorge/Documents/M2i/TFM/data/B-Pillar"
    diagnostics = KeywordDiagnostics()
    if diagnostics.check_keyword_directory(input_keyword_path):
        print_correct("Keyword directory is valid.")
    else:
        print_error("Keyword directory is invalid.")