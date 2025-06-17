import os
from modules.utilities import *
from modules.simulation_case import SimulationCase

class KeywordManager:
    def __init__(self):
        print("Initializing KeywordManager...")

    
    def manage_keyword(self, input_keyword_path: str, case: SimulationCase, output_directory: str) -> bool:
        """
        Manage the keywords for the simulation
        """
        if not os.path.exists(input_keyword_path):
            print_error(f"Error: The input keyword file '{input_keyword_path}' does not exist.")
            return False
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Copy the keyword file to the output directory
        keyword_name = os.path.basename(input_keyword_path)
        output_keyword_path = os.path.join(output_directory, keyword_name)

        copy_succes = copy_file(input_keyword_path, output_keyword_path)
        if not copy_succes:
            print_error(f"Failed to copy keyword file '{keyword_name}' to '{output_directory}'.")
            return False
        
        if self.__update_keyword(output_keyword_path, case):
            print_correct(f"Keyword file '{output_keyword_path}' managed successfully.")
            return True
        else:
            print_error(f"Failed to update keyword file '{output_keyword_path}'.")
            return False
    
    def __update_keyword(self, keyword_path: str, case: SimulationCase) -> bool:
        """
        Update the keyword file with the parameters from the simulation case
        """
        # TODO: Implement the logic to update the keyword file with the parameters from the simulation case

        if not os.path.exists(keyword_path):
            print_error(f"Error: The keyword file '{keyword_path}' does not exist.")
            return False
        
        try:
            with open(keyword_path, 'a') as f:
                f.write("\n*PARAMETERS\n")
                for key, value in case.get_parameters().items():
                    f.write(f"{key}={value}\n")

            case.keyword_path = keyword_path  # Update the case with the new keyword path
            case.add_parameter("keyword_path", keyword_path)  # Update the case parameters
            print_correct(f"Keyword file '{keyword_path}' updated successfully.")
            return True
        except Exception as e:
            print_error(f"Error updating keyword file: {e}")
            return False
    

    