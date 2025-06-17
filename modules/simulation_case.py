import os
from modules.utilities import  print_correct, print_error, print_info
class SimulationCase:
    def __init__(self, keyword_path: str, n_cores: int = 4, memory: int = 20):
        self.keyword_path = keyword_path
        self.n_cores = n_cores
        self.memory = memory
        self.parameters = {
            "keyword_path": keyword_path,
            "n_cores": n_cores,
            "memory": memory
        }

    def add_parameter(self, key: str, value):
        """
        Add a parameter to the simulation case
        """
        self.parameters[key] = value

    def get_parameters(self) -> dict:
        """
        Get the parameters of the simulation case
        """
        return self.parameters
    
    def get_parameters_string(self) -> str:
        """
        Get the parameters of the simulation case as a string
        """
        return ', '.join([f"{key}={value}" for key, value in self.parameters.items()])
    
    def is_valid(self) -> bool:
        """
        Check if the simulation case is valid
        """
        if not os.path.exists(self.keyword_path):
            print_error(f"Error: The keyword file '{self.keyword_path}' does not exist.")
            return False
        if self.n_cores <= 0 or self.n_cores > 4:
            print_error("Error: Number of cores must be greater than 0 and less than or equal to 4.")
            return False
        if self.memory <= 0:
            print("Error: Memory must be greater than 0.")
            return False
        return True