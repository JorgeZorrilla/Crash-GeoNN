from modules.utilities import *
from modules.keyword_manager import KeywordManager, SimulationCase

class SimulationCasesManager: 
    def __init__(self, keyword_path: str, n_cores: int = 4, memory: int = 20):
        self.keyword_path = keyword_path
        self.n_cores = n_cores
        self.memory = memory
        self.keyword_manager = KeywordManager()
        self.simulation_cases = []

    def get_available_cases(self) -> list:
        """
        Get the list of simulation cases
        """
        return self.simulation_cases
    
    def generate_available_cases(self) -> bool:
        """
        Generate simulation cases based on the provided parameters
        """
        # TODO: Implement a more complex parameter generation logic
        return True
    
    def manage_keyword(self, input_keyword_path: str, case: SimulationCase, output_path: str) -> bool:
        """
        Manage the keyword for the simulation case
        """
        return self.keyword_manager.manage_keyword(input_keyword_path, case, output_path)

    

        

        