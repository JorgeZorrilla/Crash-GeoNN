import os
import time

from modules.utilities import *
from modules.simulation_cases_manager import SimulationCasesManager
from modules.simulation_case import SimulationCase

class SimulationManager:
    def __init__(self):
        self.data_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/"
        self.template_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/templates/main.k"
        self.__output_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/"
    
   
    def generate(self, input_keyword_path: str) -> bool:
        """
        Run the ANSYS simulation for the selected geometry and parameters
        """
        if not os.path.exists(input_keyword_path):
            print(f"Error: The input keyword file '{input_keyword_path}' does not exist.")
            return False
        input_keyword = os.path.basename(input_keyword_path).replace('.k', '')
        output_directory = self.__prepare_output_directory(input_keyword, clean=True)

        output_file_path = os.path.join(output_directory, "main.k")
        with open(self.template_path, 'r') as template_file:
            with open(output_file_path, 'w') as output_file:
                template_content = template_file.read()
                adapted_content = template_content.replace("${GEOMETRY_KEYWORD_PATH}", input_keyword_path.replace('\\', '/'))  # Ensure the path is in the correct format
                output_file.write(adapted_content)
        
        os.chdir(output_directory) # Change to the simulation directory to save results
        case = SimulationCase(
            keyword_path=output_file_path,
            n_cores=4,  # Default number of cores
            memory=20  # Default memory in MB
        )
        if self.__run_simulation(case):
            print_correct(f"Simulation for keyword '{input_keyword}' completed successfully.")
            return True
        else:
            print_error(f"Simulation for keyword '{input_keyword}' failed.")
            return False


        # case_manager = SimulationCasesManager(input_keyword_path, output_directory, n_cores=4, memory=20)
        # if case_manager.generate_available_cases():
        #     available_cases = case_manager.get_available_cases()
        #     if not available_cases:
        #         print_error("No available simulation cases found. Please generate cases first.")
        #         return False
            
        #     successfull_cases = []
        #     failed_cases = []

        #     n_cases = len(available_cases)
        #     print_info(f"Starting simulation for {n_cases} cases for keyword '{input_keyword}'.")
        #     for i in range(n_cases):
        #         print_info(f"Case {i+1}/{n_cases} for keyword '{input_keyword}'...")

        #         sim_path = os.path.join(output_directory, f"{i}")

        #         create_clean_dir(sim_path)
        #         os.chdir(sim_path) # Change to the simulation directory to save results

        #         case = available_cases[i]

        #         if self.__run_simulation(case):
        #             successfull_cases.append(case)
        #         else:
        #             print(f"Error: Simulation failed for {input_keyword} with parameter {i}.")
        #             failed_cases.append(case)

        #         os.chdir(output_directory) # Change back to the output directory 

        #     self.__create_report(output_directory, successfull_cases, failed_cases)          
        #     os.chdir(output_directory) # Change back to the output directory
        #     return True
        # else:  
        #     print_error("Failed to generate available simulation cases.")
        #     return False

    def __create_report(self, output_directory: str, succesfull_cases: list, failed_cases: list) -> None:
        """
        Create a report of the simulation results
        """
        n_success = len(succesfull_cases)
        n_failed = len(failed_cases)
        n_total = n_success + n_failed
        report_path = os.path.join(output_directory, "simulation_report.txt")
        with open(report_path, 'w') as report_file:
            report_file.write("Simulation Report\n")
            report_file.write("=================\n\n")
            report_file.write(f"Total Cases: {n_total}\n")
            report_file.write(f"Successful Cases: {n_success}/{n_total}({n_success/n_total}%)\n")
            report_file.write(f"Failed Cases: {n_failed}/{n_total}({n_failed/n_total}%)\n")

            if succesfull_cases:
                report_file.write("Successful Cases:\n")
                for case in succesfull_cases:
                    report_file.write(f"- {case.get_parameters_string()}\n")

            if failed_cases:
                report_file.write("\nFailed Cases:\n")
                for case in failed_cases:
                    report_file.write(f"- {case.get_parameters_string()}\n")

        print_correct(f"Report created at {report_path}")

    def __prepare_output_directory(self, keyword_name: str, clean: bool = True) -> str:
        """
        Prepare the output directory for simulation results
        """
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)
        output_path = os.path.join(self.__output_dir, keyword_name).replace('.k', '')
        create_clean_dir(output_path)
        return output_path
    
    
    
    def __run_simulation(self, case: SimulationCase) -> bool:
        """
        Run the ANSYS simulation with the specified parameters
        """
        if case.is_valid() == True:
            print_info(f"Running simulation with {case.keyword_path}, {case.n_cores} cores, and {case.memory} MB memory...")
            print_info(f"Parameters: {case.get_parameters_string()}")
            ansys_ls_dyna_path = '"C:/Program Files/LS-DYNA Suite R14 Student/lsdyna/ls-dyna_smp_d_R14.1.1s_1-gef50e1efb1_winx64_ifort190.exe"'
            command = ansys_ls_dyna_path + f' i={case.keyword_path} ncpu={case.n_cores} memory={case.memory}m > lsrun.out.txt 2>&1'
            start = time.perf_counter()
            os.system(command)
            elapsed = time.perf_counter() - start
            if self.__check_simulation_status("lsrun.out.txt"):
                print_correct(f"Simulation completed successfully in {elapsed:.2f} seconds.")
                return True
            else:
                print_error(f"Simulation failed or did not terminate normally. Check 'lsrun.out.txt' for details.")
                return False
        return False
        

    def __check_simulation_status(self, path: str) -> bool:
        """
        Check the status of the simulation
        """
        if not os.path.exists(path):
            print(f"Error: The simulation output file '{path}' does not exist.")
            return False
        else:
            with open(path, 'r') as file:
                content = file.read()
                return "N o r m a l    t e r m i n a t i o n" in content


if __name__ == "__main__":
    input_keyword_path = "C:/Users/jorge/Documents/M2i/TFM/data/Mallas/BaseModelKeyword.k"

    # Example usage
    sim_manager = SimulationManager()
    sim_manager.generate(input_keyword_path)
