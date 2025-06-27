import os
import time

from modules.utilities import *
from modules.simulation_cases_manager import SimulationCasesManager
from modules.simulation_case import SimulationCase

class SimulationManager:
    def __init__(self):
        self.data_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/"
        self.template_dir = os.path.join(self.data_dir, "templates")
        self.template_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/templates/main.k"
        self.__output_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/"
    
    def generate_all(self, input_keyword_dir: str) -> bool:
        """
        Run the ANSYS simulation for all keywords in the specified directory
        """
        if not os.path.exists(input_keyword_dir):
            print(f"Error: The input keyword directory '{input_keyword_dir}' does not exist.")
            return False
        
        keywords = [k for k in os.listdir(input_keyword_dir) if k.endswith('.k')]
        n_keywords = len(keywords)
        print_info(f"Found {n_keywords} keywords in the directory.")

        failed_keywords = []
        for i in range(n_keywords):
            keyword = keywords[i]
            keyword_path = os.path.join(input_keyword_dir, keyword)
            print_info(f"({i}/{n_keywords}) Processing keyword: {keyword}")
            first = i == 0
            if not self.generate(keyword_path, not first):
                failed_keywords.append(keyword)

        if failed_keywords:
            print_error("Simulation failed for the following keywords:")
            for keyword in failed_keywords:
                print_error(f"{keyword}")
            return False
        
        return True
    
    def generate(self, input_keyword_path: str, wait = False) -> bool:
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
        if self.__run_simulation(case, wait=wait):
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
    def postprocess_all(self) -> bool:
        """
        Postprocess for all keywords in the specified directory
        """
        if not os.path.exists(self.__output_dir):
            print(f"Error: The simulation_results directory '{self.__output_dir}' does not exist.")
            return False
        
        results_directories = os.listdir(self.__output_dir)
        n_results = len(results_directories)
        print_info(f"Found {n_results} results in the results directory.")

        cfile = os.path.join(self.template_dir, "extract_results.cfile")

        failed_results = []
        for i in range(n_results):
            result = results_directories[i]
            result_path = os.path.join(self.__output_dir, result)
            print_info(f"({i}/{n_results}) PostProcessing result: {result}")
            
            # copy_file(cfile, result_path)
            # cfile_path = os.path.join(result_path, "extract_results.cfile")
            # with open(cfile_path, 'r') as cfile_file:
            #     cfile_content = cfile_file.read()
            #     adapted_content = cfile_content.replace("${PATH}", result_path.replace('\\', '/').join("output_data.txt"))
            #     with open(cfile_path, 'w') as cfile_file:
            #         cfile_file.write(adapted_content)
            

        
            if not self.postprocess(result_path):
                failed_results.append(result)

        if failed_results:
            print_error("Postprocessing failed for the following resullts:")
            for result in failed_results:
                print_error(f"{result}")
            return False
        
        return True
    def postprocess(self, results_directory: str) -> bool:
        """
        Postprocess the simulation results for the specified directory
        """
        if not os.path.exists(results_directory):
            print(f"Error: The results directory '{results_directory}' does not exist.")
            return False
        if self.__check_simulation_status(results_directory + "/lsrun.out.txt"):
            print_correct(f"Simulation completed successfully in {results_directory}.")
        else:   
            print_error(f"Cancelling postprocessing: Simulation failed or did not terminate normally in {results_directory}. Check 'lsrun.out.txt' for details.")
            return False
        
        cfile = os.path.join(self.template_dir, "extract_results.cfile")
        copy_file(cfile, results_directory)
        
        cfile_path = os.path.join(results_directory, "extract_results.cfile")
        adapted_content = ""
        with open(cfile_path, 'r') as cfile_file:
            cfile_content = cfile_file.read()
            adapted_content = cfile_content.replace("${PATH}", results_directory.replace('\\', '/') + "/output_data.txt")
        with open(cfile_path, 'w') as cfile_file:
            cfile_file.write(adapted_content)
        print_info(f"Postprocessing results in {results_directory}...")
        ansys_lspp_path = '"C:/Program Files/LS-DYNA Suite R14 Student/lspp/lsprepost4.10_x64.exe"'
        command = ansys_lspp_path + f' -nographics c="{cfile_path}"'
        command = 'start "" ' + command  # Use start to run in a new window
        start = time.perf_counter()
        os.system(command)
        elapsed = time.perf_counter() - start       
        print_correct(f"Postprocessing completed in {elapsed:.2f} seconds.")

        print_info(f"Postprocessing completed for {results_directory}.")
        return True
        


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
    
    
    
    def __run_simulation(self, case: SimulationCase, wait = False) -> bool:
        """
        Run the ANSYS simulation with the specified parameters
        """
        if case.is_valid() == True:
            print_info(f"Running simulation with {case.keyword_path}, {case.n_cores} cores, and {case.memory} MB memory...")
            print_info(f"Parameters: {case.get_parameters_string()}")
            # ansys_ls_dyna_path = '"C:/Program Files/LS-DYNA Suite R14 Student/lsdyna/ls-dyna_smp_d_R14.1.1s_1-gef50e1efb1_winx64_ifort190.exe"'
            ansys_ls_run_path = 'start "" "C:/Program Files/LS-DYNA Suite R14 Student/lspp/LS-Run/lsrun.exe"'
            command = ansys_ls_run_path + f' -input {case.keyword_path} -submit'
            if wait:
                command += ' -wait -1'
            # command = ansys_ls_dyna_path + f' i={case.keyword_path} ncpu={case.n_cores} memory={case.memory}m > lsrun.out.txt 2>&1'
            start = time.perf_counter()
            os.system(command)
            elapsed = time.perf_counter() - start
            time.sleep(5)
            return True
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
