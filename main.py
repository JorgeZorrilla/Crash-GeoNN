from modules.keyword_diagnostics import *
from modules.simulation_manager import SimulationManager
from modules.data_postprocessing import DataPostProcessing
import os

# input_keyword_path = "C:/Users/jorge/Documents/M2i/TFM/data/B-Pillar"
# diagnostics = KeywordDiagnostics()
# if diagnostics.check_keyword_directory(input_keyword_path):
#     print_correct("Keyword directory is valid.")
# else:
#     print_error("Keyword directory is invalid.")

# input_keyword_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/b-pillar/Geometry-0.k"

# # Example usage
# sim_manager = SimulationManager()
# sim_manager.generate(input_keyword_path)


# Current approach to generate all simulations in a directory
sim_manager = SimulationManager()


# STEP 1: Generate
# keywords_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/b-pillar"
# sim_manager.generate_all(keywords_path)

# STEP 2: Get simulation results from the 3dplots
sim_manager.postprocess_all()

# STEP 3: Generate the BBDD
# postprocessing = DataPostProcessing()
# input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/Geometry-037/output_data.txt"
# postprocessing.post_process(input_file)




# keywords = os.listdir(keywords_path)
# keywords = [k for k in keywords if k.endswith('.k')]
# n_keywords = len(keywords)
# print_info(f"Found {n_keywords} keywords in the directory.")
# count = 0
# failed_keywords = []
# for keyword in keywords:
#     keyword_path = os.path.join(keywords_path, keyword)
#     print_info(f"({count}/{n_keywords})Processing keyword: {keyword}")
#     if  not sim_manager.generate(keyword_path):
#             failed_keywords.append(keyword)

# if failed_keywords:
#     print_error(f"Simulation failed for the following keywords: ")
#     for keyword in failed_keywords:
#         print_error(f"{keyword}")
    