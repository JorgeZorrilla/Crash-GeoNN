from modules.keyword_diagnostics import *
from modules.simulation_manager import SimulationManager
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


sim_manager = SimulationManager()

keywords_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/b-pillar"
keywords = os.listdir(keywords_path)
keywords = [k for k in keywords if k.endswith('.k')]
n_keywords = len(keywords)
print_info(f"Found {n_keywords} keywords in the directory.")
count = 0
failed_keywords = []
for keyword in keywords:
    keyword_path = os.path.join(keywords_path, keyword)
    print_info(f"({count}/{n_keywords})Processing keyword: {keyword}")
    if  not sim_manager.generate(keyword_path):
            failed_keywords.append(keyword)

if failed_keywords:
    print_error(f"Simulation failed for the following keywords: ")
    for keyword in failed_keywords:
        print_error(f"{keyword}")
    