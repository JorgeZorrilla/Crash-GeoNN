from modules.keyword_diagnostics import *
from modules.simulation_manager import SimulationManager
from modules.data_postprocessing import DataPostProcessing
import os


# Current approach to generate all simulations in a directory
sim_manager = SimulationManager()


# STEP 1: Generate
# keywords_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/data/b-pillar"
# sim_manager.generate_all(keywords_path)

# STEP 2: Get simulation results from the 3dplots
sim_manager.postprocess_all()

    