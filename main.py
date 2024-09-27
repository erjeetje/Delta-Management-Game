# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import matplotlib.style
from PyQt5.QtWidgets import QApplication

matplotlib.style.use("fast")

import demo_visualizations as visualizer

sys.path.insert(1, r'C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 Network model Bouke\version 4.3.4\IMSIDE netw\mod 4.3.4 netw')
import runfile_td_v1 as imside_model

def main():
    #model = imside_model.IMSIDE()
    #model_output = model.output
    #print(model_output.head())
    model_scenarios, game_scenarios, world_bbox, game_bbox, salinity_range, water_level_range, water_velocity_range = visualizer.load_scenarios()
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)
    time_steps = list(sorted(set(model_scenarios["time"])))
    time_index = 0
    starting_scenario = "0_0mzss_2000m3s"
    starting_variable = "water_salinity"
    viz_tracker = visualizer.VisualizationTracker(
        starting_scenario=starting_scenario, starting_variable=starting_variable,
        time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range,
        water_level_range=water_level_range, water_velocity_range=water_velocity_range)
    colorbar_salinity, colorbar_water_level, colorbar_water_velocity = visualizer.load_images()
    gui = visualizer.ApplicationWindow(
        scenarios=model_scenarios, viz_tracker=viz_tracker, bbox=world_bbox,
        salinity_colorbar_image=colorbar_salinity, water_level_colorbar_image=colorbar_water_level,
        water_velocity_image=colorbar_water_velocity)
    side_window = visualizer.GameVisualization(
        scenarios=game_scenarios, viz_tracker=viz_tracker, bbox=game_bbox)
    gui.show()
    side_window.show()
    gui.activateWindow()
    gui.raise_()
    qapp.exec()
    #locations = model_locations()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()