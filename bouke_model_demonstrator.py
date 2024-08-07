# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import pandas as pd
from copy import deepcopy
from PyQt5.QtWidgets import QApplication
import demo_visualizations as visualizer
import load_functions as load_files
import model_to_game as game_sync
import transform_functions as transform_func

sys.path.insert(1, r'C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 Network model Bouke\version 4.3.4\IMSIDE netw\mod 4.3.4 netw')
import runfile_td_v1 as imside_model

class DMG():
    def __init__(self):
        self.load_paths()
        self.load_model()
        self.load_shapes()
        self.transform_functions()
        self.build_game_network()
        model_output_df = self.run_model()
        self.model_output_to_game(model_output_df, initialize=True)
        print("we are here")
        return

    def update(self):
        model_output_df = self.run_model(turn=2)
        self.model_output_to_game(model_output_df)


    def load_paths(self):
        """
        set any core paths that need to be accessed.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_files = os.path.join(dir_path, "input_files")
        self.save_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 DMG design\coding (notebooks)\live demonstrator coding"
        return

    def load_model(self):
        """
        load the IMSIDE model and extract network.
        """
        self.model = imside_model.IMSIDE()
        model_network_df = self.model.network
        self.model_network_gdf = game_sync.process_model_network(model_network_df)
        return

    def load_shapes(self):
        """
        load polygons (world map) and hexagon shapes (board).
        """
        self.world_polygons = load_files.read_json_features(filename='hexagon_shapes_in_layers_Bouke_network.json',
                                                       path=self.input_files)
        self.game_hexagons = load_files.read_geojson(filename='hexagons_clean0.geojson', path=self.input_files)
        return

    def transform_functions(self):
        """
        create transform functions and transform polygons to correct (world) coordinates
        """
        self.transform_calibration = transform_func.create_calibration_file(self.world_polygons)
        self.world_polygons = transform_func.transform(self.world_polygons, self.transform_calibration,
                                                       export="warped", path="")
        self.game_hexagons = game_sync.find_neighbours(self.game_hexagons)
        self.world_polygons = game_sync.match_hexagon_properties(self.world_polygons, self.game_hexagons,
                                                                 "neighbours")
        return

    def build_game_network(self):
        self.game_hexagons = game_sync.find_neighbour_edges(self.game_hexagons)
        self.world_polygons, self.model_network_gdf = game_sync.find_branch_intersections(deepcopy(self.world_polygons),
                                                                                self.model_network_gdf)
        self.game_hexagons = game_sync.match_hexagon_properties(deepcopy(self.game_hexagons), self.world_polygons,
                                                           ["branches", "branch_crossing"])
        self.model_network_gdf = game_sync.determine_polygon_intersections(self.model_network_gdf, self.world_polygons)
        self.game_network_gdf = game_sync.draw_branch_network(self.game_hexagons, self.model_network_gdf)
        return

    def run_model(self, turn=1):
        self.model.run_model(turn=turn)
        return self.model.output

    def model_output_to_game(self, model_output_df, initialize=False):
        double_exploded_output_df, exploded_output_df = game_sync.process_model_output(model_output_df)
        if initialize == True:
            model_output_gdf = game_sync.output_df_to_gdf(double_exploded_output_df)
            self.model_output_gdf = game_sync.add_polygon_ids(model_output_gdf, self.world_polygons)
            # NOTE: first point in Lek is double ? check source
            # NOTE 2: first and last points all match earlier runs, but there are less points in Lek ? check source
            self.game_output_gdf = game_sync.model_output_to_game_locations(self.game_network_gdf,
                                                                            self.model_output_gdf, exploded_output_df)
            timestep_0 = self.model_output_gdf.iloc[0]["time"]
            self.model_output_ref_gdf = self.model_output_gdf.loc[self.model_output_gdf['time'] == timestep_0]
            self.model_output_ref_gdf = self.model_output_ref_gdf.drop(columns=["time", "sb_st"])
            self.game_output_ref_gdf = self.game_output_gdf.loc[self.game_output_gdf['time'] == timestep_0]
            self.game_output_ref_gdf = self.game_output_ref_gdf.drop(columns=["time", "sb_st"])
            self.model_output_gdf = game_sync.output_to_timeseries(self.model_output_gdf, scenario="2100he")
            self.game_output_gdf = game_sync.output_to_timeseries(self.game_output_gdf, scenario="2100he")
        else:
            output_to_merge_df = double_exploded_output_df[["id", "time", "sb_st"]]
            self.model_output_gdf = self.model_output_ref_gdf.merge(output_to_merge_df, on="id")
            self.game_output_gdf = self.game_output_ref_gdf.merge(output_to_merge_df, on="id")
            self.model_output_gdf = game_sync.output_to_timeseries(self.model_output_gdf, scenario="2100le")
            self.game_output_gdf = game_sync.output_to_timeseries(self.game_output_gdf, scenario="2100le")
        if True:
            self.model_output_gdf.to_excel(os.path.join(self.save_path, "model_network_gdf.xlsx"), index=False)
            self.game_output_gdf.to_excel(os.path.join(self.save_path, "game_network_gdf.xlsx"), index=False)
        return




def main():
    game = DMG()
    print("initiliazed")
    game.update()
    print("updated")







    """
    model_scenarios, game_scenarios, world_bbox, game_bbox, salinity_range = load_files.load_scenarios() # , water_level_range, water_velocity_range
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)
    time_steps = list(sorted(set(model_scenarios["time"])))
    time_index = 0
    starting_scenario = "2017"
    starting_variable = "water_salinity"
    viz_tracker = visualizer.VisualizationTracker(
        starting_scenario=starting_scenario, starting_variable=starting_variable,
        time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range)
    # ,water_level_range = water_level_range, water_velocity_range = water_velocity_range
    colorbar_salinity, colorbar_water_level, colorbar_water_velocity = load_files.load_images()
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
    """


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()