# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import time
import pandas as pd
from datetime import timedelta
from copy import deepcopy
from PyQt5.QtWidgets import QApplication
import demo_visualizations as visualizer
import load_functions as load_files
import model_to_game as game_sync
import transform_functions as transform_func
import demo_processing

sys.path.insert(1, r'C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 Network model Bouke\version 4.3.4\IMSIDE netw\mod 4.3.4 netw')
import runfile_td_v1 as imside_model

class DMG():
    def __init__(self):
        self.load_paths()
        self.load_model()
        self.load_shapes()
        self.transform_functions()
        self.build_game_network()
        self._turn = 1
        self._scenario = "2017"
        model_output_df = self.run_model()
        self.model_output_to_game(model_output_df, initialize=True)
        #self.create_visualizations()
        print("we are here")
        return

    @property
    def turn(self):
        return self._turn
    @turn.setter
    def turn(self, turn):
        self._turn = turn
        return

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario
        return

    def update(self):
        """
        function that handles running the model and retrieving the model output.
        """
        model_output_df = self.run_model()
        # to test if this overrides values or not, otherwise adjust code in the function below to remove any values
        # from the same scenario of this exist (for logging purposes, perhaps do store those somewhere).
        self.model_output_to_game(model_output_df, scenario=self.scenario)
        if self.turn == 4:
            self.create_visualizations()
        return

    def update_forcings(self):
        """
        function that handles updating the model forcings to subsequent turns (final scenario forcings to be determined)
        """
        if self.turn == 1:
            self.scenario = "2017"
        elif self.turn == 2:
            self.scenario = "2018"
        elif self.turn == 3:
            self.scenario = "2100le"
        elif self.turn == 4:
            self.scenario = "2100he"
        else:
            print("unsupported turn")
        self.model.change_forcings(scenario=self.scenario)
        return

    def add_sea_level_rise(self, slr):
        self.model.add_sea_level_rise(slr)
        return

    def update_channel_geometries(self, change_type):
        """
        function that handles any players actions to update the correct geometry in the model. Currently hard-coded,
        will be updated with the live board link.
        """
        if self.turn == 1:
            return
        elif self.turn == 2:
            channels_to_update = ["Nieuwe Waterweg v2"]
            for channel in channels_to_update:
                self.model.change_channel_geometry(channel, change_type=change_type)
        elif self.turn == 3:
            channels_to_update = ["Nieuwe Maas 1 old", "Nieuwe Maas 2 old"]
            for channel in channels_to_update:
                self.model.change_channel_geometry(channel, change_type=change_type)
        elif self.turn == 4:
            channels_to_update = ["Oude Maas 1", "Oude Maas 2", "Oude Maas 3", "Oude Maas 4"]
            for channel in channels_to_update:
                self.model.change_channel_geometry(channel, change_type=change_type)
        else:
            print("unsupported turn, no change to any geometry")
        return


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
        create transform functions and transform polygons to correct (world) coordinates.
        """
        self.transform_calibration = transform_func.create_calibration_file(self.world_polygons)
        self.world_polygons = transform_func.transform(self.world_polygons, self.transform_calibration,
                                                       export="warped", path="")
        self.game_hexagons = game_sync.find_neighbours(self.game_hexagons)
        self.world_polygons = game_sync.match_hexagon_properties(self.world_polygons, self.game_hexagons,
                                                                 "neighbours")
        return

    def build_game_network(self):
        """
        function that calls all functions to process the model network to the game network on a regular grid.
        """
        self.game_hexagons = game_sync.find_neighbour_edges(self.game_hexagons)
        self.world_polygons, self.model_network_gdf = game_sync.find_branch_intersections(deepcopy(self.world_polygons),
                                                                                          self.model_network_gdf)
        self.game_hexagons = game_sync.match_hexagon_properties(deepcopy(self.game_hexagons), self.world_polygons,
                                                                ["branches", "branch_crossing"])
        self.model_network_gdf = game_sync.determine_polygon_intersections(self.model_network_gdf, self.world_polygons)
        self.game_network_gdf = game_sync.draw_branch_network(self.game_hexagons, self.model_network_gdf)
        return

    def run_model(self, scenario="2017"):
        """
        call the function run_model function of the model object and retrieve the output.
        """
        self.model.run_model()
        return self.model.output

    def model_output_to_game(self, model_output_df, initialize=False, scenario="2017"):
        """
        function to process the model output to the model and game output geodataframes respectively.
        """
        start_time = time.perf_counter()
        double_exploded_output_df, exploded_output_df = game_sync.process_model_output(model_output_df)
        if initialize == True:
            model_output_gdf = game_sync.output_df_to_gdf(double_exploded_output_df)
            model_output_gdf = game_sync.add_polygon_ids(model_output_gdf, self.world_polygons)

            # NOTE: first point in Lek is double ? check source
            # NOTE 2: first and last points all match earlier runs, but there are less points in Lek ? check source
            self.game_output_gdf = game_sync.model_output_to_game_locations(self.game_network_gdf,
                                                                            model_output_gdf, exploded_output_df)
            model_output_gdf = model_output_gdf.reset_index()
            self.model_output_gdf = model_output_gdf.drop(columns="index")
            timestep_0 = self.model_output_gdf.iloc[0]["time"]
            self.model_output_ref_gdf = self.model_output_gdf.loc[self.model_output_gdf['time'] == timestep_0]
            self.model_output_ref_gdf = self.model_output_ref_gdf.drop(columns=["time", "sb_st"])
            self.game_output_ref_gdf = self.game_output_gdf.loc[self.game_output_gdf['time'] == timestep_0]
            self.game_output_ref_gdf = self.game_output_ref_gdf.drop(columns=["time", "sb_st"])
            self.model_output_gdf = game_sync.output_to_timeseries(self.model_output_gdf, scenario=scenario)
            self.game_output_gdf = game_sync.output_to_timeseries(self.game_output_gdf, scenario=scenario)
            duration = timedelta(seconds=time.perf_counter() - start_time)
            print('Initial output processing took: ', duration)
        else:
            """
            # The code below overrides the output GeoDataFrames
            output_to_merge_df = double_exploded_output_df[["id", "time", "sb_st"]]
            self.model_output_gdf = self.model_output_ref_gdf.merge(output_to_merge_df, on="id")
            self.game_output_gdf = self.game_output_ref_gdf.merge(output_to_merge_df, on="id")
            self.model_output_gdf = game_sync.output_to_timeseries(self.model_output_gdf, scenario="2100le")
            self.game_output_gdf = game_sync.output_to_timeseries(self.game_output_gdf, scenario="2100le")
            print(self.model_output_gdf)
            """

            # The code below appends the output GeoDataFrames
            output_to_merge_df = double_exploded_output_df[["id", "time", "sb_st"]]
            model_output_gdf = self.model_output_ref_gdf.merge(output_to_merge_df, on="id")
            game_output_gdf = self.game_output_ref_gdf.merge(output_to_merge_df, on="id")
            model_output_gdf = game_sync.output_to_timeseries(model_output_gdf, scenario=scenario)
            game_output_gdf = game_sync.output_to_timeseries(game_output_gdf, scenario=scenario)
            self.model_output_gdf = pd.concat([self.model_output_gdf, model_output_gdf])
            self.game_output_gdf = pd.concat([self.game_output_gdf, game_output_gdf])
            duration = timedelta(seconds=time.perf_counter() - start_time)
            print('Update output processing took: ', duration)
        #print(self.model_output_gdf)
        if False:
            self.model_output_gdf.to_excel(os.path.join(self.save_path, "model_network_gdf.xlsx"), index=False)
            self.game_output_gdf.to_excel(os.path.join(self.save_path, "game_network_gdf.xlsx"), index=False)
        return

    def create_visualizations(self):
        """
        function that sets up and runs the demonstrator visualizations.
        """
        world_bbox = demo_processing.get_bbox(self.model_output_gdf, gdf_type="world")
        game_bbox = demo_processing.get_bbox(self.game_output_gdf, gdf_type="game")
        salinity_range = demo_processing.get_salinity_scale(self.model_output_gdf)
        qapp = QApplication.instance()
        if not qapp:
            qapp = QApplication(sys.argv)
        time_steps = list(sorted(set(self.model_output_gdf["time"])))
        time_index = 0
        starting_scenario = "2017"
        starting_variable = "water_salinity"
        viz_tracker = visualizer.VisualizationTracker(
            starting_scenario=starting_scenario, starting_variable=starting_variable,
            time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range)
        # ,water_level_range = water_level_range, water_velocity_range = water_velocity_range
        colorbar_salinity, colorbar_water_level, colorbar_water_velocity = load_files.load_images()
        gui = visualizer.ApplicationWindow(
            scenarios=self.model_output_gdf, viz_tracker=viz_tracker, bbox=world_bbox,
            salinity_colorbar_image=colorbar_salinity, water_level_colorbar_image=colorbar_water_level,
            water_velocity_image=colorbar_water_velocity)
        side_window = visualizer.GameVisualization(
            scenarios=self.game_output_gdf, viz_tracker=viz_tracker, bbox=game_bbox)
        gui.show()
        side_window.show()
        gui.activateWindow()
        gui.raise_()
        qapp.exec()
        return


def main():
    game = DMG()
    print("initiliazed")
    for turn in range(2,5):
        game.turn = turn
        game.update_forcings()
        if turn == 3:
            game.add_sea_level_rise(slr=1)
        game.update_channel_geometries(change_type="undeepen")
        game.update()
        print("updated to turn ", turn)
    """

    game.update_geometries_test(turn=2)
    print("updated to turn 2")
    game.update_geometries_test(turn=3)
    print("updated to turn 3")
    game.update_geometries_test(turn=4)
    print("updated to turn 4")
    """








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