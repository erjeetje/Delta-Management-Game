import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PyQt5.QtWidgets import QApplication
from ast import literal_eval
from game import demo_visualizations as visualizer
from game import load_functions as load_files
from game import model_to_game as game_sync
from game import transform_functions as transform_func
from game import demo_processing
from game import index_channels as indexing
from game import channel_manipulations as update_func

#sys.path.insert(1, r'C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 Network model Bouke\version 4.3.4\IMSIDE netw\mod 4.3.4 netw')
from model import runfile_td_v1 as imside_model

class DMG():
    def __init__(self, mode):
        self.mode = mode
        self._turn = 1
        self._scenario = self.mode["scenarios"][0]
        self.weir_tracker = 3  # there are already 2 weirs in the default schematization, next number to add is 3
        self.hexagon_index = None
        self.hexagons_tracker = None
        self.load_paths()
        self.load_model()
        self.load_shapes()
        self.transform_functions()
        self.build_game_network()
        self.index_channels()
        self.turn_split_channels = {}
        self.all_split_channels = {}
        self.turn_updates = {}
        model_output_df = self.run_model()
        self.model_output_to_game(model_output_df, initialize=True)
        self.end_round()
        self.create_visualizations()
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

    def load_paths(self):
        """
        set any core paths that need to be accessed.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_files = os.path.join(dir_path, "game", "input_files")
        self.save_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 DMG design\coding (notebooks)\game to IMSIDE"
        return

    def load_model(self):
        """
        load the IMSIDE model and extract network.
        """
        self.model = imside_model.IMSIDE(scenario=self.scenario)
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

    def run_model(self, scenario="2017"):
        """
        call the function run_model function of the model object and retrieve the output.
        """
        self.model.run_model()
        return self.model.output

    def get_changes(self, updates):
        updates = "{" + updates + "}"
        try:
            updates = literal_eval(updates)
            print(updates)
        except ValueError:
            print("Error in typed entry, no update applied")
            return
        self.hexagons_tracker = update_func.update_polygon_tracker(self.hexagons_tracker, updates)
        self.model.change_local_boundaries(updates)
        return

    def update(self):
        """
        function that handles running the model and retrieving the model output.
        """
        slr = self.mode["slr"][self.turn-1] - self.mode["slr"][self.turn-2]
        print(slr)
        self.add_sea_level_rise(slr=slr)
        turn_change = self.hexagons_tracker.loc[self.hexagons_tracker['changed'] == True]
        turn_change = update_func.to_change(turn_change)
        new_model_network_df = self.model_network_gdf.copy()
        new_model_network_df = update_func.geometry_to_update(turn_change, new_model_network_df)
        # TODO: consider how to cut channels into segments, as otherwise width changes also affect unchanged polygons
        new_model_network_df = update_func.update_channel_length(new_model_network_df)

        new_model_network_df = update_func.update_channel_references(new_model_network_df)
        new_model_network_df = update_func.update_channel_geometry(new_model_network_df)
        new_model_network_df, self.turn_split_channels, split_names = update_func.apply_split(new_model_network_df, self.weir_tracker)
        self.weir_tracker = self.weir_tracker + (len(self.turn_split_channels) * 2)

        # TODO: add a check function if segments can be "knitted" back together (basically, ensure lowest # of segments)
        self.model.update_channel_geometries(new_model_network_df, self.turn_split_channels)
        self.model_network_gdf = new_model_network_df
        if self.turn_split_channels:
            self.hexagon_index = indexing.create_polygon_id_tracker(self.model_network_gdf,
                                                                    hexagon_tracker_df=self.hexagons_tracker) #self.split_channels, split_names
            self.all_split_channels.update(self.turn_split_channels)
        # TODO: also update hexagon_tracker (references are updated with split channel)
        #if self.turn == 2:
        #    self.split_channel(channel="Nieuwe Maas 1 old")
        model_output_df = self.run_model()
        # to test if this overrides values or not, otherwise adjust code in the function below to remove any values
        # from the same scenario of this exist (for logging purposes, perhaps do store those somewhere).
        #self.model_output_to_game(model_output_df, scenario=self.scenario)
        self.model_output_to_game(model_output_df)
        print("updated to turn", self.turn)
        self.end_round()
        return


    def end_round(self):
        self.hexagons_tracker["ref_red_marker"] = self.hexagons_tracker["red_marker"]
        self.hexagons_tracker["ref_blue_marker"] = self.hexagons_tracker["blue_marker"]
        try:
            self.model_network_gdf["ref_L"] = self.model_network_gdf["L"]
        except KeyError:
            pass
        try:
            self.model_network_gdf["ref_b"] = self.model_network_gdf["b"]
        except KeyError:
            pass
        try:
            self.model_network_gdf["ref_Hn"] = self.model_network_gdf["Hn"]
        except KeyError:
            pass
        try:
            self.model_network_gdf["ref_dx"] = self.model_network_gdf["dx"]
        except KeyError:
            pass
        self.turn += 1
        if self.turn < 4:
            self.update_forcings()
        return

    def update_forcings(self):
        """
        function that handles updating the model forcings to subsequent turns (final scenario forcings to be determined)
        """
        self.scenario = self.mode["scenarios"][self.turn-1]
        if self.scenario == self.mode["scenarios"][self.turn-2]:
            print("scenario remains the same, boundary conditions not updated")
            return
        # TODO consider if the add_row approach below is eventually the way to go.
        self.model.change_forcings(scenario=self.scenario, add_rows=self.weir_tracker-3)
        return

    def add_sea_level_rise(self, slr=0):
        if slr != 0:
            self.model_network_gdf["ref_Hn"] = self.model_network_gdf["ref_Hn"].apply(
                lambda x: np.array([y + slr for y in x]))
            self.model_network_gdf["Hn"] = self.model_network_gdf["Hn"].apply(
                lambda x: np.array([y + slr for y in x]))
            self.model.add_sea_level_rise(slr)
            print("added", slr, "meters sea level rise in model")
        else:
            print("no sea level rise (in meters) provided or set to 0, no sea level rise added in model")
        return


    def update_channel_geometries(self, channels_to_update=[], change_type="deepen"):
        """
        function that handles any players actions to update the correct geometry in the model. Currently hard-coded,
        will be updated with the live board link.
        """
        if not channels_to_update:
            print("no channels provided to change, no change to any geometry")
        else:
            for channel in channels_to_update:
                self.model.change_channel_geometry(channel, change_type=change_type)
        return

    def split_channel(self, channel):
        """

        TODO: track which channels are split, so that a channel can also be "unsplit"
        """
        if isinstance(channel, str):
            self.split_channels[channel] = [channel + "_1", channel + "_2"]
            self.model.split_channel(channel_to_split=channel, location=0.5, next_weir_number=self.weir_tracker)
            self.weir_tracker += 2 #functions adds two weirs, one to each channel
        else:
            print("no channel key provided (should be string), channel not split")
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
        #self.model_network_gdf = game_sync.branches_to_segment(self.model_network_gdf)
        #self.model.add_segments_to_channels(self.model_network_gdf)
        self.game_network_gdf = game_sync.draw_branch_network(self.game_hexagons, self.model_network_gdf)
        return

    def index_channels(self):
        self.model_network_gdf = indexing.index_polygons_to_channel_geometry(self.model_network_gdf)
        self.hexagons_tracker = indexing.create_polygon_id_tracker(self.model_network_gdf)
        print(self.hexagons_tracker)
        return

    def model_output_to_game(self, model_output_df, initialize=False):
        """
        function to process the model output to the model and game output geodataframes respectively.
        """
        #filename = "model_output_gdf_" + str(self.turn) + ".xlsx"
        #model_output_df.to_excel(os.path.join(self.save_path, filename), index=True)
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
            self.model_output_ref_gdf = self.model_output_ref_gdf.drop(columns=["time", "sb_st", 'sb_mgl'])
            self.game_output_ref_gdf = self.game_output_gdf.loc[self.game_output_gdf['time'] == timestep_0]
            self.game_output_ref_gdf = self.game_output_ref_gdf.drop(columns=["time", "sb_st", 'sb_mgl'])
            self.model_output_gdf = game_sync.output_to_timeseries(self.model_output_gdf, turn=self.turn)
            self.game_output_gdf = game_sync.output_to_timeseries(self.game_output_gdf, turn=self.turn)
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
            output_to_merge_df = double_exploded_output_df[["id", "branch_rank", "time", "sb_st", 'sb_mgl']]
            if self.all_split_channels:
                output_to_merge_df = game_sync.update_split_channel_ids(output_to_merge_df, self.all_split_channels)
            output_to_merge_df = output_to_merge_df.drop(columns=["branch_rank"])
            model_output_gdf = self.model_output_ref_gdf.merge(output_to_merge_df, on="id")
            game_output_gdf = self.game_output_ref_gdf.merge(output_to_merge_df, on="id")
            model_output_gdf = game_sync.output_to_timeseries(model_output_gdf, turn=self.turn)
            game_output_gdf = game_sync.output_to_timeseries(game_output_gdf, turn=self.turn)
            self.model_output_gdf = pd.concat([self.model_output_gdf, model_output_gdf])
            self.game_output_gdf = pd.concat([self.game_output_gdf, game_output_gdf])
            duration = timedelta(seconds=time.perf_counter() - start_time)
            print('Update output processing took: ', duration)
        return

    def export_output(self):
        """
        The part before saving is only to make it easier to convert the output back into numpy arrays in a jupyter
        notebook, as the output file has to be read in raw binary mode (making all values in the DataFrame strings).
        This way, all numpy arrays are saved as a list with the "," separator.
        """
        df_copy = self.model_network_gdf.copy()
        for column in ["Hn", "L", "b", "dx", "plot x", "plot y"]:
            df_copy[column] = df_copy.apply(lambda row: row[column].tolist(), axis=1)
        df_copy.to_excel(os.path.join(self.save_path, "model_network_gdf.xlsx"), index=True)
        self.game_network_gdf.to_excel(os.path.join(self.save_path, "game_network_gdf.xlsx"), index=True)
        self.model_output_gdf.to_excel(os.path.join(self.save_path, "model_output_gdf.xlsx"), index=True)
        self.game_output_gdf.to_excel(os.path.join(self.save_path, "game_output_gdf.xlsx"), index=True)

    def temp_output(self):
        model_network_df = self.model.network
        print(model_network_df.head())
        model_network_gdf = game_sync.process_model_network(model_network_df)
        self.world_polygons, model_network_gdf = game_sync.find_branch_intersections(deepcopy(self.world_polygons),
                                                                                     model_network_gdf)
        model_network_gdf = game_sync.determine_polygon_intersections(model_network_gdf, self.world_polygons)
        #self.index_channels()
        for column in ["Hn", "L", "b", "dx", "plot x", "plot y"]:
            model_network_gdf[column] = model_network_gdf.apply(lambda row: row[column].tolist(), axis=1)
        model_network_gdf.to_excel(os.path.join(self.save_path, "model_network_split_channel_gdf.xlsx"), index=True)
        return

    def create_visualizations(self):
        """
        function that sets up and runs the demonstrator visualizations.
        """
        world_bbox = demo_processing.get_bbox(self.model_output_gdf, gdf_type="world")
        game_bbox = demo_processing.get_bbox(self.game_output_gdf, gdf_type="game")
        salinity_range = demo_processing.get_salinity_scale(self.model_output_gdf)
        salinity_category = demo_processing.get_salinity_scale(self.model_output_gdf, column="salinity_category")
        qapp = QApplication.instance()
        if not qapp:
            qapp = QApplication(sys.argv)
        time_steps = list(sorted(set(self.model_output_gdf["time"])))
        time_index = 0
        starting_variable = "water_salinity"
        viz_tracker = visualizer.VisualizationTracker(
            starting_turn=self.turn-1, scenarios=self.mode["scenarios"], starting_variable=starting_variable,
            time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range,
            salinity_category=salinity_category)
        # ,water_level_range = water_level_range, water_velocity_range = water_velocity_range
        colorbar_salinity, labels_salinity_categories = load_files.load_images()
        gui = visualizer.ApplicationWindow(
            game=self, viz_tracker=viz_tracker, bbox=world_bbox,
            salinity_colorbar_image=colorbar_salinity, salinity_category_image=labels_salinity_categories)
        side_window = visualizer.GameVisualization(game=self, viz_tracker=viz_tracker, bbox=game_bbox)
        gui.show()
        side_window.show()
        gui.activateWindow()
        gui.raise_()
        print("game initialized")
        qapp.exec()
        return


def main(mode):
    game = DMG(mode)
    print("initialized")
    return


scenario_settings1 = {"scenarios": ["2018", "2018", "2018", "2018"], "slr": [0, 0, 0, 0]}

scenario_settings2 = {"scenarios": ["2017", "2018", "2100le", "2100he"], "slr": [0, 0, 1, 1]}

scenario_settings3 = {"scenarios": ["2018", "2050he", "2100he", "2150he"], "slr": [0, 0.5, 1, 1.5]}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(scenario_settings2)