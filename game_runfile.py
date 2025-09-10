import os
import sys
import time
import geojson
import pandas as pd
import geopandas as gpd
#import numpy as np
from datetime import timedelta
from copy import deepcopy
from PyQt5.QtWidgets import QApplication
from multiprocessing import Process, Manager
from game import game_visualizations as visualizer
from game import load_functions as load_files
from game import model_to_game as game_sync
from game import transform_functions as transform_func
from game import visualizations_input as viz_input
from game import index_channels as indexing
from game import channel_manipulations as update_func
from game import table_functions as table_func
from game import operational_manipulations as operation_func
from game import inlet_functions as inlet_func

#sys.path.insert(1, r'C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 Network model Bouke\version 4.3.4\IMSIDE netw\mod 4.3.4 netw')
from model import runfile_td_v1 as imside_model
from table import table_runfile as game_table

class DMG():
    def __init__(self, mode):
        self.mode = mode
        self._turn = 1
        self._turn_count = 1
        self._scenario = self.mode["scenarios"][0]
        self.debug = self.mode["debug"]
        self.sim_count = self.mode["sim_count"]
        self.export = self.mode["export"]
        self.simulations = ["drought", "normal", "average"]
        operational = {0: ['Qhag', 5, 0, True], 1: ['Qhar', 50, 1, True], 2: ['Qhar_threshold', 1100, 1, True],
                       3: ['Qhij', 2, 0, True], 4: ['Qhij_threshold', 800, 0, True]}
        self.operational_df = pd.DataFrame.from_dict(
            operational, orient='index', columns=['Qtype', 'Qvalue', 'red_markers', 'red_changed'])
        self.weir_tracker = 3  # there are already 2 weirs in the default schematization, next number to add is 3
        self.hexagon_index = None
        self.hexagons_tracker = None
        self.inlets = None
        self.inlet_salinity_tracker = None
        self.forcing_conditions = None
        self.gui = None
        self.load_paths()
        self.load_model()
        # options to send: mirror (-1, 0 or 1), test (bool), save (bool), debug (bool)
        self.table = game_table.Table(self, mirror=1, test=True, save=True)
        new_polygons = True
        self.load_shapes(new_polygons)
        self.load_inlets()
        self.transform_functions(new_polygons)
        self.build_game_network()
        self.index_channels()
        self.run_table(update=False)
        self.turn_split_channels = {}
        self.turn_updates = {}
        self.start_game()

    @property
    def turn(self):
        return self._turn

    @turn.setter
    def turn(self, turn):
        self._turn = turn
        return

    @property
    def turn_count(self):
        return self._turn_count

    @turn_count.setter
    def turn_count(self, count):
        self._turn_count = count
        return

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario
        return

    def start_game(self):
        model_drought_output_df, model_normal_output_df, model_average_output_df = self.run_simulations()
        #print(self.scenario, "drought scenario output:")
        #print(model_drought_output_df[["sb_st", "htot"]].head())
        #print(self.scenario, "normal scenario output:")
        #print(model_normal_output_df[["sb_st", "htot"]].head())
        (merged_model_output_df, columns_to_explode, next_columns_to_explode) = (
            game_sync.merge_model_output(self.simulations, model_drought_output_df,
                                         model_output_df2=model_normal_output_df,
                                         model_output_df3=model_average_output_df))
        self.columns_to_explode = columns_to_explode
        self.next_columns_to_explode = next_columns_to_explode
        self.model_output_to_game(merged_model_output_df, initialize=True)
        self.index_inlets()
        self.update_inlet_salinity()
        if self.export:
            self.export_output()
        self._turn_count = 2
        self.end_round()
        self.create_visualizations()
        return

    def load_paths(self):
        """
        set any core paths that need to be accessed.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_files = os.path.join(dir_path, "game", "input_files")
        self.save_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\09 Prototype refinement summer 2025\coding (notebooks)\hydrodynamics test"
        self.debug_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\09 Prototype refinement summer 2025\debug"
        return

    def load_model(self):
        """
        load the IMSIDE model and extract network.
        """
        self.model = imside_model.IMSIDE(
            scenario=self.scenario, timeseries_type="drought", timeseries_length=self.mode["timeseries"])
        model_network_df = self.model.network
        model_network_df = game_sync.process_model_network(model_network_df)
        model_network_df = game_sync.remove_sea_river_domains(model_network_df)
        self.model_network_gdf = model_network_df
        return

    def load_shapes(self, new_polygons):
        """
        load polygons (world map) and hexagon shapes (board).
        """
        if new_polygons:
            self.world_polygons = load_files.read_json_features(filename='new_RMM_polygons.json',
                                                                path=self.input_files)
        else:
            self.world_polygons = load_files.read_json_features(filename='hexagon_shapes_in_layers_Bouke_network.json',
                                                                path=self.input_files)
        self.game_hexagons = load_files.read_geojson(filename='hexagons_clean.geojson', path=self.input_files)
        return

    def load_inlets(self):
        inlet_tracker = load_files.read_csv(filename='policy_inlet_data.csv', path=self.input_files)
        #inlet_tracker = load_files.read_csv(filename='WSHD_modified_inlet_data.csv', path=self.input_files)
        # this determines which inlets to include or not (only a few to test for now)
        #selected_inlets = ["Inlaat Oostkade", "Inlaatsluis Bernisse", "Inlaat Trekdam",
        #                   "Hevel IJsselmonde - Oostdijk", "Gemaal Delta", "Hevel De Noord - Crezeepolder"]
        #self.inlets = inlet_tracker[inlet_tracker['name'].isin(selected_inlets)]
        self.inlets = inlet_tracker
        return

    def load_debug_files(self):
        filename = "hexagons" + str(self.turn) + "_" + str(self.turn_count) + ".geojson"
        print("loading debug file", filename)
        path_and_file = os.path.join(self.debug_path, filename)
        with open(path_and_file) as f:
            hexagons = geojson.load(f)
        return hexagons

    def index_inlets(self):
        self.inlets = inlet_func.index_inlets_to_model_locations(self.inlets, self.model_output_gdf)
        return

    def update_inlet_salinity(self):
        inlet_salinity_tracker = inlet_func.get_inlet_salinity(
            self.inlets, self.model_output_gdf, turn=self.turn, run=self.turn_count) #, turn_count = self.turn_count):
        inlet_salinity_tracker = inlet_func.get_exceedance_at_inlets(inlet_salinity_tracker)
        if self.inlet_salinity_tracker is None:
            self.inlet_salinity_tracker = inlet_salinity_tracker
        else:
            self.inlet_salinity_tracker = pd.concat([self.inlet_salinity_tracker, inlet_salinity_tracker])
        return

    def run_simulations(self):
        start_time = time.perf_counter()
        if __name__ == '__main__':
            manager = Manager()
            return_dict = manager.dict()
            jobs = []

            for i in range(self.sim_count):
                self.model.change_forcings(scenario=self.scenario, timeseries_type=self.simulations[i])
                self.model.change_local_boundaries(self.operational_df)
                turn_forcing_conditions = self.store_forcings(sim_type=self.simulations[i])
                if self.forcing_conditions is not None:
                    self.forcing_conditions = pd.concat([self.forcing_conditions, turn_forcing_conditions])
                else:
                    self.forcing_conditions = turn_forcing_conditions
                p = Process(target=run_model, args=(self.model, i, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            model_drought_output_df = return_dict[0]
            try:
                model_normal_output_df = return_dict[1]
            except KeyError:
                model_normal_output_df = None
            try:
                model_average_output_df = return_dict[2]
            except KeyError:
                model_average_output_df = None

        duration1 = timedelta(seconds=time.perf_counter() - start_time)
        print('Parallel simulation duration:', duration1)

        """
        start_time = time.perf_counter()
        for i in range(test_range):
            model_drought_output_df = self.run_model()

        duration2 = timedelta(seconds=time.perf_counter() - start_time)
        print('Parallel simulation duration:', duration1)
        print('Serial simulation duration:', duration2)
        """
        return model_drought_output_df, model_normal_output_df, model_average_output_df

    """ currently not used
    #call the function run_model function of the model object and retrieve the output.
    
    def run_model(self):
        self.model.run_model()
        #self.hexagons_tracker.to_excel(os.path.join(self.save_path, "hexagons_tracker.xlsx"), index=True)
        return self.model.output
    """

    def run_table(self, update=True):
        self.table.get_board_state()
        if self.debug:
            hexagons_board = self.load_debug_files()
        else:
            hexagons_board = self.table.hexagons
        self.hexagons_board_gdf = table_func.get_board_gdf(hexagons_board)
        self.hexagons_tracker = table_func.update_hexagon_tracker(self.hexagons_board_gdf, self.hexagons_tracker,
                                                                  update=update)
        return

    """ currently not used
    def get_changes(self, updates):
        updates = "{" + updates + "}"
        try:
            updates = literal_eval(updates)
            print(updates)
        except ValueError:
            print("Error in typed entry, no update applied")
            return
        # TODO update hexagons_tracker instead from the game table object.
        self.hexagons_tracker = update_func.update_polygon_tracker(self.hexagons_tracker, updates)
        self.model.change_local_boundaries(updates)
        return
    """

    def update(self):
        """
        function that handles running the model and retrieving the model output.
        """
        if self.turn_count > 3:
            print("max tries reached for this turn, please press 'End round'")
            return
        self.model.reset_geometry(slr=self.mode["slr"][self.turn - 1])
        self.weir_tracker = 3
        self.run_table()
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

        self.model.update_channel_geometries(new_model_network_df, self.turn_split_channels)

        if self.turn_split_channels:
            self.hexagon_index = indexing.create_polygon_id_tracker(self.model_network_gdf,
                                                                    hexagon_tracker_df=self.hexagons_tracker) #self.split_channels, split_names
            #self.all_split_channels.update(self.turn_split_channels)
        self.operational_df = operation_func.update_operational_rules(self.operational_df, self.hexagons_board_gdf)
        self.model.change_local_boundaries(self.operational_df)
        # TODO: also update hexagon_tracker (references are updated with split channel)?
        #turn_forcing_conditions = self.store_forcings()
        #self.forcing_conditions = pd.concat([self.forcing_conditions, turn_forcing_conditions])

        model_drought_output_df, model_normal_output_df, model_average_output_df = self.run_simulations()
        #print(self.scenario, "drought scenario output:")
        #print(model_drought_output_df[["sb_st", "htot"]].head())
        #print(self.scenario, "normal scenario output:")
        #print(model_normal_output_df[["sb_st", "htot"]].head())
        merged_model_output_df, columns_to_explode, next_columns_to_explode = game_sync.merge_model_output(
            self.simulations, model_drought_output_df, model_output_df2=model_normal_output_df,
            model_output_df3=model_average_output_df)
        self.model_output_to_game(merged_model_output_df)

        self.update_inlet_salinity()
        """
        This is not the most neat way as the time_steps below needs to be updated in the viz_tracker, which the game
        does not have as an object, but the gui object does. So this works and is robust. Neater would be if it can
        be set directly. It is possible to do it dynamically, but it makes more sense to just store it once.
        
        If the ControlWidget can access the game object, then this could work, but this updates it on every
        visualization change:
        
        self.viz_tracker.time_steps = list(
            sorted(set(self.game.model_output_gdf[(self.game.model_output_gdf["turn"] == self.viz_tracker.turn) &
                                                  (self.game.model_output_gdf["run"] == self.viz_tracker.run)][
                           "time"])))
        """
        time_steps = {self.turn: list(sorted(set(self.model_output_gdf[self.model_output_gdf["turn"] == self.turn]["time"])))}
        self.gui.show_turn_button(self.turn, self.turn_count, time_steps)
        if self.export:
            self.export_output()
        print("updated to turn", self.turn, "- run", self.turn_count)
        self.turn_count += 1
        return


    def end_round(self):
        if self.turn_count == 1:
            print("No run was conducted yet, please first run the model!")
            return
        #self.hexagons_tracker["ref_red_markers"] = self.hexagons_tracker["red_markers"]
        #self.hexagons_tracker["ref_blue_markers"] = self.hexagons_tracker["blue_markers"]
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
        self.turn_count = 1

        if self.turn <= len(self.mode["scenarios"]):
            self.scenario = self.mode["scenarios"][self.turn - 1]
        self.table.end_round()
        return

    """
    currently not used
    
    function that handles updating the model forcings to subsequent turns (final scenario forcings to be determined)
    
    def update_forcings(self):
        if self.scenario == self.mode["scenarios"][self.turn-2]:
            print("scenario remains the same, boundary conditions not updated")
            return
        # TODO consider if the add_row approach below is eventually the way to go.
        #self.model.change_forcings(scenario=self.scenario, add_rows=self.weir_tracker-3)
        self.model.change_forcings(scenario=self.scenario)
        slr = self.mode["slr"][self.turn - 1] - self.mode["slr"][self.turn - 2]
        self.add_sea_level_rise(slr=slr)
        return
    """

    def store_forcings(self, sim_type="drought"):
        forcing_conditions = self.model.get_forcings()
        forcing_conditions_df = pd.DataFrame.from_dict(forcing_conditions)
        forcing_conditions_df = forcing_conditions_df.round(0).astype(int)
        forcing_conditions_df["Sea Level Rise"] = self.mode["slr"][self.turn-1]
        forcing_conditions_df["turn"] = self.turn
        forcing_conditions_df["run"] = self.turn_count
        forcing_conditions_df["type"] = sim_type
        return forcing_conditions_df

    """
    currently not used
    
    def add_sea_level_rise(self, slr=0):
        if slr != 0:
            try:
                self.model_network_gdf["ref_Hn"] = self.model_network_gdf["ref_Hn"].apply(
                    lambda x: np.array([y + slr for y in x]))
            except KeyError:
                pass
            try:
                self.model_network_gdf["Hn"] = self.model_network_gdf["Hn"].apply(
                    lambda x: np.array([y + slr for y in x]))
            except KeyError:
                pass
            #self.model.add_sea_level_rise(slr)
            print("added", slr, "meters sea level rise in model")
        else:
            print("no sea level rise (in meters) provided or set to 0, no sea level rise added in model")
        return
    """


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

    def transform_functions(self, new_polygons):
        """
        create transform functions and transform polygons to correct (world) coordinates.
        """
        self.transform_calibration = transform_func.create_calibration_file(self.world_polygons, new_polygons)
        self.world_polygons = transform_func.transform(self.world_polygons, self.transform_calibration,
                                                       export="warped", path="")
        self.game_hexagons = game_sync.find_neighbours(self.game_hexagons)
        self.world_polygons = game_sync.match_hexagon_properties(self.world_polygons, self.game_hexagons,
                                                                 "neighbours")
        if False:
            polygons_gdf = gpd.GeoDataFrame.from_features(self.world_polygons.features)
            filename = "polygons_gdf.xlsx"
            polygons_gdf.to_excel(os.path.join(self.save_path, filename), index=True)

            hexagons_gdf = gpd.GeoDataFrame.from_features(self.game_hexagons.features)
            filename = "hexagons_gdf.xlsx"
            hexagons_gdf.to_excel(os.path.join(self.save_path, filename), index=True)
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
        return

    def model_output_to_game(self, model_output_df, initialize=False):
        """
        function to process the model output to the model and game output geodataframes respectively.
        """
        start_time = time.perf_counter()
        double_exploded_output_df, exploded_output_df = game_sync.process_model_output(
            model_output_df, self.columns_to_explode, self.next_columns_to_explode, self.sim_count, self.simulations,
            scenario=self.mode["scenarios"][self.turn-1])

        if initialize == True:
            model_output_gdf = game_sync.output_df_to_gdf(double_exploded_output_df)
            model_output_gdf = game_sync.add_polygon_ids(model_output_gdf, self.world_polygons)

            # NOTE: first point in Lek is double ? check source
            # NOTE 2: first and last points all match earlier runs, but there are less points in Lek ? check source
            game_output_gdf = game_sync.model_output_to_game_locations(self.game_network_gdf,
                                                                       model_output_gdf, exploded_output_df)
            model_output_gdf = model_output_gdf.reset_index()
            model_output_gdf = model_output_gdf.drop(columns="index")
            timestep_0 = model_output_gdf.iloc[0]["time"]

            possible_columns_to_drop = ['time', 'sb_st_drought', 'water_salinity_drought', 'sb_st_normal',
                                        'water_salinity_normal', 'sb_st_average', 'water_salinity_average',
                                        'htot_drought', 'htot_normal', 'htot_average']
            columns_to_drop = []
            for column in possible_columns_to_drop:
                if column in model_output_gdf.columns.values:
                    columns_to_drop.append(column)

            model_output_ref_gdf = model_output_gdf.copy()
            model_output_ref_gdf = model_output_ref_gdf.loc[model_output_ref_gdf['time'] == timestep_0]
            model_output_ref_gdf = model_output_ref_gdf.drop(columns=columns_to_drop)

            game_output_ref_gdf = game_output_gdf.copy()
            game_output_ref_gdf = game_output_ref_gdf.loc[game_output_ref_gdf['time'] == timestep_0]
            game_output_ref_gdf = game_output_ref_gdf.drop(columns=columns_to_drop)

            model_output_gdf = game_sync.output_to_timeseries(
                model_output_gdf, self.sim_count, self.simulations, turn=self.turn, turn_count=self.turn_count)
            game_output_gdf = game_sync.output_to_timeseries(
                game_output_gdf, self.sim_count, self.simulations, turn=self.turn, turn_count=self.turn_count)
            self.model_output_gdf = model_output_gdf
            self.model_output_ref_gdf = model_output_ref_gdf
            self.game_output_gdf = game_output_gdf
            self.game_output_ref_gdf = game_output_ref_gdf

            duration = timedelta(seconds=time.perf_counter() - start_time)
            print('Initial output processing took:', duration)
        else:
            possible_columns_to_merge = ['id', 'branch_rank', 'time', 'sb_st_drought', 'water_salinity_drought',
                                         'sb_st_normal', 'water_salinity_normal', 'sb_st_average',
                                         'water_salinity_average', 'htot_drought', 'htot_normal', 'htot_average']
            columns_to_merge = []
            for column in possible_columns_to_merge:
                if column in double_exploded_output_df.columns.values:
                    columns_to_merge.append(column)
            output_to_merge_df = double_exploded_output_df[columns_to_merge]

            if self.turn_split_channels:
                output_to_merge_df = game_sync.update_split_channel_ids(output_to_merge_df, self.turn_split_channels)
            output_to_merge_df = output_to_merge_df.drop(columns=["branch_rank"])

            model_output_gdf = self.model_output_ref_gdf.merge(output_to_merge_df, on="id")
            game_output_gdf = self.game_output_ref_gdf.merge(output_to_merge_df, on="id")
            model_output_gdf = game_sync.output_to_timeseries(
                model_output_gdf, self.sim_count, self.simulations, turn=self.turn, turn_count=self.turn_count)
            game_output_gdf = game_sync.output_to_timeseries(
                game_output_gdf, self.sim_count, self.simulations, turn=self.turn, turn_count=self.turn_count)
            self.model_output_gdf = pd.concat([self.model_output_gdf, model_output_gdf])
            self.game_output_gdf = pd.concat([self.game_output_gdf, game_output_gdf])

            duration = timedelta(seconds=time.perf_counter() - start_time)
            print('Update output processing took:', duration)
        return

    def export_output(self): #, type="drought"):
        """
        The part before saving is only to make it easier to convert the output back into numpy arrays in a jupyter
        notebook, as the output file has to be read in raw binary mode (making all values in the DataFrame strings).
        This way, all numpy arrays are saved as a list with the "," separator.
        """
        filename = "model_output_gdf%s_%d.xlsx" % (self.turn, self.turn_count)
        self.model_output_gdf.to_excel(os.path.join(self.save_path, filename), index=True)
        filename = "game_output_gdf%s_%d.xlsx" % (self.turn, self.turn_count)
        self.game_output_gdf.to_excel(os.path.join(self.save_path, filename), index=True)
        return

    def debug_output(self):
        model_network_df = self.model.network
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
        world_bbox = viz_input.get_bbox(self.model_output_gdf, gdf_type="world")
        game_bbox = viz_input.get_bbox(self.game_output_gdf, gdf_type="game")
        salinity_range = viz_input.get_salinity_scale(self.model_output_gdf)
        salinity_category = viz_input.get_salinity_scale(self.model_output_gdf, column="salinity_category_drought")
        qapp = QApplication.instance()
        if not qapp:
            qapp = QApplication(sys.argv)
        time_steps = {1: list(sorted(set(self.model_output_gdf["time"])))}
        time_index = 0
        starting_variable = "salinity_category"
        viz_tracker = visualizer.VisualizationTracker(
            starting_turn=self.turn-1, scenarios=self.mode["scenarios"], starting_variable=starting_variable,
            time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range,
            salinity_category=salinity_category, inlet_to_plot="Inlaatsluis Bernisse")
        colorbar_salinity, labels_salinity_categories = load_files.load_images()
        self.gui = visualizer.ApplicationWindow(
            game=self, viz_tracker=viz_tracker, bbox=world_bbox,
            salinity_colorbar_image=colorbar_salinity, salinity_category_image=labels_salinity_categories)
        side_window = visualizer.GameVisualization(game=self, viz_tracker=viz_tracker, bbox=game_bbox)
        self.gui.show()
        side_window.show()
        self.gui.activateWindow()
        self.gui.raise_()
        print("game initialized")
        qapp.exec()
        return


def run_model(model, procnumber, return_dict):
    """
    call the function run_model function of the model object and retrieve the output.
    """
    model.run_model()
    return_dict[procnumber] = model.output
    return

def main(mode):
    game = DMG(mode)
    print("initialized")
    return


scenario_settings1 = {"scenarios": ["reference", "reference", "reference", "reference"], "slr": [0, 0, 0, 0],
                      "timeseries": "month", "sim_count": 2, "debug": False, "export": False}

scenario_settings2 = {"scenarios": ["reference", "2050Md", "2100Md", "2150Md"], "slr": [0, 0.25, 0.59, 1.41],
                      "timeseries": "dummy", "sim_count": 2, "debug": True, "export": False}

scenario_settings3 = {"scenarios": ["reference", "2050Hd", "2100Hd", "2150Hd"], "slr": [0, 0.27, 0.82, 2],
                      "timeseries": "month", "sim_count": 2, "debug": False, "export": False}

scenario_settings4 = {"scenarios": ["reference", "2050Hd", "2100Hd"], "slr": [0, 0.27, 0.82],
                      "timeseries": "dummy", "sim_count": 2, "debug": True, "export": True}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(scenario_settings4)