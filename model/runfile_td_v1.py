# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, December 2023
# =============================================================================
# import functions
# =============================================================================
import pandas as pd
from model import settings_td_v1
from model import core_td_v1
#from inputfile_v1 import input_network
#from functions_all_v1 import network_funcs
import numpy as np
from math import sqrt
from copy import deepcopy
from shapely import get_coordinates, line_interpolate_point
from shapely.geometry import LineString


class IMSIDE():
    def __init__(self, scenario, timeseries_length=None):
        self.timeseries_length = timeseries_length
        self._current_forcing = settings_td_v1.set_forcing(scenario=scenario, timeseries_length=self.timeseries_length)
        self.delta = core_td_v1.mod42_netw(settings_td_v1.constants, settings_td_v1.geo_pars, self._current_forcing,
                                           settings_td_v1.phys_pars)#, pars_seadom = (25000,100,10), pars_rivdom = (200000,2000,0))
        network_df = pd.DataFrame.from_dict(self.delta.ch_gegs)
        network_df = network_df.T
        self._network = network_df
        self._output = None
        self.Qhar = 0
        self.Qhar_threshold = 0
        self.Qhag = 0
        self.Qhij = 0
        self.Qhij_threshold = 0
        self.ref_Qwaal = deepcopy(self.delta.Qriv[0])
        self.ref_Qhij = deepcopy(self.delta.Qweir[0])
        self.ref_Qlek = deepcopy(self.delta.Qweir[1])
        self.ref_Qhar = deepcopy(self.delta.Qhar[0])
        return

    @property
    def output(self):
        return self._output

    @property
    def network(self):
        return self._network

    @property
    def current_forcing(self):
        return self._current_forcing

    """
    @network.setter
    def network(self, network):
        self._network = network
        return
    """

    def run_model(self):
        # calculate river discharge distribution
        self.delta.run_model()
        self.delta.calc_output()
        output_df = pd.DataFrame.from_dict(self.delta.ch_outp)
        self._output = output_df.T
        return

    def get_forcings(self):
        Waal = deepcopy(self.delta.Qriv[0])
        Meuse = deepcopy(self.delta.Qriv[1])
        Hol_IJssel = deepcopy(self.delta.Qweir[0])
        Lek = deepcopy(self.delta.Qweir[1])
        Haringvliet = deepcopy(self.delta.Qhar[0])
        return {"Waal": Waal, "Meuse": Meuse, "Hol_IJssel": Hol_IJssel, "Lek": Lek, "Haringvliet": Haringvliet}

    def change_forcings(self, scenario, add_rows=0):
        try:
             self._current_forcing = settings_td_v1.set_forcing(scenario=scenario, timeseries_length=self.timeseries_length)
             [self.delta.Qriv, self.delta.Qweir, self.delta.Qhar, self.delta.n_sea, self.delta.soc,
              self.delta.sri, self.delta.swe, self.delta.tid_per, self.delta.a_tide, self.delta.p_tide,
              self.delta.T, self.delta.DT] = self._current_forcing
        except TypeError:
            return
        if add_rows != 0:
            for i in range(add_rows):
                self.delta.Qweir = np.vstack([self.delta.Qweir, np.zeros(len(self.delta.Qweir[0]))])
                self.delta.swe = np.vstack([self.delta.swe, np.array([0.15 + np.zeros(len(self.delta.swe[0]))])])
        self.ref_Qwaal = deepcopy(self.delta.Qriv[0])
        self.ref_Qhij = deepcopy(self.delta.Qweir[0])
        self.ref_Qlek = deepcopy(self.delta.Qweir[1])
        self.ref_Qhar = deepcopy(self.delta.Qhar[0])
        return

    def change_local_boundaries(self, operational_updates_df):
        updates = operational_updates_df.copy()
        updates = updates.set_index("Qtype")
        print(4)
        """
        Function sets local boundary conditions.

        TODO: In the version connected to the table, this needs to be run at every update, hence the "resetting" below.
        """
        self.delta.Qriv[0] = self.ref_Qwaal
        self.delta.Qweir[0] = self.ref_Qhij
        self.delta.Qweir[1] = self.ref_Qlek
        self.delta.Qhar[0] = self.ref_Qhar
        for key, row in updates.iterrows():
            if key == "Qhar":
                self.Qhar = row["Qvalue"]
            elif key == "Qhar_threshold":
                self.Qhar_threshold = row["Qvalue"]
            elif key == "Qhag":
                self.Qhag = row["Qvalue"]
            elif key == "Qhij":
                self.Qhij = row["Qvalue"]
            elif key == "Qhij_threshold":
                self.Qhij_threshold = row["Qvalue"]
            if row["red_changed"]:
                print("updated", key, "to", row["Qvalue"])
        print(5)

        # update operations
        print("Qwaal:", self.delta.Qriv[0])
        print("Qhar:", self.delta.Qhar[0])
        for i, Q in enumerate(self.delta.Qriv[0]):
            if Q / 0.75 <= self.Qhar_threshold:  # Waal takes approximately 75% of Lobith discharge (low flow)
                self.delta.Qhar[0][i] = 0
            elif self.delta.Qhar[0][i] <= self.Qhar:
                self.delta.Qhar[0][i] = self.Qhar
        print("Set Qhar to", self.delta.Qhar)

        print("ref_Qlek", self.delta.Qweir[1])
        print("ref_Qwaal", self.delta.Qriv[0])
        for i, Q in enumerate(self.delta.Qweir[1]):
            if Q < self.Qhag:
                self.delta.Qweir[1][i] = self.Qhag
                self.delta.Qriv[0][i] -= (self.Qhag - Q)
        print("new_Qlek", self.delta.Qweir[1])
        print("new_Qwaal", self.delta.Qriv[0])

        print("ref_Qhij", self.delta.Qweir[0])
        print("ref_Qwaal", self.delta.Qriv[0])
        for i, Q in enumerate(self.delta.Qriv[0]):
            if Q / 0.75 <= self.Qhij_threshold:  # Waal takes approximately 75% of Lobith discharge (low flow)
                Qold = self.delta.Qweir[0][i]
                self.delta.Qweir[0][i] = self.Qhij
                self.delta.Qriv[0][i] -= (self.Qhij - Qold)
        print("new_Qhij", self.delta.Qweir[0])
        print("new_Qwaal", self.delta.Qriv[0])
        return


    def change_local_boundaries_old(self, values):
        """
        Function sets local boundary conditions.

        TODO: In the version connected to the table, this needs to be run at every update, hence the "resetting" below.
        """
        self.delta.Qriv[0] = self.ref_Qwaal
        self.delta.Qweir[0] = self.ref_Qhij
        self.delta.Qweir[1] = self.ref_Qlek
        self.delta.Qhar[0] = self.ref_Qhar
        for key, value in values.items():
            if not isinstance(key, str):
                print("no boundary input", key)
                continue
            else:
                Qloc = key
                if isinstance(value, int):
                    print("int")
                    Qout = value
                elif isinstance(value, list):
                    print("list")
                    Qthreshold = value[0]
                    Qout = value[1]
                if Qloc == "Qhar":
                    print("Qwaal:", self.delta.Qriv[0])
                    print("Qhar:", self.delta.Qhar[0])
                    for i, Q in enumerate(self.delta.Qriv[0]):
                        if Q / 0.75 <= Qthreshold: # Waal takes approximately 75% of Lobith discharge (low flow)
                            self.delta.Qhar[0][i] = 0
                        elif self.delta.Qhar[0][i] <= Qout:
                            self.delta.Qhar[0][i] = Qout
                    print("Set Qhar to", self.delta.Qhar)
                elif Qloc == "Qlek":
                    print("ref_Qlek", self.delta.Qweir[1])
                    print("ref_Qwaal", self.delta.Qriv[0])
                    for i, Q in enumerate(self.delta.Qweir[1]):
                        if Q < Qout:
                            self.delta.Qweir[1][i] = Qout
                            self.delta.Qriv[0][i] -= (Qout - Q)
                    print("new_Qlek", self.delta.Qweir[1])
                    print("new_Qwaal", self.delta.Qriv[0])
                elif Qloc == "Qhij":
                    print("ref_Qhij", self.delta.Qweir[0])
                    print("ref_Qwaal", self.delta.Qriv[0])
                    for i, Q in enumerate(self.delta.Qriv[0]):
                        if Q / 0.75 <= Qthreshold: # Waal takes approximately 75% of Lobith discharge (low flow)
                            Qold = self.delta.Qweir[0][i]
                            self.delta.Qweir[0][i] = Qout
                            self.delta.Qriv[0][i] -= (Qout - Qold)
                    print("new_Qhij", self.delta.Qweir[0])
                    print("new_Qwaal", self.delta.Qriv[0])
                else:
                    print("unknown type given, no local boundary is updated.")
        return

    def change_local_boundaries_old(self, type, value):
        if type == "Qhar":
            self.delta.Qhar[0] = np.array([value + np.zeros(len(self.delta.Qhar[0]))])
            print("Set Qhar to", self.delta.Qhar)
        elif type == "Qlek":
            ref_Qlek = self.delta.Qweir[1]
            print("ref_Qlek", ref_Qlek)
            print("ref_Qwaal", self.delta.Qriv[0])
            dif_Qlek = np.array([value - x for x in ref_Qlek])
            print("dif_Qlek", dif_Qlek)
            self.delta.Qweir[1] = np.array([value + np.zeros(len(self.delta.Qweir[1]))])
            print("new_Qlek", self.delta.Qweir[1])
            self.delta.Qriv[0] = np.subtract(self.delta.Qriv[0], dif_Qlek)
            print("new_Qwaal", self.delta.Qriv[0])
        elif type == "Qhij":
            ref_Qhij = self.delta.Qweir[0]
            print("ref_Qhij", ref_Qhij)
            print("ref_Qwaal", self.delta.Qriv[0])
            dif_Qhij = np.array([value - x for x in ref_Qhij])
            print("dif_Qhij", dif_Qhij)
            self.delta.Qweir[0] = np.array([value + np.zeros(len(self.delta.Qweir[0]))])
            print("new_Qlek", self.delta.Qweir[0])
            self.delta.Qriv[0] = np.subtract(self.delta.Qriv[0], dif_Qhij)
            print("new_Qwaal", self.delta.Qriv[0])
        else:
            print("unknown type given, no local boundary is updated.")
        return

    """
    def add_segments_to_channels(self, model_network_gdf):
        model_network_gdf = model_network_gdf.set_index("Name")
        for index, row in model_network_gdf.iterrows():
            for key in ["Hn", "L", "b", "dx"]:
                self.delta.ch_gegs[index][key] = row[key]
            self.delta.add_properties(index, new_channel=False)
        self.delta.run_checks()
        return
    """

    def update_channel_geometries(self, model_network_gdf, channels_to_split):
        model_network_change_gdf = model_network_gdf.loc[model_network_gdf['changed'] == True]
        model_network_change_gdf = model_network_change_gdf.reset_index()
        model_network_change_gdf = model_network_change_gdf.set_index("Name")
        for old_channel, new_channels in channels_to_split.items():
            self.delta.ch_gegs.pop(old_channel)
            self.delta.ch_pars.pop(old_channel)
            self.delta.ch_outp.pop(old_channel)
            self.delta.ch_tide.pop(old_channel)
            self.delta.ch_keys.remove(old_channel)
        for index, row in model_network_change_gdf.iterrows():
            new_channel = False
            if index in self.delta.ch_gegs:
                print("old", index, "geometry:",
                      self.delta.ch_gegs[index]["Hn"], "(Hn)",
                      self.delta.ch_gegs[index]["L"], "(L)",
                      self.delta.ch_gegs[index]["b"], "(b)",
                      self.delta.ch_gegs[index]["dx"], "(dx)")
            else:
                self.delta.ch_gegs[index] = {}
                new_channel = True
                print("channel not yet in the old, possibly split?")
            for key in ["Hn", "L", "b", "dx"]:
                self.delta.ch_gegs[index][key] = row[key]
            if new_channel:
                self.delta.ch_keys.append(index)
                self.delta.ch_gegs[index]["Name"] = index
                for key in ["loc x=0", "loc x=-L", "plot x", "plot y", 'plot color']:
                    self.delta.ch_gegs[index][key] = row[key]
                self.delta.Qweir = np.vstack([self.delta.Qweir, np.zeros(len(self.delta.Qweir[0]))])
                self.delta.swe = np.vstack([self.delta.swe, np.array([0.15 + np.zeros(len(self.delta.swe[0]))])])
            self.delta.add_properties(index, new_channel=new_channel)
            print("new", index, "geometry:",
                  self.delta.ch_gegs[index]["Hn"], "(Hn)",
                  self.delta.ch_gegs[index]["L"], "(L)",
                  self.delta.ch_gegs[index]["b"], "(b)",
                  self.delta.ch_gegs[index]["dx"], "(dx)")
            if new_channel:
                print("added",
                      self.delta.ch_gegs[index]["loc x=0"], "(loc x=0)",
                      self.delta.ch_gegs[index]["loc x=-L"], "(loc x=-L)",
                      self.delta.ch_gegs[index]["Name"], "(Name)")
        self.delta.run_checks()
        network_df = pd.DataFrame.from_dict(self.delta.ch_gegs)
        network_df = network_df.T
        self._network = network_df
        return

    def update_channel_splits(self, model_network_gdf, channels_to_split):

        self.delta.run_checks()
        network_df = pd.DataFrame.from_dict(self.delta.ch_gegs)
        network_df = network_df.T
        self._network = network_df
        return

    def update_channels_geometry(self, model_network_gdf):
        channels_to_change = model_network_gdf.loc[model_network_gdf['changed'] == True]
        channels_to_change = channels_to_change.set_index("Name")
        for index, row in channels_to_change.iterrows():
            for key in ["Hn", "L", "b", "dx"]:
                self.delta.ch_gegs[index][key] = row[key]
            self.delta.add_properties(index, new_channel=True)
        self.delta.run_checks()
        network_df = pd.DataFrame.from_dict(self.delta.ch_gegs)
        network_df = network_df.T
        self._network = network_df
        return

    def add_sea_level_rise(self, slr):
        """
        Adds SLR by increasing the depth of channels. The fraction of SLR on water level increase is close to 1.0 for
        median/low river discharges for the entire playable area (see "Analyse van bouwstenen en adaptatiepaden voor
        aanpassen aan zeespiegelstijging in Nederland", Figure 14 on p. 61).

        TODO: as the schematisation goes further in-land, may need to add some ratios to Waal? (Lek is not to impacted
        as it ends at Hagestein). Not a priority
        """
        for channel in self.delta.ch_keys:
            self.delta.ch_gegs[channel]["Hn"] += slr
        return

    def change_channel_geometry(self, channel_to_change, segments_to_update=[0], change_type="widen"):
        if change_type == "widen":
            key = "b"
            ratio = 1.2
        elif change_type == "narrow":
            key = "b"
            ratio = 0.8
        elif change_type == "deepen":
            key = "Hn"
            ratio = 1.2
        elif change_type == "undeepen":
            key = "Hn"
            ratio = 0.8
        else:
            print("Invalid change type given, no changes applied")
            return
        # segments_to_update should always be a list for the below implementation
        for segment in segments_to_update:
            #print("" + key + " of " + channel_to_change + " is " +
            #      str(self.delta.ch_gegs[channel_to_change][key][segment]))
            self.delta.ch_gegs[channel_to_change][key][segment] *= ratio
            #print("" + key + " of " + channel_to_change +
            #      " is now " + str(self.delta.ch_gegs[channel_to_change][key][segment]))
        return

    def split_channel(self, channel_to_split="Hartelkanaal v2", location=0.5, next_weir_number=3):
        # split channel, example HK:
        # becomes HK1 & HK2 --> one junction to HK1, another to HK2
        old_channel = deepcopy(self.delta.ch_gegs[channel_to_split])

        new_channel1 = deepcopy(old_channel)
        new_channel2 = deepcopy(old_channel)
        new_channel1['Name'] = new_channel1['Name'] + "_1"
        new_channel2['Name'] = new_channel2['Name'] + "_2"
        # currently, this function only works for channels with one segment
        # TODO update for multiple segments
        print(old_channel)
        if len(new_channel1['L']) == 1:
            # channel 1 is sea side, channel 2 land side - check
            width_at_break_location = ((old_channel['b'][1] - old_channel['b'][0]) * location) + old_channel['b'][0]
            new_channel1['L'] = new_channel1['L'] * location
            new_channel1['b'][0] = width_at_break_location
            new_channel1['loc x=-L'] = 'w' + str(next_weir_number)

            new_channel2['L'] = new_channel2['L'] * (1 - location)
            new_channel2['b'][1] = width_at_break_location
            new_channel2['loc x=0'] = 'w' + str(next_weir_number + 1)

            # print(new_channel1)
            # print(new_channel2)

        def multiline_interpolate_point(line_geometry, distance):
            new_line1_coordinates = []
            line_points = get_coordinates(line_geometry)
            new_line1_coordinates.append(list(line_points[0]))
            for i in range(len(line_points) - 1):
                # create LineString and use .length instead?
                dist = sqrt(
                    (line_points[i + 1][0] - line_points[i][0]) ** 2 + (line_points[i + 1][1] - line_points[i][1]) ** 2)
                if distance <= dist:
                    """
                    line_segment = LineString(
                        [(line_points[i][0], line_points[i][1]), (line_points[i + 1][0], line_points[i + 1][1])])
                    split_point = line_interpolate_point(line_segment, distance)
                    new_line1_coordinates.append([split_point.x, split_point.y])
                    """
                    break
                else:
                    new_line1_coordinates.append(list(line_points[i + 1]))
                    distance = distance - dist

            new_line1_coordinates = np.array(new_line1_coordinates)
            new_line2_coordinates = []
            for points in line_points:
                if not np.any(new_line1_coordinates == points):
                    new_line2_coordinates.append(points)
            new_line2_coordinates = np.array(new_line2_coordinates)
            return new_line1_coordinates, new_line2_coordinates

        line = LineString(zip(old_channel['plot x'], old_channel['plot y']))
        distance_to_split = line.length * location
        new_line1, new_line2 = multiline_interpolate_point(line, distance_to_split)

        new_channel1['plot x'] = new_line1[:, 0]
        new_channel1['plot y'] = new_line1[:, 1]
        new_channel2['plot x'] = new_line2[:, 0]
        new_channel2['plot y'] = new_line2[:, 1]

        key1 = channel_to_split + "_1"
        key2 = channel_to_split + "_2"

        self.delta.ch_gegs.pop(channel_to_split)
        self.delta.ch_pars.pop(channel_to_split)
        self.delta.ch_outp.pop(channel_to_split)
        self.delta.ch_tide.pop(channel_to_split)
        self.delta.ch_keys.remove(channel_to_split)

        self.delta.ch_gegs[key1] = new_channel1
        self.delta.ch_keys.append(key1)
        self.delta.ch_gegs[key2] = new_channel2
        self.delta.ch_keys.append(key2)

        self.delta.add_properties(key1)
        self.delta.add_properties(key2)

        for i in range(2):
            self.delta.Qweir = np.vstack([self.delta.Qweir, np.zeros(len(self.delta.Qweir[0]))])
            self.delta.swe = np.vstack([self.delta.swe, np.array([0.15 + np.zeros(len(self.delta.swe[0]))])])
        self.delta.run_checks()
        network_df = pd.DataFrame.from_dict(self.delta.ch_gegs)
        network_df = network_df.T
        self._network = network_df
        return

    def create_plots(self):
        # visualisation
        self.delta.plot_s_gen_td(0)
        self.delta.plot_s_gen_td(-1)

        # for Rhine-Meuse
        out427 = self.delta.plot_salt_pointRM(4.2, 51.95, 1)
        out427 = self.delta.plot_salt_pointRM(4.5, 51.85, 1)
        out427 = self.delta.plot_salt_pointRM(4.5, 51.92, 1)

        self.delta.calc_X2_td()
        return



