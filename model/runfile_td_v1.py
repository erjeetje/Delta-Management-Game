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
    def __init__(self, scenario):
        self.current_forcing = settings_td_v1.set_forcing(scenario=scenario)
        self.delta = core_td_v1.mod42_netw(settings_td_v1.constants, settings_td_v1.geo_pars, self.current_forcing,
                                           settings_td_v1.phys_pars)#, pars_seadom = (25000,100,10), pars_rivdom = (200000,2000,0))
        network_df = pd.DataFrame.from_dict(settings_td_v1.geo_pars)
        network_df = network_df.T
        self._network = network_df
        self._output = None
        return

    @property
    def output(self):
        return self._output

    @property
    def network(self):
        return self._network

    def run_model(self):
        # calculate river discharge distribution
        self.delta.run_model()
        self.delta.calc_output()
        output_df = pd.DataFrame.from_dict(self.delta.ch_outp)
        self._output = output_df.T
        return

    def change_forcings(self, scenario, add_rows=0):
        try:
             self.current_forcing = settings_td_v1.set_forcing(scenario=scenario)
             [self.delta.Qriv, self.delta.Qweir, self.delta.Qhar, self.delta.n_sea, self.delta.soc,
              self.delta.sri, self.delta.swe, self.delta.tid_per, self.delta.a_tide, self.delta.p_tide,
              self.delta.T, self.delta.DT] = self.current_forcing
        except TypeError:
            return
        if add_rows != 0:
            for i in range(add_rows):
                self.delta.Qweir = np.vstack([self.delta.Qweir, np.zeros(len(self.delta.Qweir[0]))])
                self.delta.swe = np.vstack([self.delta.swe, np.array([0.15 + np.zeros(len(self.delta.swe[0]))])])
        return

    """
    def add_segments_to_channels(self, model_network_gdf):
        model_network_gdf = model_network_gdf.set_index("Name")
        for index, row in model_network_gdf.iterrows():
            for key in ["Hn", "L", "b", "dx"]:
                self.delta.ch_gegs[index][key] = row[key]
            self.delta.add_properties(index, initial_update=False)
        self.delta.run_checks()
        return
    """

    def update_channel_geometries(self, model_network_gdf):
        model_network_gdf = model_network_gdf.set_index("Name")
        for index, row in model_network_gdf.iterrows():
            if row["changed"]:
                for key in ["Hn", "L", "b", "dx"]:
                    old = self.delta.ch_gegs[index][key]
                    print("changed:", key, "of", index, "from", old, "to", row[key])
                    self.delta.ch_gegs[index][key] = row[key]
                self.delta.add_properties(index, initial_update=False)
        self.delta.run_checks()
        return

    def update_channels_geometry(self, model_network_gdf):
        channels_to_change = model_network_gdf.loc[model_network_gdf['changed'] == True]
        channels_to_change = channels_to_change.set_index("Name")
        for index, row in channels_to_change.iterrows():
            for key in ["Hn", "L", "b", "dx"]:
                self.delta.ch_gegs[index][key] = row[key]
            self.delta.add_properties(index, initial_update=True)
        self.delta.run_checks()
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



