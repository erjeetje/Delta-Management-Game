import numpy as np
import pandas as pd
from copy import deepcopy
from math import sqrt
from shapely import get_coordinates, line_interpolate_point
from shapely.geometry import LineString


def update_polygon_tracker(polygon_df, markers):
    for key, value in markers.items():
        if polygon_df.loc[key, "red_marker"] != value[0]:
            polygon_df.at[key, "red_marker"] = value[0]
            polygon_df.at[key, "changed"] = True
        if polygon_df.loc[key, "blue_marker"] != value[1]:
            polygon_df.at[key, "blue_marker"] = value[1]
            polygon_df.at[key, "changed"] = True
    return polygon_df


def to_change(changed_polygons_df):
    polygons_df = changed_polygons_df.copy()
    def compare_reds(ref_red, cur_red):
        if ref_red == cur_red:
            return None
        if cur_red == 3:
            return "split"
        if ref_red > cur_red:
            return "deepen"
        if ref_red < cur_red:
            return "undeepen"
    polygons_df["vertical_change"] = polygons_df.apply(lambda row: compare_reds(row["ref_red_marker"], row["red_marker"]), axis=1)

    def compare_blues(ref_blue, cur_blue):
        if ref_blue == cur_blue:
            return None
        if ref_blue < cur_blue:
            return "widen"
        if ref_blue > cur_blue:
            return "narrow"
    polygons_df["horizontal_change"] = polygons_df.apply(lambda row: compare_blues(row["ref_blue_marker"], row["blue_marker"]), axis=1)

    return polygons_df


def geometry_to_update(changed_polygons_df, new_model_network_df):
    new_model_network_df["changed_polygons"] = [list() for x in range(len(new_model_network_df.index))]
    new_model_network_df["vertical_change"] = new_model_network_df.apply(lambda row: [None for x in range(len(row["polygon_ids"]))], axis=1)
    new_model_network_df["horizontal_change"] = new_model_network_df.apply(lambda row: [None for x in range(len(row["polygon_ids"]))], axis=1)
    new_model_network_df["changed"] = False
    for index, row in changed_polygons_df.iterrows():
        for channel in row["index_in_polygon"]:
            new_model_network_df.at[channel, "changed_polygons"] += [index]
            new_model_network_df.at[channel, "vertical_change"][list(new_model_network_df.at[channel, "polygon_ids"]).index(index)] = row["vertical_change"]
            new_model_network_df.at[channel, "horizontal_change"][list(new_model_network_df.at[channel, "polygon_ids"]).index(index)] = row["horizontal_change"]
            new_model_network_df.at[channel, "changed"] = True
    return new_model_network_df

def update_channel_length(model_network_df):
    """
    Function that creates new segments based on changes on the game board.

    TODO: modify code (or add function) to "knit" segments back together if depths are the same and if width interpolates

    TODO: modify how segments are created so that depth/width changes in one part of the channel do not/only limitedly affect the rest of the channel
    """
    updated_model_network_df = model_network_df.copy()

    def update_L(old_L, segment_L, polygon_ids, changed_polygons, changed, ver_change, hor_change, ref_dx):
        if not changed:
            # print("I am not changed")
            return pd.Series([old_L, ref_dx, [None for l in old_L], [None for l in old_L]])
        # if np.array_equal(polygon_ids, np.array(changed_polygons)):
        #    #print("All my polygons changed")
        #    return pd.Series([old_L, [None for l in old_L], [None for l in old_L]])
        poly_check = set(changed_polygons)
        changed_poly_idx = [i for i, e in enumerate(polygon_ids) if e in poly_check]
        unchanged_poly_idx = [i for i, e in enumerate(polygon_ids) if e not in poly_check]
        changed_segments_idx = []
        unchanged_segments_idx = []
        for i, idx in enumerate(changed_poly_idx):
            try:
                if (idx + 1) != changed_poly_idx[i + 1]:
                    changed_segments_idx.append(idx)
            except IndexError:
                changed_segments_idx.append(idx)
        for i, idx in enumerate(unchanged_poly_idx):
            try:
                if (idx + 1) != unchanged_poly_idx[i + 1]:
                    unchanged_segments_idx.append(idx)
            except IndexError:
                unchanged_segments_idx.append(idx)

        merged_segment_idx = changed_segments_idx + unchanged_segments_idx
        merged_segment_order = sorted(merged_segment_idx)
        new_L = []
        substract_L = 0
        for idx in merged_segment_order:
            new_L.append(segment_L[idx] - substract_L)
            substract_L = sum(new_L)
            """
            try:
                new_L.append(segment_L[idx] - new_L[-1])
            except IndexError:
                new_L.append(segment_L[idx])
            """

        old_L_idx = [0] * len(new_L)
        if len(old_L) > 1:
            for j, l in enumerate(old_L):
                for i, k in enumerate(new_L):
                    if np.sum(new_L[:i]) >= np.sum(old_L[:j]):
                        # new_depths[i] = branch_depth[j]
                        old_L_idx[i] = j
        new_L_dx = []
        new_dx = []
        dx_fraction_ref = 0
        for i, idx in enumerate(old_L_idx):
            segment_dx = ref_dx[idx]
            dx_fraction = round(np.sum(new_L[:i + 1]) / segment_dx, 0)
            new_segment = ((dx_fraction - dx_fraction_ref) * segment_dx)
            new_L_dx.append(new_segment)
            new_dx.append(float(segment_dx))
            dx_fraction_ref = dx_fraction
        new_dx = np.array(new_dx, dtype=float)
        new_L_dx = np.array(new_L_dx, dtype=float)
        ver_changes_to_segments = []
        hor_changes_to_segments = []
        for idx in merged_segment_order:
            if idx in changed_segments_idx:
                ver_changes_to_segments.append(ver_change[idx])
                hor_changes_to_segments.append(hor_change[idx])
            elif idx in unchanged_segments_idx:
                ver_changes_to_segments.append(ver_change[idx])
                hor_changes_to_segments.append(hor_change[idx])
        return pd.Series([new_L_dx, new_dx, ver_changes_to_segments, hor_changes_to_segments])

    updated_model_network_df[
        ["L", "dx", "ver_changed_segments", "hor_changed_segments"]] = updated_model_network_df.apply(
        lambda row: update_L(row["ref_L"], row["polygon_to_L"], row["polygon_ids"], row["changed_polygons"],
                             row["changed"], row["vertical_change"], row["horizontal_change"], row["ref_dx"]), axis=1)
    return updated_model_network_df


def update_channel_references(turn_model_network_df):
    model_network_df = turn_model_network_df.copy()
    def segment_width(branch_width, branch_segments, old_branch_length, changed):
        if not changed:
            return branch_width
        segments = np.concatenate(([0], branch_segments))
        cum_segments = np.cumsum(segments)
        branch_x = np.cumsum(np.concatenate(([0], old_branch_length)))
        interp_width = np.interp(cum_segments, branch_x, branch_width)
        return interp_width
    model_network_df["b"] = model_network_df.apply(
        lambda row: segment_width(row["ref_b"], row["L"], row["ref_L"], row["changed"]), axis=1)

    def segment_depth(branch_depth, branch_segments, old_branch_length, changed):
        if not changed:
            return branch_depth
        new_depths = [branch_depth[0]] * len(branch_segments)
        if len(old_branch_length) > 1:
            for j, old_L in enumerate(old_branch_length):
                for i, new_L in enumerate(branch_segments):
                    if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                        new_depths[i] = branch_depth[j]
        return np.array(new_depths)
    model_network_df.at["Lek", "ref_Hn"] = [5.3, 7.3]
    model_network_df["Hn"] = model_network_df.apply(
        lambda row: segment_depth(row["ref_Hn"], row["L"], row["ref_L"], row["changed"]), axis=1)
    return model_network_df


def update_channel_geometry(turn_model_network_df):
    model_network_df = turn_model_network_df.copy()
    def update_depth(branch_depth, changed, ver_change):
        if not changed:
            return branch_depth
        for i, change in enumerate(ver_change):
            #print(change)
            if change == "deepen":
                branch_depth[i] *= 1.25
            if change == "undeepen":
                branch_depth[i] *= 0.75
        return branch_depth
    model_network_df["Hn"] = model_network_df.apply(
        lambda row: update_depth(row["Hn"], row["changed"], row["ver_changed_segments"]), axis=1)

    def update_width(branch_width, changed, hor_change):
        if not changed:
            return branch_width
        for i, change in enumerate(hor_change):
            #print(change)
            if change == "widen":
                branch_width[i] *= 1.25
                branch_width[i+1] *= 1.25
            if change == "narrow":
                branch_width[i] *= 0.75
                branch_width[i+1] *= 0.75
        return branch_width
    model_network_df["b"] = model_network_df.apply(
        lambda row: update_width(row["b"], row["changed"], row["hor_changed_segments"]), axis=1)
    return model_network_df


def apply_split(turn_model_network_df, next_weir_number=3):
    model_network_df = turn_model_network_df.copy()

    def check_split(index, row):
        return_value = None
        if not row["changed"]:
            return return_value
        for value in row["ver_changed_segments"]:
            if value == "split":
                return_value = index
        return return_value

    channels_to_split = []
    for index, row in model_network_df.iterrows():
        channel_name = check_split(index, row)
        if channel_name is not None:
            channels_to_split.append(channel_name)
    if not channels_to_split:
        return model_network_df, {}

    def split_channel(channel, next_weir_number=3):
        old_channel = channel.copy()
        new_channel1 = old_channel.copy()
        new_channel2 = old_channel.copy()
        new_channel1.name = new_channel1.name + "_1"
        new_channel1['Name'] = new_channel1['Name'] + "_1"
        new_channel2.name = new_channel2.name + "_2"
        new_channel2['Name'] = new_channel2['Name'] + "_2"

        split_index = old_channel["ver_changed_segments"].index("split")
        # TODO check that any segment L >= dx * 4
        # TODO check
        reference_L = old_channel['L']

        new_channel1['L'] = deepcopy(reference_L[:split_index + 1])
        new_channel1['L'][-1] = new_channel1['L'][-1] / 2
        location = sum(new_channel1["L"]) / sum(old_channel["L"])
        print("split point (fraction of total L) =", location)
        width_at_break_location = (old_channel['b'][split_index + 1] - old_channel['b'][split_index]) * location + \
                                  old_channel['b'][split_index]
        reference_b = old_channel['b']
        reference_Hn = old_channel['Hn']
        reference_ver = old_channel['ver_changed_segments']
        reference_hor = old_channel['hor_changed_segments']

        new_channel1['b'] = deepcopy(reference_b[:split_index + 2])
        new_channel1['b'][-1] = width_at_break_location
        new_channel1['dx'] = np.array([new_channel1['dx'][0] for i in new_channel1['L']])
        new_channel1['loc x=-L'] = 'w' + str(next_weir_number)
        new_channel1['Hn'] = deepcopy(reference_Hn[:split_index + 1])
        new_channel1['ver_changed_segments'] = deepcopy(reference_ver[:split_index + 1])
        new_channel1['hor_changed_segments'] = deepcopy(reference_hor[:split_index + 1])

        new_channel2['L'] = deepcopy(reference_L[split_index:])
        new_channel2['L'][0] = new_channel2['L'][0] / 2
        new_channel2['b'] = deepcopy(reference_b[split_index:])
        new_channel2['b'][0] = width_at_break_location
        new_channel2['dx'] = np.array([new_channel2['dx'][0] for i in new_channel1['L']])
        new_channel2['loc x=0'] = 'w' + str(next_weir_number + 1)
        new_channel2['Hn'] = deepcopy(reference_Hn[split_index:])
        new_channel2['ver_changed_segments'] = deepcopy(reference_ver[split_index:])
        new_channel2['hor_changed_segments'] = deepcopy(reference_hor[split_index:])

        polygon_ids = list(old_channel["polygon_ids"])
        polygon_id_idx = polygon_ids.index(new_channel2["changed_polygons"][0])
        new_channel1["polygon_ids"] = new_channel1["polygon_ids"][:polygon_id_idx + 1]
        new_channel1["vertical_change"] = new_channel1["vertical_change"][:polygon_id_idx + 1]
        new_channel1["horizontal_change"] = new_channel1["horizontal_change"][:polygon_id_idx + 1]
        new_channel2["polygon_ids"] = new_channel2["polygon_ids"][polygon_id_idx:]
        new_channel2["vertical_change"] = new_channel2["vertical_change"][polygon_id_idx:]
        new_channel2["horizontal_change"] = new_channel2["horizontal_change"][polygon_id_idx:]

        polygon_segments = list(old_channel["polygon_to_segment"])
        polygon_segment_idx = polygon_segments.index(new_channel1["changed_polygons"][0])
        new_channel1["polygon_to_segment"] = new_channel1["polygon_to_segment"][:polygon_id_idx + 1]
        new_channel2["polygon_to_segment"] = new_channel2["polygon_ids"][polygon_segment_idx:]

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
        new_channels = pd.concat([new_channel1, new_channel2], axis=1)
        return new_channels.T

    channel_reference = {}
    for channel in channels_to_split:
        new_channels = split_channel(model_network_df.loc[channel], next_weir_number=next_weir_number)
        channel_reference[model_network_df.loc[channel, 'Name']] = list(new_channels['Name'].values)
        model_network_df = model_network_df.drop(channel)
        # model_network_df = model_network_df.concat(new_channel1) # ignore_index=True
        # model_network_df = model_network_df.concat(new_channel2)
        merged_df = pd.concat([model_network_df, new_channels])
    return merged_df, channel_reference