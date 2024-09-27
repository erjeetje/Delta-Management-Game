import numpy as np
import pandas as pd


def index_polygons_to_channel_geometry(network_gdf):
    new_network_gdf = network_gdf.copy()

    def add_segments(branch_length, polygon_ids):
        number_of_segments = len(polygon_ids) * 2 - 2
        segment_length = sum(branch_length) / number_of_segments
        segments = [segment_length for segments in range(number_of_segments)]
        # segments = [segment_length for segments in range(number_of_segments + len(branch_length) - 1)]
        # index_to_update = {}

        """

        if a channel already has segments, split it exactly where it's current segments end

        TODO: This works well for all channels that are completely inside the polygons. Some
        channels are however also outside of these, like the Lek. This is not yet taken into
        account, polygon based updates there will yield incorrect results. Update is needed
        to also track what part of the channel is inside the polygon and only divide that part
        into segments.



        this part can be dropped, used later when actually making changes

        if len(branch_length) > 1:
            for s in range(number_of_segments):
                for i, l in enumerate(branch_length):
                    if s * segment_length <= np.sum(branch_length[:i+1]):
                        try:
                            if (s + 1) * segment_length > np.sum(branch_length[:i+1]):
                                index_to_update[i] = [s, s+1]
                                break
                        except IndexError:
                            pass
        """
        segment_to_polygons = []
        for i in range(0, len(segments)):
            index = min(int(0.5 + (i / 2)), len(polygon_ids))
            segment_to_polygons.append(polygon_ids[index])
        """
        for key, value in index_to_update.items():
            old_L = branch_length[key]
            segment_left = np.sum(segments[:value[0]])
            segment_right = np.sum(segments[:value[1]])
            segments[value[0]] = old_L - segment_left
            segments[value[1]] = segment_right - old_L
            index1 = min(int(0.5 + (value[0] / 2)), len(polygon_ids))
            index2 = min(int(0.5 + (value[1] / 2)), len(polygon_ids))
            segment_to_polygons[value[0]] = [polygon_ids[index1], polygon_ids[index2]]
            del segment_to_polygons[-1]
        segment_to_polygon_flat = []
        for s in segment_to_polygons:
            if isinstance(s, (int, np.integer)):
                segment_to_polygon_flat.append(s)
            elif isinstance(s, list):
                for p in s:
                    segment_to_polygon_flat.append(p)
        """
        polygons_segment_id = {}
        for i, polygon in enumerate(segment_to_polygons):
            polygons_segment_id[polygon] = np.sum(segments[:i + 1])
        return pd.Series([list(polygons_segment_id.values()), segment_to_polygons])

    # test = new_network_gdf.loc[["Hollandse IJssel", "Lek"]]
    # test["polygon_to_L"] = test.apply(lambda row: add_segments(row["L"], row["polygon_ids"]), axis=1)
    new_network_gdf[["polygon_to_L", "polygon_to_segment"]] = new_network_gdf.apply(
        lambda row: add_segments(row["L"], row["polygon_ids"]), axis=1)

    new_network_gdf = new_network_gdf.rename(
        columns={"Hn": "ref_Hn", "L": "ref_L", "b": "ref_b", "dx": "ref_dx"})
    return new_network_gdf

def create_polygon_id_tracker(network_gdf):
    polygon_index_df = pd.DataFrame(network_gdf) # convert to df as gdf does not support multi-column explode
    polygon_index_df = polygon_index_df.explode(["polygon_ids", "polygon_to_L"])
    polygon_index_df = polygon_index_df.reset_index()
    polygon_index_df = polygon_index_df[["index", "Name", "polygon_ids"]]
    polygon_index_df["index_in_polygon"] = polygon_index_df.groupby(["polygon_ids"])["index"].transform(
        lambda x: ','.join(x))
    polygon_index_df["name_in_polygon"] = polygon_index_df.groupby(["polygon_ids"])["Name"].transform(
        lambda x: ','.join(x))
    polygon_index_df = polygon_index_df[["polygon_ids", "index_in_polygon", "name_in_polygon"]]
    polygon_index_df = polygon_index_df.drop_duplicates()
    polygon_index_df["index_in_polygon"] = polygon_index_df["index_in_polygon"].apply(lambda x: x.split(','))
    polygon_index_df["name_in_polygon"] = polygon_index_df["name_in_polygon"].apply(lambda x: x.split(','))
    polygon_index_df = polygon_index_df.set_index("polygon_ids")
    polygon_index_df["ref_red_marker"] = 1
    polygon_index_df["ref_blue_marker"] = 1
    polygon_index_df["red_marker"] = 1
    polygon_index_df["blue_marker"] = 1
    polygon_index_df["changed"] = False
    return polygon_index_df