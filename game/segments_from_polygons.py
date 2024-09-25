import numpy as np
import pandas as pd

def branches_to_segment(network_gdf):
    new_network_gdf = network_gdf.copy()
    def add_segments(branch_length, polygon_ids):
        number_of_segments = len(polygon_ids) * 2 - 2
        segment_length = sum(branch_length) / number_of_segments
        segments = [segment_length for segments in range(number_of_segments + len(branch_length) - 1)]
        index_to_update = {}
        """
        if a channel already has segments, split it exactly where it's current segments end

        TODO: This works well for all channels that are completely inside the polygons. Some
        channels are however also outside of these, like the Lek. This is not yet taken into
        account, polygon based updates there will yield incorrect results. Update is needed
        to also track what part of the channel is inside the polygon and only divide that part
        into segments.
        """
        if len(branch_length) > 1:
            for i, l in enumerate(branch_length):
                for s in range(number_of_segments):
                    if s * segment_length < np.sum(branch_length[: i +1]):
                        try:
                            if (s + 1) * segment_length > np.sum(branch_length[: i +1]):
                                index_to_update[i] = [s, s+ 1]
                        except IndexError:
                            pass
        segment_to_polygons = []
        for i in range(0, len(segments)):
            index = min(int(0.5 + (i / 2)), len(polygon_ids))
            segment_to_polygons.append(polygon_ids[index])
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
        return pd.Series([np.array(segments), np.array(segment_to_polygon_flat)])

    new_network_gdf[["new_L", "segment_ids"]] = new_network_gdf.apply(
        lambda row: add_segments(row["L"], row["polygon_ids"]), axis=1)

    def update_width(branch_width, branch_segments, old_branch_length):
        segments = np.concatenate(([0], branch_segments))
        cum_segments = np.cumsum(segments)
        branch_x = np.cumsum(np.concatenate(([0], old_branch_length)))
        interp_width = np.interp(cum_segments, branch_x, branch_width)
        return interp_width

    new_network_gdf["new_b"] = new_network_gdf.apply(lambda row: update_width(row["b"], row["new_L"], row["L"]), axis=1)

    def update_depth(branch_depth, branch_segments, old_branch_length):
        new_depths = [branch_depth[0]] * len(branch_segments)
        for j, old_L in enumerate(old_branch_length):
            for i, new_L in enumerate(branch_segments):

                if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                    new_depths[i] = branch_depth[j]
        return np.array(new_depths)

    new_network_gdf["new_Hn"] = new_network_gdf.apply(lambda row: update_depth(row["Hn"], row["new_L"], row["L"]),
                                                      axis=1)

    def update_depth(branch_dx, branch_segments, old_branch_length):
        new_dx = [branch_dx[0]] * len(branch_segments)
        for j, old_L in enumerate(old_branch_length):
            for i, new_L in enumerate(branch_segments):
                if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                    new_dx[i] = branch_dx[j]
        return np.array(new_dx)


    new_network_gdf["new_dx"] = new_network_gdf.apply(lambda row: update_depth(row["dx"], row["new_L"], row["L"]),
                                                      axis=1)

    new_network_gdf = new_network_gdf.drop(columns=["Hn", "L", "b", "dx"])
    new_network_gdf = new_network_gdf.rename(columns={"new_Hn": "Hn", "new_L": "L", "new_b": "b", "new_dx": "dx"})
    return new_network_gdf