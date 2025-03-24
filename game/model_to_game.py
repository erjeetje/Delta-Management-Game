#import time
import numpy as np
import pandas as pd
import geopandas as gpd
#from datetime import timedelta
from shapely import intersects, get_coordinates, line_interpolate_point
from shapely.geometry import Point, LineString, shape
from scipy.spatial import cKDTree
from copy import deepcopy


def process_model_network(model_network_df):
    model_network_df = model_network_df.reset_index()
    model_network_df = model_network_df.rename(columns={"Name": "index", "index": "Name"})
    model_network_df = model_network_df.set_index("index")

    def add_linestring_geometry(x_coor, y_coor):
        geometry = LineString([Point(xy) for xy in zip(x_coor, y_coor)])
        return geometry

    model_network_df["line_geometry"] = model_network_df.apply(
        lambda row: add_linestring_geometry(row["plot x"], row["plot y"]), axis=1)

    model_network_gdf = gpd.GeoDataFrame(model_network_df, geometry=model_network_df["line_geometry"], crs="EPSG:4326")
    return model_network_gdf

def remove_sea_river_domains(model_network_gdf):
    def remove_added_geometry(name, L, Hn, b, dx):
        if name == "Breeddiep":
            L = L[:-1]
            Hn = Hn[:-1]
            b = b[:-1]
            dx = dx[:-1]
        river_channels = ["Waal", "Maas"]
        if name in river_channels:
            L = L[1:]
            Hn = Hn[1:]
            b = b[1:]
            dx = dx[1:]
        return pd.Series([L, Hn, b, dx])

    model_network_gdf[["L", "Hn", "b", "dx"]] = model_network_gdf.apply(lambda row: remove_added_geometry(
        row.name, row["L"], row["Hn"], row["b"], row["dx"]), axis=1)
    return model_network_gdf

def find_neighbours(hexagons):
    hex_coor = []
    polygons = []
    hexagon0_y = 0
    hexagon1_y = 0
    hexagon_count = 0
    for feature in hexagons.features:
        if not feature.properties["ghost_hexagon"]:
            hexagon_count += 1
        geom = shape(feature.geometry)
        x_hex = geom.centroid.x
        y_hex = geom.centroid.y
        if feature.id == 0:
            hexagon0_y = y_hex
        if feature.id == 1:
            hexagon1_y = y_hex
        hex_coor.append([x_hex, y_hex])
        polygons.append(shape)
    hex_coor = np.array(hex_coor)
    hex_locations = cKDTree(hex_coor)
    limit = abs((hexagon0_y - hexagon1_y) * 1.5)
    def remove_values_from_array(array, val):
        return [value for value in array if value < val]
    for feature in hexagons.features:
        if feature.properties["ghost_hexagon"]:
            continue
        geom = shape(feature.geometry)
        x_hex = geom.centroid.x
        y_hex = geom.centroid.y
        xy = np.array([x_hex, y_hex])
        # find all hexagons within the limit radius
        dist, indices = hex_locations.query(
                xy, k=7, distance_upper_bound=limit)
        # remove missing neighbours (return as self.n, equal to total_hex)
        indices = remove_values_from_array(indices, hexagon_count)
        # convert from int32 to regular int (otherwise JSON error)
        indices = list(map(int, indices))
        # remove itself
        indices.pop(0)
        if False:
            print("Neighbouring hexagons for hexagon " + str(feature.id) +
                  " are: " + str(indices))
        feature.properties["neighbours"] = indices
    return hexagons

def match_hexagon_properties(hexagons_to_update, hexagons_with_properties, property_key):
    assert len(hexagons_to_update.features) == len(hexagons_with_properties.features)
    for feature in hexagons_to_update.features:
        ref_hex = hexagons_with_properties.features[feature.id]
        if type(property_key) is str:
            feature.properties[property_key] = ref_hex.properties[property_key]
        elif type(property_key) is list:
            for key in property_key:
                feature.properties[key] = ref_hex.properties[key]
        else:
            print("unsupported type for property keys")
    return hexagons_to_update

def find_neighbour_edges(hexagons):
    """
    - store midpoint of each hexagon edge in list
    - store feature.id of each midpoint in list
    - nearest neighbour of each midpoint (with narrow limit)
    - index edge location (linestring, midpoint, feature.ids of both sides)
    """
    edges = []
    edges_midpoint = []
    edges_id = []
    shortest_line = 0
    for feature in hexagons.features:
        points = feature.geometry["coordinates"][0]
        for i in range(len(points)):
            x1 = points[i][0]
            y1 = points[i][1]
            try:
                x2 = points[i+1][0]
                y2 = points[i+1][1]
            except IndexError:
                continue
            edge = LineString([(x1 ,y1), (x2, y2)])
            edges.append(edge)
            mid_point_x = (x1 + x2) / 2
            mid_point_y = (y1 + y2) / 2
            edges_midpoint.append([mid_point_x, mid_point_y])
            edges_id.append(feature.id)
            if shortest_line == 0:
                shortest_line = edge.length
            elif edge.length < shortest_line:
                shortest_line = edge.length
    edges_midpoint = np.array(edges_midpoint)
    edges_locations = cKDTree(edges_midpoint)
    limit = abs(shortest_line * 0.1)
    edge_count = len(edges_midpoint)
    connected_edges = []
    sorted_midpoints = []

    def remove_values_from_array(array, val):
        return [value for value in array if value < val]
    for i, midpoint in enumerate(edges_midpoint):
        dist, indices = edges_locations.query(
                midpoint, k=2, distance_upper_bound=limit)
        if edge_count not in indices:
            connected_edges.append([edges_id[indices[0]], edges_id[indices[1]]])
            sorted_midpoints.append(midpoint)

    for feature in hexagons.features:
        feature.properties["neighbour_edge_midpoint"] = [0 for i in range(len(feature.properties["neighbours"]))]
        for i, ids in enumerate(connected_edges):
            if len(ids) != 2:
                continue
            if feature.id in ids:
                ids_copy = deepcopy(ids)
                ids_copy.remove(feature.id)
                neighbour_id = ids_copy[0]
                neighbour_index = feature.properties["neighbours"].index(neighbour_id)
                feature.properties["neighbour_edge_midpoint"][neighbour_index] = sorted_midpoints[i].tolist()
    return hexagons

def find_branch_intersections(polygons, branches):
    for polygon in polygons.features:
        polygon.properties["branches"] = []
        polygon.properties["branch_crossing"] = {}
        polygon.properties["branch_crossing_coor"] = {}
        poly_geom = shape(polygon.geometry)
        branches["polygon_ids"] = ''
        branches["polygon_ids"] = branches["polygon_ids"].apply(list)
        for index, branch in branches.iterrows():
            line = branch["geometry"]
            if intersects(poly_geom, line):
                polygon.properties["branches"].append(branch["Name"])
                polygon_bb = poly_geom.boundary.coords
                linestrings = [LineString(polygon_bb[k:k+2]) for k in range(len(polygon_bb) - 1)]
                for i, linestring in enumerate(linestrings):
                    if intersects(linestring, line):
                        midpoint = Point(
                            (linestring.coords[0][0] + linestring.coords[-1][0]) / 2,
                            (linestring.coords[0][1] + linestring.coords[-1][1]) / 2)
                        inters = linestring.intersection(line)
                        if inters.geom_type == "Point":
                            coords = np.asarray(inters.coords.xy)
                        elif inters.geom_type == "MultiPoint":
                            coords = np.asarray(l.coords.xy for l in inters.geoms)
                        else:
                            print(inters.geom_type)
                        for neighbour in polygon.properties["neighbours"]:
                            neighbour_polygon = polygons.features[neighbour]
                            neighbour_shape = shape(neighbour_polygon.geometry)
                            neighbour_bb = neighbour_shape.boundary.coords
                            neighbour_linestrings = [LineString(
                                neighbour_bb[k:k+2]) for k in range(len(neighbour_bb) - 1)]
                            for neighbour_linestring in neighbour_linestrings:
                                neighbour_midpoint = Point(
                                    (neighbour_linestring.coords[0][0] + neighbour_linestring.coords[-1][0]) / 2,
                                    (neighbour_linestring.coords[0][1] + neighbour_linestring.coords[-1][1]) / 2)
                                #if polygon.id == 116:
                                #    print(neighbour_midpoint.distance(midpoint))
                                if neighbour_midpoint.distance(midpoint) < 0.006:
                                    polygon.properties["branch_crossing"][neighbour_polygon.id] = branch["Name"]
                                    #print("branch crossing between", polygon.id, "and", neighbour_polygon.id)
                                    #print(branch.properties["id"], coords)
                                    #polygon.properties["branch_crossing_coor"][branch.properties["id"]] = coords
    return polygons, branches

def determine_polygon_intersections(branches_gdf, polygons):
    def determine_polygons(line, polygons):
        """
        The nested for loops make this function rather slow, but it does correctly map the order of polygons
        """
        intersecting_polygons = []
        line_points = get_coordinates(line)
        for i, line_point in enumerate(line_points):
            point = Point(line_point)
            for polygon in polygons.features:
                poly_geom = shape(polygon.geometry)
                if poly_geom.contains(point):
                    if polygon.id not in intersecting_polygons:
                        intersecting_polygons.append(polygon.id)
        if False:
            print(i, intersecting_polygons)
        if intersecting_polygons:
            return intersecting_polygons
        else:
            return np.nan

    # this part is not so neat, but the use of points makes the order correct, what happens below is
    # redrawing the hollandse diep with more points, to ensure all polygons are covered.
    hollands_diep_geom = branches_gdf.loc["Hollands Diep 1"]["geometry"]
    distance = 0
    new_points = []
    while distance < hollands_diep_geom.length:
        new_point = hollands_diep_geom.interpolate(distance)
        new_points.append(new_point)
        distance += 0.01
    hollands_diep_geom_new = LineString(new_points)
    branches_gdf.loc["Hollands Diep 1", "geometry"] = hollands_diep_geom_new
    branches_gdf["polygon_ids"] = branches_gdf.apply(
        lambda row: determine_polygons(row["geometry"], polygons), axis=1)
    return branches_gdf.dropna()

"""
def branches_to_segment(network_gdf):
    new_network_gdf = network_gdf.copy()

    def add_segments(branch_length, polygon_ids):
        number_of_segments = len(polygon_ids) * 2 - 2
        segment_length = sum(branch_length) / number_of_segments
        segments = [segment_length for segments in range(number_of_segments + len(branch_length) - 1)]
        index_to_update = {}
        
        if a channel already has segments, split it exactly where it's current segments end

        TODO: This works well for all channels that are completely inside the polygons. Some
        channels are however also outside of these, like the Lek. This is not yet taken into
        account, polygon based updates there will yield incorrect results. Update is needed
        to also track what part of the channel is inside the polygon and only divide that part
        into segments.
        
        if len(branch_length) > 1:
            for s in range(number_of_segments):
                for i, l in enumerate(branch_length):
                    if s * segment_length <= np.sum(branch_length[:i + 1]):
                        try:
                            if (s + 1) * segment_length > np.sum(branch_length[:i + 1]):
                                index_to_update[i] = [s, s + 1]
                                break
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

    
    def old_to_new_L_obsolete(branch_segments, old_branch_length):
        old_L_idx = [0] * len(branch_segments)
        if len(old_branch_length) > 1:
            for j, old_L in enumerate(old_branch_length):
                for i, new_L in enumerate(branch_segments):
                    if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                        # new_depths[i] = branch_depth[j]
                        old_L_idx[i] = j
        return np.array(old_L_idx)


    def old_to_new_L(branch_segments, old_branch_length, branch_dx):
        old_L_idx = [0] * len(branch_segments)
        if len(old_branch_length) > 1:
            for j, old_L in enumerate(old_branch_length):
                for i, new_L in enumerate(branch_segments):
                    if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                        # new_depths[i] = branch_depth[j]
                        old_L_idx[i] = j
        branch_segments_dx = []
        dx_fraction_ref = 0
        for i, idx in enumerate(old_L_idx):
            dx_fraction = round(np.sum(branch_segments[:i + 1]) / branch_dx[idx], 0)
            # print(np.sum(branch_segments[:i+1]), branch_dx[idx],dx_fraction)
            new_segment = ((dx_fraction - dx_fraction_ref) * branch_dx[idx])
            branch_segments_dx.append(new_segment)
            dx_fraction_ref = dx_fraction
        return pd.Series([np.array(old_L_idx), np.array(branch_segments_dx)])

    new_network_gdf[["old_L_idx", "new_L"]] = new_network_gdf.apply(
        lambda row: old_to_new_L(row["new_L"], row["L"], row["dx"]), axis=1)

    #new_network_gdf["old_L_idx"] = new_network_gdf.apply(lambda row: old_to_new_L(row["new_L"], row["L"]), axis=1)

    def update_width(branch_width, branch_segments, old_branch_length):
        segments = np.concatenate(([0], branch_segments))
        cum_segments = np.cumsum(segments)
        branch_x = np.cumsum(np.concatenate(([0], old_branch_length)))
        interp_width = np.interp(cum_segments, branch_x, branch_width)
        return interp_width

    new_network_gdf["new_b"] = new_network_gdf.apply(lambda row: update_width(row["b"], row["new_L"], row["L"]), axis=1)

    
    def update_depth_old(branch_depth, branch_segments, old_branch_length):
        new_depths = [branch_depth[0]] * len(branch_segments)
        if len(old_branch_length) > 1:
            for j, old_L in enumerate(old_branch_length):
                for i, new_L in enumerate(branch_segments):
                    if np.sum(branch_segments[:i]) >= np.sum(old_branch_length[:j]):
                        new_depths[i] = branch_depth[j]
        print(new_depths)
        return np.array(new_depths)
    new_network_gdf["new_Hn"] = new_network_gdf.apply(lambda row: update_depth(row["Hn"], row["new_L"], row["L"]), axis=1)
    

    def update_depth(branch_depth, segment_idx):
        new_branch_depth = [branch_depth[i] for i in segment_idx]
        return np.array(new_branch_depth)

    new_network_gdf["new_Hn"] = new_network_gdf.apply(lambda row: update_depth(row["Hn"], row["old_L_idx"]), axis=1)

    
    def update_dx_old(branch_dx, branch_segments, old_branch_length):
        new_dx = []
        if len(old_branch_length) == 1:
            new_dx = [(branch_segments[0] / old_branch_length[0]) * branch_dx[0]] * len(branch_segments)
        else:

            #this needs to find how to divide the dx properly over the new segments. Needs to find the matches with the old
            #length and then find the fraction on the new segments, based on their length.

        points_to_divide = {}
        branch_idx = 0
        start_segment = 0
        for j, old_L in enumerate(old_branch_length):
            for i, new_L in enumerate(branch_segments):
                if np.sum(branch_segments[:i+1]) == np.sum(old_branch_length[:j+1]):
                    points_to_divide[j] = i
                    #print(np.sum(branch_segments[:i+1]))
                    #print(np.sum(old_branch_length[:j+1]))
                    #points_to_divide["old_segment"+str(j)] = i
                    #new_dx[i] = (branch_segments[j] / old_branch_length[j]) * branch_dx[j]
                    #for l in range(i):
                        #if branch_segments[i-1] == branch_segments[i]:
        print(points_to_divide)
        #new_dx = []
        #new_dx = [(branch_segments[0] / old_branch_length[0]) * branch_dx[0]] * len(branch_segments)
        #for j, old_L in enumerate(old_branch_length):
        #    for i, new_L in enumerate(branch_segments):
        #        if np.sum(branch_segments[:i]) == np.sum(old_branch_length[:j]):
        #            new_dx[i] = (branch_segments[j] / old_branch_length[j]) * branch_dx[j]
        return np.array(new_dx)
    new_network_gdf["new_dx"] = new_network_gdf.apply(lambda row: update_dx(row["dx"], row["new_L"], row["L"]), axis=1)
    

    def update_dx(branch_dx, branch_segments, old_branch_length, segment_idx):
        new_dx = [branch_dx[0]] * len(branch_segments)
        if len(old_branch_length) > 1:
            for i, idx in enumerate(segment_idx):
                new_dx[i] = branch_dx[idx]
        return np.array(new_dx)

    new_network_gdf["new_dx"] = new_network_gdf.apply(
        lambda row: update_dx(row["dx"], row["new_L"], row["L"], row["old_L_idx"]), axis=1)

    new_network_gdf = new_network_gdf.drop(columns=["Hn", "L", "b", "dx"])
    new_network_gdf = new_network_gdf.rename(columns={"new_Hn": "Hn", "new_L": "L", "new_b": "b", "new_dx": "dx"})
    return new_network_gdf
    """

def draw_branch_network(hexagons, branches_gdf):
    # possibly, below could also be done with branches_gdf.drop(columns="geometry"),
    # but not sure if it then also drops crs
    branches_properties_df = branches_gdf[["Name", "polygon_ids"]].copy() # , "obs_count", "start_node", "end_node"
    #branches_properties_df = branches_properties_df.set_index("id")
    for index, branch in branches_properties_df.iterrows():
        line_points = []
        for i in range(len(branch["polygon_ids"])):
            hexagon_ref = hexagons.features[branch["polygon_ids"][i]]
            hex_geom = shape(hexagon_ref.geometry)
            mid_point = np.asarray(hex_geom.centroid.xy)
            line_points.append(mid_point)
        if len(line_points) > 1:
            branch_line = LineString(line_points)
            branches_properties_df.loc[index, "geometry"] = branch_line
            branches_properties_df.loc[index, "distance"] = branch_line.length
        else:
            branches_properties_df.loc[index, "geometry"] = np.nan
            branches_properties_df.loc[index, "distance"] = np.nan
    branches_game_gdf = gpd.GeoDataFrame(branches_properties_df, geometry=branches_properties_df["geometry"])
    branches_game_gdf = branches_game_gdf.reset_index()
    branches_game_gdf = branches_game_gdf.rename(columns={"polygon_ids": "hexagon_ids"})
    branches_game_gdf = branches_game_gdf.set_index("index")
    return branches_game_gdf

def process_model_output(model_output_df, scenario="2018"):
    #starttime = time.perf_counter()
    timesteps = list(range(len(model_output_df.iloc[0]["sb_st"])))
    # TODO add proper timesteps
    #timestamp = scenario[:4] + "-08-01"
    #timeseries = pd.to_datetime(pd.Timestamp(timestamp)) + pd.to_timedelta(timesteps, unit='D')
    timeseries = pd.to_datetime(pd.Timestamp('2020-08-01')) + pd.to_timedelta(timesteps, unit='D')
    model_output_df["time"] = [timeseries for i in model_output_df.index]
    #duration = timedelta(seconds=time.perf_counter() - starttime)
    #print('Job took: ', duration)
    columns_to_explode = ["sss", "sb_st", "sn_st", "sp_st", "s_st", "time"]

    # just a quick check for element counts
    if False:
        for i in range(len(model_output_df)):
            for column in columns_to_explode:
                print("element count", column, ":", len(model_output_df.iloc[i][column]))
            print("")
    exploded_output_df = model_output_df.explode(columns_to_explode)

    def add_point_ids(points, name):
        number_of_points = list(range(len(points)))
        branch_name = [name for point in points]
        point_ids = []
        for n in range(len(branch_name)):
            point_ids.append("" + branch_name[n] + "_" + str(number_of_points[n]))
        #print(point_ids)
        return point_ids

    exploded_output_df["id"] = exploded_output_df.apply(lambda row: add_point_ids(row["px"], row.name), axis=1)

    def add_branch_rank(points):
        return list(range(1, len(points) + 1))

    exploded_output_df['branch_rank'] = exploded_output_df.apply(lambda row: add_branch_rank(row["px"]), axis=1)
    #print(exploded_output_df.iloc[-1])
    #print(exploded_output_df.iloc[-1].name)
    if False:
        for values in exploded_output_df.iloc[0]:
            if isinstance(values, np.ndarray):
                print(len(values))

    next_columns_to_explode = ["px", "sb_st", "sn_st", "s_st", "plot xs", "plot ys", "points", "id", "branch_rank"]

    # just a quick check for element counts
    if False:
        for i in range(len(exploded_output_df)):
            print(exploded_output_df.iloc[i].name)
            for column in next_columns_to_explode:
                print("element count", column, ":", len(exploded_output_df.iloc[i][column]))
            print("")

    double_exploded_output_df = exploded_output_df.explode(next_columns_to_explode)
    double_exploded_output_df['sb_mgl'] = double_exploded_output_df['sb_st'] / 1.807 * 1000
    return double_exploded_output_df, exploded_output_df


def output_df_to_gdf(double_exploded_output_df):
    output_points_geometry = gpd.points_from_xy(double_exploded_output_df['plot xs'],
                                                double_exploded_output_df['plot ys'], crs="EPSG:4326")
    network_model_output_gdf = gpd.GeoDataFrame(double_exploded_output_df[['id', 'branch_rank', 'time', 'sb_st', 'sb_mgl']],
                                                geometry=output_points_geometry)
    return network_model_output_gdf

def add_polygon_ids(network_model_output_gdf, polygons):
    output_gdf = network_model_output_gdf.copy()
    def match_points_to_polygon(point, polygon):
        for polygon in polygon.features:
            poly_geom = shape(polygon.geometry)
            if poly_geom.contains(point):
                return polygon.id
        return np.nan

    output_gdf["polygon_id"] = output_gdf.apply(
        lambda row: match_points_to_polygon(row["geometry"], polygons), axis=1)
    output_gdf = output_gdf.dropna()

    def update_branch_ranks(rank, branch, correction):
        branch_correction = correction[branch]
        return rank - branch_correction

    branches = list(set(output_gdf.index))

    ranks_to_update = {}
    for branch in branches:
        rank_value = min(output_gdf.loc[branch]["branch_rank"]) - 1
        ranks_to_update[branch] = rank_value
    #print(ranks_to_update)

    output_gdf["branch_rank"] = output_gdf.apply(
        lambda row: update_branch_ranks(row["branch_rank"], row.name, ranks_to_update), axis=1)
    return output_gdf


def model_output_to_game_locations(game_network_gdf, network_model_output_gdf, exploded_output_df):
    date = exploded_output_df.iloc[0]["time"]
    output_point_location_gdf = network_model_output_gdf.loc[network_model_output_gdf['time'] == date]
    output_locations_count_series = output_point_location_gdf.groupby(level=0).branch_rank.agg('count')

    game_network_gdf = game_network_gdf.reset_index()
    game_network_gdf = game_network_gdf.rename(columns={"index": "Name", "Name": "index"})
    game_network_gdf = game_network_gdf.set_index("index")

    assert len(game_network_gdf) == len(output_locations_count_series)

    game_network_gdf = game_network_gdf.merge(output_locations_count_series.to_frame(), left_index=True,
                                              right_index=True)
    game_network_gdf = game_network_gdf.rename(columns={"branch_rank": "obs_count"})

    def create_game_obs_points(obs_points_model_gdf, branches_game_gdf):
        def update_obs_point_geometry(obs_branch_id, obs_branch_rank, branches):
            branch_dist = branches.loc[obs_branch_id, "distance"]
            branch_obs_count = branches.loc[obs_branch_id, "obs_count"]
            branch_spacing = branch_dist / branch_obs_count
            obs_to_line = -0.5 * branch_spacing + obs_branch_rank * branch_spacing
            point = line_interpolate_point(branches.loc[obs_branch_id, "geometry"], obs_to_line)
            return point

        obs_points_game_df = obs_points_model_gdf.copy()
        obs_points_game_df = obs_points_game_df.drop(columns="geometry")
        obs_points_game_df = obs_points_game_df.reset_index()
        obs_points_game_df["geometry"] = obs_points_game_df.apply(
            lambda row: update_obs_point_geometry(row["index"], row["branch_rank"], branches_game_gdf), axis=1)
        obs_points_game_gdf = gpd.GeoDataFrame(obs_points_game_df, geometry=obs_points_game_df["geometry"])
        obs_points_game_gdf = obs_points_game_gdf.drop(columns="index")
        return obs_points_game_gdf

    game_output_gdf = create_game_obs_points(network_model_output_gdf, game_network_gdf)
    return game_output_gdf

def output_to_timeseries(output_gdf, turn=None, turn_count=None):
    output_gdf["sb_mgl"] = output_gdf["sb_mgl"].astype(float)
    output_gdf = output_gdf.rename(columns={"sb_mgl": "water_salinity"})
    # change bins (and names) below for different salinity concentration categories
    #bins = [0, 0.5, 1.5, 2.5, 10, 30, np.inf]
    # these are categorizations used in the "SWM redeneerlijnen"
    bins = [0, 150, 250, 500, 1000, 1500, 3000, 5000, 10000, 15000, np.inf]
    # TODO test with categorical data
    #names = ['<0.5', '0.5-1.5', '1.5-2.5', '2.5-10', '10-30', '30+']
    values = [i + 1 for i in range(len(bins) - 1)]
    #values = [0.25, 1, 2, 6.25, 20, 34]
    output_gdf['salinity_category'] = pd.cut(output_gdf['water_salinity'], bins, labels=values)
    if turn is not None:
        output_gdf["turn"] = turn
    if turn_count is not None:
        output_gdf["run"] = turn_count
    return output_gdf

def update_split_channel_ids(output_df, update_dict):
    for key, values in update_dict.items():
        split_1_df = output_df.loc[values[0]].copy()
        split_2_df = output_df.loc[values[1]].copy()
        max_rank_1 = max(split_1_df["branch_rank"].tolist())
        max_rank_2 = max(split_2_df["branch_rank"].tolist())
        """
        it seems the output is exactly mirrored from what is expected, so the code below is not as neat. Once fixed,
        the code below can be used instead.
        
        output_df.loc[values[0], "id"] = split_1_df.apply(
            lambda row: key + "_" + str(row["branch_rank"] - 1), axis=1)
        output_df.loc[values[1], "id"] = split_2_df.apply(
            lambda row: key + "_" + str(row["branch_rank"] + max_rank - 1), axis=1)
        """
        output_df.loc[values[0], "id"] = split_1_df.apply(
            lambda row: key + "_" + str(max_rank_1 + max_rank_2 - row["branch_rank"]), axis=1)
        output_df.loc[values[1], "id"] = split_2_df.apply(
            lambda row: key + "_" + str(max_rank_2 - row["branch_rank"]), axis=1)
    return output_df