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

def merge_model_output(simulations, model_output_df1, model_output_df2=None, model_output_df3=None):
    output_dfs = [model_output_df1, model_output_df2, model_output_df3]
    if model_output_df3 is None:
        del output_dfs[-1]
    if model_output_df2 is None:
        del output_dfs[-1]

    columns_to_merge = ["sss", "sb_st", "sn_st", "sp_st", "s_st", "htot", "utot"]
    next_columns_to_merge = ["sb_st", "sn_st", "s_st", "htot", "utot"]
    columns_to_keep = ["px", "plot xs", "plot ys", "points"]
    columns_to_explode = []
    next_columns_to_explode = []

    def flatten(xss):
        return [x for xs in xss for x in xs]

    for i in range(len(output_dfs)):
        #output_dfs[i] = output_dfs[i].set_index("Unnamed: 0") # this should not be needed in the game code
        output_dfs[i] = output_dfs[i][flatten([columns_to_merge, columns_to_keep])]
        if False:
            for j in range(len(output_dfs[i])):
                for column in columns_to_merge:
                    print("element count", column, ":", len(output_dfs[i].iloc[j][column]))
                print("")
        columns_to_rename = {}
        next_columns_to_rename = {}
        for column_name in columns_to_merge:
            new_column_name = column_name + "_" + simulations[i]
            columns_to_rename[column_name] = new_column_name
        for column_name in next_columns_to_merge:
            new_column_name = column_name + "_" + simulations[i]
            next_columns_to_rename[column_name] = new_column_name
        output_dfs[i] = output_dfs[i].rename(columns=columns_to_rename)
        columns_to_explode.append(list(columns_to_rename.values()))
        next_columns_to_explode.append(list(next_columns_to_rename.values()))
        if i != 0:
            output_dfs[i] = output_dfs[i].drop(columns=columns_to_keep)

    merged_model_output_df = output_dfs[0].copy()
    for i in range(1, len(output_dfs)):
        merged_model_output_df = pd.merge(merged_model_output_df, output_dfs[i].copy(), left_index=True, right_index=True)
    columns_to_explode = flatten(columns_to_explode)
    next_columns_to_explode.append(columns_to_keep)
    next_columns_to_explode = flatten(next_columns_to_explode)
    return merged_model_output_df, columns_to_explode, next_columns_to_explode

def process_model_output(model_output_df, columns_to_explode, next_columns_to_explode, sim_count, simulations,
                         scenario="2018"):
    timesteps = list(range(len(model_output_df.iloc[0]["sb_st_drought"])))
    # TODO add proper timesteps
    #timestamp = scenario[:4] + "-08-01"
    #timeseries = pd.to_datetime(pd.Timestamp(timestamp)) + pd.to_timedelta(timesteps, unit='D')
    if scenario == "reference":
        year = "2018"
    else:
        year = scenario[:4]
    key = "time"
    timeseries = pd.to_datetime(pd.Timestamp(year + '-08-01')) + pd.to_timedelta(timesteps, unit='D')
    model_output_df[key] = [timeseries for i in model_output_df.index]

    if key not in columns_to_explode:
        columns_to_explode.append(key)

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
        return point_ids

    key = "id"
    exploded_output_df[key] = exploded_output_df.apply(lambda row: add_point_ids(row["px"], row.name), axis=1)
    if key not in next_columns_to_explode:
        next_columns_to_explode.append(key)

    def add_branch_rank(points):
        return list(range(1, len(points) + 1))

    key = "branch_rank"
    exploded_output_df[key] = exploded_output_df.apply(lambda row: add_branch_rank(row["px"]), axis=1)
    if key not in next_columns_to_explode:
        next_columns_to_explode.append(key)

    if False:
        for values in exploded_output_df.iloc[0]:
            if isinstance(values, np.ndarray):
                print(len(values))

    # just a quick check for element counts
    if False:
        print_names = ["Breeddiep", "Maas", "Waal"]
        for i in range(len(exploded_output_df)):
            if exploded_output_df.iloc[i].name in print_names:
                print(exploded_output_df.iloc[i].name)
                for column in next_columns_to_explode:
                    print("element count", column, ":", len(exploded_output_df.iloc[i][column]))
                print("")

    double_exploded_output_df = exploded_output_df.explode(next_columns_to_explode)
    for i in range(sim_count):
        key = 'sb_st_' + simulations[i]
        new_key = 'water_salinity_' + simulations[i]
        double_exploded_output_df[new_key] = double_exploded_output_df[key] / 1.807 * 1000
        double_exploded_output_df[new_key] = double_exploded_output_df[new_key].astype(float)
    return double_exploded_output_df, exploded_output_df

def output_df_to_gdf(double_exploded_output_df):
    possible_columns = ['id', 'branch_rank', 'time', 'sb_st_drought', 'water_salinity_drought', 'htot_drought',
                        'sb_st_normal', 'water_salinity_normal', 'htot_normal', 'sb_st_average',
                        'water_salinity_average', 'htot_average']
    columns_to_keep = []
    for column in possible_columns:
        if column in double_exploded_output_df.columns.values:
            columns_to_keep.append(column)
    output_points_geometry = gpd.points_from_xy(double_exploded_output_df['plot xs'],
                                                double_exploded_output_df['plot ys'], crs="EPSG:4326")
    network_model_output_gdf = gpd.GeoDataFrame(
        double_exploded_output_df[columns_to_keep],
        geometry=output_points_geometry)
    return network_model_output_gdf

def add_polygon_ids(network_model_output_gdf, polygons):
    output_gdf = network_model_output_gdf.copy()
    output_slice_gdf = output_gdf.copy()
    output_slice_gdf = output_slice_gdf[output_slice_gdf["time"] == output_slice_gdf.iloc[0]["time"]]
    def match_points_to_polygon(point, polygon):
        for polygon in polygon.features:
            poly_geom = shape(polygon.geometry)
            if poly_geom.contains(point):
                return polygon.id
        return np.nan

    output_slice_gdf["polygon_id"] = output_slice_gdf.apply(
        lambda row: match_points_to_polygon(row["geometry"], polygons), axis=1)
    output_slice_gdf = output_slice_gdf.dropna()
    output_slice_gdf["polygon_id"] = output_slice_gdf["polygon_id"].astype(int)
    output_slice_gdf = output_slice_gdf[["id", "polygon_id"]]
    output_gdf = output_gdf.reset_index().merge(output_slice_gdf, on="id").set_index("index")

    def update_branch_ranks(rank, branch, correction):
        branch_correction = correction[branch]
        return rank - branch_correction

    branches = list(set(output_gdf.index))

    ranks_to_update = {}
    for branch in branches:
        rank_value = min(output_gdf.loc[branch]["branch_rank"]) - 1
        ranks_to_update[branch] = rank_value

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

def output_to_timeseries(output_gdf, sim_count, simulations, turn=None, turn_count=None):
    # these are categorizations used in the "SWM redeneerlijnen"
    # TODO check outcome
    bins = [0, 150, 250, 500, 1000, 1500, 3000, 5000, 10000, 15000, np.inf]
    # TODO test with categorical data
    #names = ['<0.5', '0.5-1.5', '1.5-2.5', '2.5-10', '10-30', '30+']
    values = [i + 1 for i in range(len(bins) - 1)]
    for i in range(sim_count):
        key = 'water_salinity_' + simulations[i]
        new_key = 'salinity_category_' + simulations[i]
        output_gdf[new_key] = pd.cut(output_gdf[key], bins, labels=values)
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