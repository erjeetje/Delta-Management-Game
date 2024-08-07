import os
import re
import json
import geojson
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
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
                                    print("branch crossing between", polygon.id, "and", neighbour_polygon.id)
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
    branches_properties_df = branches_gdf[["Name", "polygon_ids"]] # , "obs_count", "start_node", "end_node"
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

def process_model_output(model_output_df):
    if False:
        row=3
        print("number of points:", len(model_output_df.iloc[row]["px"]))
        salinity_values = model_output_df.iloc[row]["sb_st"]
        print("timesteps?", len(salinity_values))
        print("salinity values matching points?", len(salinity_values[0]))
        for values in model_output_df.iloc[0]:
            print(len(values))
    if True:
        print(type(model_output_df.iloc[0]["sb_st"]))
    timesteps = list(range(len(model_output_df.iloc[0]["sb_st"])))
    timeseries = pd.to_datetime(pd.Timestamp('2020-06-01')) + pd.to_timedelta(timesteps, unit='D')
    model_output_df["time"] = [timeseries for i in model_output_df.index]
    columns_to_explode = ["sss", "sb_st", "sn_st", "sp_st", "s_st", "time"]

    # just a quick check for element counts
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

    exploded_output_df["id"] = exploded_output_df.apply(lambda row: add_point_ids(row["px"], row.name), axis=1)

    def add_branch_rank(points):
        return list(range(1, len(points) + 1))

    exploded_output_df['branch_rank'] = exploded_output_df.apply(lambda row: add_branch_rank(row["px"]), axis=1)
    for values in exploded_output_df.iloc[0]:
        if isinstance(values, np.ndarray):
            print(len(values))

    next_columns_to_explode = ["px", "sb_st", "sn_st", "s_st", "plot xs", "plot ys", "points", "id", "branch_rank"]

    # just a quick check for element counts
    for i in range(len(exploded_output_df)):
        print(exploded_output_df.iloc[i].name)
        for column in next_columns_to_explode:
            print("element count", column, ":", len(exploded_output_df.iloc[i][column]))
        print("")

    double_exploded_output_df = exploded_output_df.explode(next_columns_to_explode)
    output_points_geometry = gpd.points_from_xy(double_exploded_output_df['plot xs'],
                                                double_exploded_output_df['plot ys'], crs="EPSG:4326")
    network_model_output_gdf = gpd.GeoDataFrame(double_exploded_output_df[['id', 'branch_rank', 'time', 'sb_st']],
                                                geometry=output_points_geometry)
    return network_model_output_gdf, exploded_output_df

def add_polygon_ids(network_model_output_gdf, polygons):
    def match_points_to_polygon(point, polygon):
        for polygon in polygon.features:
            poly_geom = shape(polygon.geometry)
            if poly_geom.contains(point):
                return polygon.id
        return np.nan

    network_model_output_gdf["polygon_id"] = network_model_output_gdf.apply(
        lambda row: match_points_to_polygon(row["geometry"], polygons), axis=1)
    network_model_output_gdf = network_model_output_gdf.dropna()

    def update_branch_ranks(rank, branch, correction):
        branch_correction = correction[branch]
        return rank - branch_correction

    branches = list(set(network_model_output_gdf.index))

    ranks_to_update = {}
    for branch in branches:
        rank_value = min(network_model_output_gdf.loc[branch]["branch_rank"]) - 1
        ranks_to_update[branch] = rank_value
    print(ranks_to_update)

    network_model_output_gdf["branch_rank"] = network_model_output_gdf.apply(
        lambda row: update_branch_ranks(row["branch_rank"], row.name, ranks_to_update), axis=1)
    return network_model_output_gdf


def model_output_to_game_locations(game_network_gdf, network_model_output_gdf, exploded_output_df):
    date = exploded_output_df.iloc[0]["time"]
    output_point_location_gdf = network_model_output_gdf.loc[network_model_output_gdf['time'] == date]
    output_locations_count_series = output_point_location_gdf.groupby(level=0).branch_rank.agg('count')

    game_network_gdf = game_network_gdf.reset_index()
    game_network_gdf = game_network_gdf.rename(columns={"index": "Name", "Name": "index"})
    game_network_gdf = game_network_gdf.set_index("index")

    print(len(game_network_gdf))
    print(len(output_locations_count_series))

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

        obs_points_game_df = obs_points_model_gdf.drop(columns="geometry")
        obs_points_game_df = obs_points_game_df.reset_index()
        obs_points_game_df["geometry"] = obs_points_game_df.apply(
            lambda row: update_obs_point_geometry(row["index"], row["branch_rank"], branches_game_gdf), axis=1)
        obs_points_game_gdf = gpd.GeoDataFrame(obs_points_game_df, geometry=obs_points_game_df["geometry"])
        return obs_points_game_gdf

    game_output_gdf = create_game_obs_points(network_model_output_gdf, game_network_gdf)
    return game_output_gdf

def output_to_timeseries(output_gdf, scenario=None):
    timeseries_gdf = output_gdf.reset_index()
    timeseries_gdf["id"] = timeseries_gdf["index"] + "_" + timeseries_gdf[
        "branch_rank"].astype(str)
    timeseries_gdf["sb_st"] = timeseries_gdf["sb_st"].astype(float)
    #timeseries_gdf["water_salinity"] = timeseries_gdf["sb_st"]
    timeseries_gdf = timeseries_gdf.rename(columns={"sb_st": "water_salinity"})
    if scenario is not None:
        timeseries_gdf["scenario"] = scenario
    return timeseries_gdf
