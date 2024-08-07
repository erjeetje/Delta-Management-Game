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


def process_model_output(model_network_df):
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
        if branch["Name"] == "Hollands Diep 1":
            print(line_points)
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