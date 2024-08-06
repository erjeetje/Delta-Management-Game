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