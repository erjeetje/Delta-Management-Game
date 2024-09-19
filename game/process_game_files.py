import os
import re
import json
import geojson
import numpy as np
from shapely import geometry
from scipy.spatial import cKDTree
from copy import deepcopy


def read_geojson(filename='hexagons_warped.geojson', path=""):
    """
    function that loads and returns the hexagons. Currently not called in
    the main script as the hexagons are stored internally.
    """
    if path == "":
        raise FileNotFoundError
    with open(os.path.join(path, filename)) as f:
        features = geojson.load(f)
    return features


def save_geojson(features, filename='hexagons.geojson', path=""):
    with open(os.path.join(path, filename), 'w') as f:
        geojson.dump(features, f, sort_keys=True, indent=2)




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

def find_neighbours(hexagons):
    hex_coor = []
    polygons = []
    hexagon0_y = 0
    hexagon1_y = 0
    hexagon_count = 0
    for feature in hexagons.features:
        if not feature.properties["ghost_hexagon"]:
            hexagon_count += 1
        shape = geometry.shape(feature.geometry)
        x_hex = shape.centroid.x
        y_hex = shape.centroid.y
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
        shape = geometry.shape(feature.geometry)
        x_hex = shape.centroid.x
        y_hex = shape.centroid.y
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
            edge = geometry.LineString([(x1 ,y1), (x2, y2)])
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


def add_geometry_dimension(hexagons):
    for feature in hexagons.features:
        check = isinstance(feature['geometry']['coordinates'][0][0], list)
        if not check:
            feature['geometry']['coordinates'] = [feature['geometry']['coordinates']]
    return hexagons


def read_json_features(filename="hexagon_shapes_warped.json", path=""):
    with open(os.path.join(path, filename)) as f:
        shapes = json.load(f)

    layers = shapes["layers"]
    new_shapes = []

    for layer in layers:
        polygon_id = layer["name"]
        polygon_id = int(re.sub('[^0-9]', '', polygon_id)) - 1
        polygon_geometry = layer["paths"][0]["points"]
        polygon_geometry.append(polygon_geometry[0])
        polygon = geojson.Polygon(polygon_geometry)
        feature = geojson.Feature(id=polygon_id, geometry=polygon)
        new_shapes.append(feature)

    new_shapes.reverse()
    warped_hexagons = geojson.FeatureCollection(new_shapes)
    warped_hexagons = add_geometry_dimension(warped_hexagons)
    return warped_hexagons

def index_points_to_polygons(polygons, points):
    def match_points_to_polygon(point, polygons):
        for polygon in polygons.features:
            shape = geometry.shape(polygon.geometry)
            if shape.contains(point):
                return polygon.id
        return np.nan
    points["polygon_id"] = points.apply(
        lambda row: match_points_to_polygon(row["geometry"], polygons), axis=1)
    points = points.dropna()

    points['obs_count'] = points.groupby('branch_id_game')['branch_id_game'].transform('count')
    points['branch_rank'] = points.groupby('branch_id_game')['chainage_game'].rank(ascending=True)
    points[['polygon_id', 'obs_count', 'branch_rank']] = points[['polygon_id', 'obs_count', 'branch_rank']].astype('int')

    points_grouped = points.groupby("polygon_id")
    for polygon in polygons.features:
        polygon.properties["obs_ids"] = []
        try:
            polygon.properties["obs_ids"] = points_grouped.get_group(polygon.id)["obs_id"].to_list()
        except KeyError:
            continue
    return polygons, points.dropna()


"""

Code below is a temporary store

    points["hexagon_id"] = np.nan
    for feature in polygons.features:
        shape = geometry.shape(feature.geometry)
        feature.properties["obs_ids"] = []
        #code below is for observation points as geojsons
        for point in points.features:
            try:
                #point = geometry.shape(obs_point.geometry) # this would make more sense, but need to adjust obs points script.
                point_coor = geometry.Point(point.properties["x"], point.properties["y"])
            except TypeError:
                #print(obs_point.properties["id"], obs_point.geometry)
                pass
            if shape.contains(point_coor):
                #print(obs_point.properties["id"], "is in hexagon", feature.id)
                feature.properties["obs_ids"].append(point.properties["id"])
                point.properties["hexagon_id"] = feature.id
        """

"""
code to fix later to not use iterrows by using something like:

pnts = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})

see: https://stackoverflow.com/questions/48097742/geopandas-point-in-polygon
for index, point in points.iterrows():
    if shape.contains(point["geometry"]):
        feature.properties["obs_ids"].append(point["obs_id"])
        points.loc[index, "hexagon_id"] = feature.id
print(polygons.features[5])
"""

def get_subbranches():
    subbranches_dict = {"Maasmond": ["Maasmond"],
                        "Nieuwe_Waterweg": ["NieuweWaterweg3", "NieuweWaterweg2", "NieuweWaterweg1"],
                        "Nieuwe_Maas_West": ["NieuweMaas11", "NieuweMaas10", "NieuweMaas09", "NieuweMaas08", "NieuweMaas07",
                                             "NieuweMaas06", "NieuweMaas05", "NieuweMaas04", "NieuweMaas03"],
                        "Nieuwe_Maas_Oost": ["NieuweMaas02", "NieuweMaas01"],
                        "Lek": ["Lek9", "Lek8", "Lek7", "Lek6", "Lek5", "Lek4",
                                "Lek3", "Lek2", "Lek1_B"],
                        "Hollandse_IJssel": ["HollandseIJssel2", "HollandseIJssel1"],
                        "Hartelkanaal": ["Calandkanaal4", "Beerkanaal3", "Beerkanaal2", "Beerkanaal1", "Hartelkanaal5",
                                         "Hartelkanaal4", "Hartelkanaal3", "Hartelkanaal2", "Hartelkanaal1"],
                        "Oude_Maas_Noord": ["OudeMaas5"],
                        "Oude_Maas_West": ["OudeMaas4"],
                        "Oude_Maas_Midden": ["OudeMaas3"],
                        "Oude_Maas_Oost": ["OudeMaas2", "OudeMaas1"],
                        "Dordtsche_Kil": ["DordtscheKil"],
                        "Noord": ["Noord3", "Noord2", "Noord1"],
                        "Spui": ["Beningen", "Spui"],
                        "Voordelta_Zuid": ["VD-Slijkgat", "VD-NoordPampus", "VD-Buitenhaven"],
                        "Voordelta_Noord": ["VD-Bokkegat", "VD-Scheelhoek2", "VD-Scheelhoek1"],
                        "Haringvliet_West": ["Haringvliet7", "Haringvliet6-AG"],
                        "Haringvliet_Hollands_Diep": ["Haringvliet5", "Haringvliet4",
                                                      "Haringvliet3", "Haringvliet2", "Haringvliet1", "HollandsDiep5",
                                                      "HollandsDiep4", "HollandsDiep3"],
                        "Hollands_Diep_Oost": ["HollandsDiep2", "HollandsDiep1"],
                        "Nieuwe_Merwede": ["NieuweMerwede13", "NieuweMerwede12", "NieuweMerwede11", "NieuweMerwede10",
                                           "NieuweMerwede09", "NieuweMerwede08", "NieuweMerwede07", "NieuweMerwede06",
                                           "NieuweMerwede05", "NieuweMerwede04", "NieuweMerwede03", "NieuweMerwede02",
                                           "NieuweMerwede01"],
                        "Beneden_Merwede": ["BenedenMerwede5", "BenedenMerwede4", "BenedenMerwede3",
                                            "BenedenMerwede2", "BenedenMerwede1"],
                        "Waal": ["Waal6", "Waal5", "Waal4"],
                        "Amer": ["Amer5", "Amer4", "Amer3", "Amer2", "Amer1"]}
    return subbranches_dict