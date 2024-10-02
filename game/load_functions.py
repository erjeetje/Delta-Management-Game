import os
import re
import json
import geojson
#import geopandas as gpd
from PyQt5.QtGui import QPixmap
#from matplotlib.colors import LogNorm, Normalize, CenteredNorm


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

    def add_geometry_dimension(hexagons):
        for feature in hexagons.features:
            check = isinstance(feature['geometry']['coordinates'][0][0], list)
            if not check:
                feature['geometry']['coordinates'] = [feature['geometry']['coordinates']]
        return hexagons

    warped_hexagons = add_geometry_dimension(warped_hexagons)
    return warped_hexagons

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

def load_images():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    colorbar_location = os.path.join(dir_path, "input_files")
    colorbar_salinity_file = os.path.join(colorbar_location, "salinity_colorbar_horizontal_small.png")
    colorbar_salinity = QPixmap(colorbar_salinity_file)
    labels_salinity_categories_file = os.path.join(colorbar_location, "salinity_categorized_labels_horizontal_small.png")
    labels_salinity_categories = QPixmap(labels_salinity_categories_file)
    #basemap_image_file = os.path.join(colorbar_location, "basemap.png")
    #basemap_image = plt.imread(basemap_image_file)
    return colorbar_salinity, labels_salinity_categories