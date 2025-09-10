import os
import geojson
import cv2
import json
import numpy as np


def create_calibration_file(polygons, save=False, path=""):
    """
    Function that creates the calibration file (json format) and returns the
    transforms that can be used by other functions.

    - wereld coordinaten (in m)
    - board (in cm / m)
    - beamer (resolutie)
    """

    def compute_transforms(calibration):
        """compute transformation matrices based on calibration data"""

        point_names = [
            "model",
            "img",
            "img_flipped",
            "polygons",
            "beamer",
            "tygron"
        ]

        point_arrays = {}
        for name in point_names:
            if name in calibration:
                arr = np.array(calibration[name], dtype='float32')
            elif name + "_points" in calibration:
                arr = np.array(calibration[name + "_points"], dtype='float32')
            else:
                continue
            point_arrays[name] = arr

        transforms = {}
        for a in point_names:
            for b in point_names:
                if a == b:
                    continue
                if not (a in point_arrays):
                    continue
                if not (b in point_arrays):
                    continue
                transform_name = a + '2' + b
                transform = cv2.getPerspectiveTransform(
                    point_arrays[a],
                    point_arrays[b]
                )
                transforms[transform_name] = transform
        return transforms

    img_y = 1000
    img_x = int(round(img_y * 1.3861874976470018770202169598726))

    calibration = {}
    """
    # EPSG:3897 - note: variables out of date
    world_top_left = [426804, 6813127]
    world_top_right = [530228, 6811988]
    world_bottom_left = [426804, 6726933]
    world_bottom_right = [530228, 6726933]
    """
    """
    # EPSG:28992 - note: variables out of date
    world_top_left = [48518, 454868]
    world_top_right = [112205, 453213]
    world_bottom_left = [47381, 401643]
    world_bottom_right = [111205, 400643]
    """
    # EPSG:4326
    world_top_left = [3.834043, 52.0718455]
    world_top_right = [4.7631053, 52.0655544]
    world_bottom_left = [3.8340423, 51.5933656]
    world_bottom_right = [4.7551842, 51.5929827]

    # TODO switch to these coordinates with the new polygon mapping
    #world_top_left = [3.84, 52.075]  # [3.834043, 52.0718455]
    #world_top_right = [4.82, 52.07]  # [4.7631053, 52.0655544]
    #world_bottom_left = [3.84, 51.65]  # [3.8340423, 51.5933656]
    #world_bottom_right = [4.82, 51.645]  # [4.7551842, 51.5929827]

    def get_bbox(polygons):
        x_coor = []
        y_coor = []
        for feature in polygons.features:
            for point in feature['geometry']['coordinates'][0]:
                x_coor.append(point[0])
                y_coor.append(point[1])
        return [min(x_coor), max(x_coor), min(y_coor), max(y_coor)]

    x_min, x_max, y_min, y_max = get_bbox(polygons)

    # calibration['model'] = [62259, 448539], [113255, 448539], [113255, 406910], [62259, 406910]
    calibration['model'] = world_top_right, world_top_left, world_bottom_left, world_bottom_right
    # resolution camera; pixels of recut image of calibration points
    calibration['img'] = [0, 0], [img_x, 0], [img_x, img_y], [0, img_y]
    # resolution camera; pixels of recut image of calibration points - y flipped
    calibration['img_flipped'] = [0, img_y], [img_x, img_y], [img_x, 0], [0, 0]
    # warped hexagon features, illustrator file (pixels)
    calibration['polygons'] = [x_max, y_min], [x_min, y_min], [x_min, y_max], [x_max, y_max]
    # resolution beamer; based on current VRG implementation
    calibration['beamer'] = [0, 0], [600, 0], [600, 450], [0, 450]
    # resolution tygron; 1000 x 750 meters in Tygron world
    calibration['tygron'] = [0, 0], [1000, 0], [1000, -750], [0, -750]
    transforms = compute_transforms(calibration)
    calibration.update(transforms)
    if save:
        with open(os.path.join(path, 'calibration.json'), 'w') as f:
            json.dump(calibration, f, sort_keys=True, indent=2,
                      cls=NumpyEncoder)
    return transforms


def transform(features, transforms, export=None, path=""):
    """
    Function that transforms geojson files to new coordinates based on where
    the geojson needs to be transformed to (e.g. from the image processed to
    the model: 'img_post_cut2model').
    """

    def execute_transform(x, y, M):
        """perspective transform x,y with M"""
        xy_t = np.squeeze(
            cv2.perspectiveTransform(
                np.dstack(
                    [
                        x,
                        y
                    ]
                ),
                np.asarray(M)
            )
        )
        return xy_t[:, 0], xy_t[:, 1]

    transformed_features = []
    # get correct transform.
    if export == "model":
        transform = transforms['img2model']
    elif export == "beamer":
        transform = transforms['img2beamer']
    elif export == "tygron":
        transform = transforms['img2tygron']
    elif export == "warped":
        transform = transforms['polygons2model']
    else:
        print("unknown export method")
        return features
    # transform each feature to new coordinates.
    for feature in features.features:
        pts = np.array(feature.geometry["coordinates"][0], dtype="float32")
        # points should be channels.
        x, y = pts[:, 0], pts[:, 1]
        x_t, y_t = execute_transform(x, y, transform)
        xy_t = np.c_[x_t, y_t]
        new_feature = geojson.Feature(id=feature.id,
                                      geometry=geojson.Polygon([xy_t.tolist()]),
                                      properties=feature.properties)
        transformed_features.append(new_feature)
    # different export handlers.
    if export == "warped":
        crs = {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::28992"  # 3857
            }
        }
        transformed_features = geojson.FeatureCollection(transformed_features,
                                                         crs=crs)
    else:
        transformed_features = geojson.FeatureCollection(transformed_features)

    return transformed_features


def get_bbox(hexagons):
    x_coor = []
    y_coor = []
    for feature in hexagons.features:
        for point in feature['geometry']['coordinates'][0]:
            x_coor.append(point[0])
            y_coor.append(point[1])
    return [min(x_coor), max(x_coor), min(y_coor), max(y_coor)]