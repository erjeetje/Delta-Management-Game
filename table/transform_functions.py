# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: haanrj
"""

import json
import geojson
import numpy as np
from os.path import join
from cv2 import getPerspectiveTransform, perspectiveTransform


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def create_calibration_file(img_x=None, img_y=None, path="", debug=False, save=False):
    """
    Function that creates the calibration file (json format) and returns the
    transforms that can be used by other functions.
    """
    def compute_transforms(calibration):
        """compute transformation matrices based on calibration data"""

        point_names = [
            "model",
            "img",
            "img_flipped",
            "beamer",
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
                transform = getPerspectiveTransform(
                    point_arrays[a],
                    point_arrays[b]
                )
                transforms[transform_name] = transform

        return transforms

    calibration = {}
    # model points following SandBox implementation; between [-600, -400] and [600, 400]
    calibration['model'] = (
        [-400, 300 ], [400, 300], [400, -300], [-400, -300])
    # corners of image after image cut
    calibration['img'] = (
        [0, 0], [img_x, 0], [img_x, img_y],  [0, img_y])
    calibration['img_flipped'] = (
        [0, img_y], [img_x, img_y], [img_x, 0], [0, 0])
    # beamer resolution
    calibration['beamer'] = [0, 0], [640, 0], [640, 480], [0, 480]
    transforms = compute_transforms(calibration)
    calibration.update(transforms)
    if debug:
        with open(join(path, 'calibration.json'), 'w') as f:
            json.dump(calibration, f, sort_keys=True, indent=2,
                      cls=NumpyEncoder)
    return transforms


def transform_features(features, transforms, export=None):
    """
    Function that transforms geojson files to new coordinates based on where
    the geojson needs to be transformed to (e.g. from the image processed to
    the model: 'img_post_cut2model').
    """

    def execute_transform(x, y, M):
        """perspective transform x,y with M"""
        xy_t = np.squeeze(
            perspectiveTransform(
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
    elif export == "img_flip":
        transform = transforms['img2img_flipped']
    elif export == "img_beamer":
        transform = transforms['img2img_beamer']
    else:
        print("unknown export method, current supported are: 'model', 'img_flip' & 'img_beamer'")
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
    transformed_features = geojson.FeatureCollection(transformed_features)
    return transformed_features