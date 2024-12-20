# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: haanrj
"""


def compare_hex(hexagons_new, hexagons_old):
    """
    compares the current state (red and blue markers) of each hexagon tile to the previous state.
    """
    for feature in hexagons_new.features:
        reference_hex = hexagons_old[feature.id]
        if feature.properties["red_markers"] != reference_hex.properties["red_markers"]:
            print("Hexagon", str(feature.id), "red markers changed from", reference_hex.properties["red_markers"],
                  "to", feature.properties["red_markers"])
            feature.properties["red_changed"] = True
        else:
            feature.properties["red_changed"] = False
        if (feature.properties["blue_markers"] != reference_hex.properties["blue_markers"]):
            print("Hexagon", str(feature.id), "blue markers changed from", reference_hex.properties["blue_markers"],
                  "to", feature.properties["blue_markers"])
            feature.properties["blue_changed"] = True
        else:
            feature.properties["blue_changed"] = False
    return hexagons_new