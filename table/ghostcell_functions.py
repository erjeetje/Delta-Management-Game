# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: HaanRJ
"""

import geojson


def set_values(hexagons):
    """
    This is a function used in the Virtual River Game to add "ghost" hexagons, adjecent to the game board but out of
    control of the players (they simply exist).

    It is not updated as a standard function for the game table, but there to adapt if desired.
    """
    dike_values = {"z_reference": 4,
                   "z": 16,
                   "landuse": 10,
                   "water": False,
                   "land": True,
                   "behind_dike": False
                   }
    floodplain_values_grass = {"z_reference": 2,
                               "z": 8,
                               "landuse": 2,
                               "water": False,
                               "land": True,
                               "behind_dike": False
                               }
    floodplain_values_forest = {"z_reference": 2,
                                "z": 8,
                                "landuse": 5,
                                "water": False,
                                "land": True,
                                "behind_dike": False
                                }
    floodplain_values_reed = {"z_reference": 2,
                              "z": 8,
                              "landuse": 3,
                              "water": False,
                              "land": True,
                              "behind_dike": False
                              }
    channel_values = {"z_reference": 0,
                      "z": 0,
                      "landuse": 9,
                      "water": True,
                      "land": False,
                      "behind_dike": False
                      }
    behind_dike_values = {"z_reference": 2,
                          "z": 8,
                          "landuse": 1,
                          "water": False,
                          "land": True,
                          "behind_dike": True
                          }
    channel_hexagons = [147, 148, 157, 158, 167, 168, 176, 177,
                        184, 185, 193, 194, 203, 204, 213, 214]
    dike_hexagons = [143, 152, 153, 161, 162, 171, 172, 180,
                     181, 189, 191, 198, 201, 207, 211, 216]
    floodplain_forest = [144, 151, 154, 160, 163, 170, 173,
                         182, 188, 192, 197, 206]
    floodplain_reed = [149, 156, 166, 195, 202, 215]
    behind_dike = [190, 199, 200, 208, 209, 210, 217, 218]
    for feature in hexagons.features:
        if not feature.properties["ghost_hexagon"]:
            continue
        else:
            if feature.id in dike_hexagons:
                values = dike_values
            elif feature.id in channel_hexagons:
                values = channel_values
            elif feature.id in floodplain_forest:
                values = floodplain_values_forest
            elif feature.id in floodplain_reed:
                values = floodplain_values_reed
            elif feature.id in behind_dike:
                values = behind_dike_values
            else:
                values = floodplain_values_grass
            feature.properties["z_reference"] = values["z_reference"]
            feature.properties["z"] = values["z"]
            feature.properties["landuse"] = values["landuse"]
            feature.properties["water"] = values["water"]
            feature.properties["land"] = values["land"]
            feature.properties["behind_dike"] = values["behind_dike"]
    if False:
        with open('ghost_cells_set_test.geojson', 'w') as f:
            geojson.dump(hexagons, f, sort_keys=True, indent=2)
    return hexagons


def update_values(hexagons):
    for feature in hexagons.features:
        if not feature.properties["ghost_hexagon"]:
            continue
        else:
            feature.properties["red_changed"] = False
            feature.properties["blue_changed"] = False
    return hexagons
    