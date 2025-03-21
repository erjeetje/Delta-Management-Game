# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: haanrj
"""

import cv2
import numpy as np
import geojson
from os.path import join


def detect_corners(img, method='standard', debug=False, path=""):
    """
    Function that detects the corners of the board (the four white circles)
    and returns their coordinates as a 2D array.
    """
    try:
        height, width, channels = img.shape
    except AttributeError:
        print("There appears to be no webcam connected to the system, "
              "entering test mode")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    if method == 'adaptive':
        blur = cv2.medianBlur(gray, 5)  # blur grayscale image
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        # store threshold image
        if debug:
            cv2.imwrite(join(path, 'Adaptive_threshold.jpg'), thresh)
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # blur grayscale image
        # threshold grayscale image as binary
        ret, thresh = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # store threshold image
        if debug:
            cv2.imwrite(join(path, 'Standard_threshold.jpg'), thresh)

    # create mask to only search for circles in the corner since we know that's
    # where the circles are. The code is rather sensitive and sometimes it sees
    # incorrect circles. This section prevents that from happening. Even when
    # it does however, it still detects the real circles more clearly, thus
    # lists them in index 0-3.
    mask = np.zeros((height, width), dtype="uint8") 
    margin_x = round(width * 0.2)
    margin_y = round(height * 0.14)
    cv2.rectangle(mask, (0, 0), (margin_x, margin_y), (255, 255, 255), -1)
    cv2.rectangle(mask, (width-margin_x, 0), (width, margin_y),
                  (255, 255, 255), -1)
    cv2.rectangle(mask, (0, height-margin_y), (margin_x, height),
                  (255, 255, 255), -1)
    cv2.rectangle(mask, (width-margin_x, height-margin_y), (width, height),
                  (255, 255, 255), -1)
    masked_tresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # detect corner circles in the image (min/max radius ensures only
    # finding those we want)
    circles = cv2.HoughCircles(masked_tresh, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=14, minRadius=18,
                               maxRadius=21)

    # ensure at least some circles were found, such falesafes (also for certain
    # error types) should be build in in later versions --> this should have
    # the effect that the script either aborts or goes to test mode.
    if circles is None:
        print('ERROR: No circles were detected in the image')
        return
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    canvas = circles[:, :2]

    # this drawing of the circles is not necessary for the program to run. Left
    # in as it only happens in the initialization.
    for (x, y, r) in circles:
        # draw circle around detected corner
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        # draw rectangle at center of detected corner
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # store the corner detection image
    if debug:
        cv2.imwrite(join(path, 'CornersDetected.jpg'), img)
    return canvas


def rotate_grid(canvas):
    """
    Function that sorts the four corners in the right order (top left, top
    right, bottom right, bottom left) and returns the perspective transform
    to be used throughout the session.
    """
    # get index of one of the two top corners, store it and delete from array
    lowest_y = int(np.argmin(canvas, axis=0)[1:])
    top_corner1 = canvas[lowest_y]
    x1 = top_corner1[0]
    canvas = np.delete(canvas, (lowest_y), axis=0)

    # get index of the second top corner, store it and delete from array
    lowest_y = int(np.argmin(canvas, axis=0)[1:])
    top_corner2 = canvas[lowest_y]
    x2 = top_corner2[0]
    canvas = np.delete(canvas, (lowest_y), axis=0)

    # store the two bottom corners
    """
    An AttributeError was triggered at canvas[1] in the pilot session -->
    most likely a corner was not found (len(canvas) 3 instead of 4) -->
    When that happens, this should stop any update and retry.
    """
    bottom_corner1 = canvas[0]
    x3 = bottom_corner1[0]
    bottom_corner2 = canvas[1]
    x4 = bottom_corner2[0]

    # sort corners along top left, top right, bottom left, bottom right
    if x1 > x2:
        top_left = top_corner2
        top_right = top_corner1
    else:
        top_left = top_corner1
        top_right = top_corner2
    if x3 > x4:
        bottom_left = bottom_corner2
        bottom_right = bottom_corner1
    else:
        bottom_left = bottom_corner1
        bottom_right = bottom_corner2

    # match image points to new corner points according to known ratio
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

    # this value needs changing according to image size
    img_y = 1000  # warped image height
    # height/width ratio given current grid
    ratio = 1.3861874976470018770202169598726
    img_x = int(round(img_y * ratio))  # warped image width
    # size for warped image
    pts2 = np.float32([[0, 0],[img_x, 0],[img_x, img_y],[0, img_y]])
    # get perspective to warp image
    perspective = cv2.getPerspectiveTransform(pts1, pts2)

    # warp image according to the perspective transform and store image
    # warped = cv2.warpPerspective(img, perspective, (img_x, img_y))
    # cv2.imwrite('warpedGrid.jpg', warped)
    #features, origins, radius = create_features(img_y, img_x)
    return perspective, img_x, img_y, pts1


def create_features(height, width):
    """
    Function that calculates the midpoint coordinates of each hexagon in the
    transformed picture.

    TODO: change to a (geo)dataframe?
    """
    # determine size of grid circles from image and step size in x direction
    radius = (height / 10)
    x_step = np.cos(np.deg2rad(30)) * radius
    origins = []
    column = []
    # determine x and y coordinates of gridcells midpoints
    for a in range(1, 16):  # range reflects gridsize in x direction
        x = (x_step * a)
        for b in range(1, 11):  # range reflects gridsize in y direction
            if a % 2 == 0:
                if b == 10:
                    continue
                y = (radius * b)
            else:
                y = (radius * (b - 0.5))
            origins.append([x, y])
            column.append(a)
    origins = np.array(origins)
    board_cells = len(origins)

    y_jump = radius/2
    dist = y_jump/np.cos(np.deg2rad(30))
    x_jump = dist/2
    features = []
    for i, (x, y) in enumerate(origins):
        # determine all the corner points of the hexagon
        point1 = [x+dist, y]
        point2 = [x+x_jump, y+y_jump]
        point3 = [x-x_jump, y+y_jump]
        point4 = [x-dist, y]
        point5 = [x-x_jump, y-y_jump]
        point6 = [x+x_jump, y-y_jump]
        # create a geojson polygon for the hexagon
        polygon = geojson.Polygon([[point1, point2, point3, point4, point5,
                                    point6, point1]])
        feature = geojson.Feature(id=i, geometry=polygon)
        feature.properties["red_markers"] = 100
        feature.properties["red_changed"] = True
        feature.properties["blue_markers"] = 100
        feature.properties["blue_changed"] = True
        feature.properties["column"] = column[i]
        feature.properties["ghost_hexagon"] = False
        #feature.properties["tygron_id"] = i
        # these x and y centers are not actually relevant --> features are
        # transformed to other coordinates.
        feature.properties["x_center"] = int(round(x))
        feature.properties["y_center"] = int(round(y))
        features.append(feature)
    # create geojson featurecollection with all hexagons.
    features = geojson.FeatureCollection(features)
    #TODO: sent back geodataframe rather than geojson file
    #features_gpd = gpd.GeoDataFrame.from_features(features['features'])
    if False:
        with open('ghost_cells_test.geojson', 'w') as f:
            geojson.dump(features, f, sort_keys=True, indent=2)
    return features, radius


def drawMask(origins, img, path=""):
    """
    Function that can be called to draw the mask and print hexagon numbers.
    This function is currently not called. Can be removed at a later stage.
    """
    global count
    global radius
    r = int(round(radius / 2))
    for (x, y, count) in origins:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        #cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.putText(img, str(count), (x - 50, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)
    # save image with grid
    cv2.imwrite(join(path, 'drawGrid.jpg'), img)
    print('success')
    return
