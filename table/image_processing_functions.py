# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: haanrj
"""

import cv2
import numpy as np
from os import mkdir
from os.path import join
from shapely.geometry import shape


def detect_markers(img, pers, img_x, img_y, r, features, turn=0,
                   method='rgb', path="", debug=False, old_opencv=False):
    # warp image it to calibrated perspective
    warped = cv2.warpPerspective(img, pers, (img_x, img_y))
    # save the file of this turn.
    if debug:
        filename = 'calibrated_image%d.jpg' % turn
        cv2.imwrite(join(path, filename), warped)
    if method == 'LAB':
        # convert the image to labspace and get the A and B channels.
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        A = cv2.medianBlur(A, 5)
        B = cv2.medianBlur(B, 5)
        # set ranges for blue and red to create masks, may need updating.
        lower_blue = 0
        upper_blue = 110
        lower_red = 160
        upper_red = 255
        # isolate red and blue in the image.
        blue_mask = cv2.inRange(B, lower_blue, upper_blue)
        red_mask = cv2.inRange(A, lower_red, upper_red)
        # add dilation to the image.
        kernel = np.ones((2, 2), np.uint8)
        red_dilate = cv2.dilate(red_mask, kernel, iterations=1)
        blue_dilate = cv2.dilate(blue_mask, kernel, iterations=1)
        if debug:
            cv2.imwrite(join(path, 'red_mask_dilated_LAB%d.jpg' % turn),
                        red_dilate)
            cv2.imwrite(join(path, 'blue_mask_dilated_LAB%d.jpg' % turn),
                        blue_dilate)
    elif method == 'YCrCb':
        # convert to image to YCRCb and get the Cb and Cr channels.
        ycrcb = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        Cb = cv2.medianBlur(Cb, 5)
        Cr = cv2.medianBlur(Cr, 5)
        # set ranges for blue and red to create masks, may need updating.
        lower_blue = 160
        upper_blue = 255
        lower_red = 160
        upper_red = 255
        # isolate red and blue in the image.
        blue_mask = cv2.inRange(Cb, lower_blue, upper_blue)
        red_mask = cv2.inRange(Cr, lower_red, upper_red)
        # add dilation to the image.
        kernel = np.ones((2, 2), np.uint8)
        red_dilate = cv2.dilate(red_mask, kernel, iterations=1)
        blue_dilate = cv2.dilate(blue_mask, kernel, iterations=1)
        if debug:
            cv2.imwrite(join(path, 'red_mask_dilated_YCrCb%d.jpg' % turn),
                        red_dilate)
            cv2.imwrite(join(path, 'blue_mask_dilated_YCrCb%d.jpg' % turn),
                                     blue_dilate)
    else:
        # split the RGB channels and get the B and R channels.
        B, G, R = cv2.split(warped)
        B = cv2.medianBlur(B, 5)
        R = cv2.medianBlur(R, 5)
        # set ranges for blue and red to create masks, may need updating.
        lower_blue = 220
        upper_blue = 255
        lower_red = 230
        upper_red = 255
        # isolate red and blue in the image.
        blue_mask = cv2.inRange(B, lower_blue, upper_blue)
        red_mask = cv2.inRange(R, lower_red, upper_red)
        # add dilation to the image.
        kernel = np.ones((2, 2), np.uint8)
        red_dilate = cv2.dilate(red_mask, kernel, iterations=1)
        blue_dilate = cv2.dilate(blue_mask, kernel, iterations=1)
        # save masks, can be removed later.
        if debug:
            cv2.imwrite(join(path, 'red_mask_dilated_RGB%d.jpg' % turn),
                        red_dilate)
            cv2.imwrite(join(path, 'blue_mask_dilated_RGB%d.jpg' % turn),
                        blue_dilate)

    # create a mask for the region of interest processing. convert diameter to actual radius as int value.
    y_cell = int(round(r / 2))
    x_cell = int(round(1.25 * y_cell))
    # slightly lower radius to create a mask that removes contours from slight grid misplacement.
    margin = int(round(y_cell * 0.95))
    # empty mask of grid cell size.
    mask = np.zeros((2 * y_cell, 2 * x_cell), dtype="uint8")
    # calculate the x and y differences for the points of the hexagon shaped mask.
    dist = margin/np.cos(np.deg2rad(30))
    x_jump = int(round(dist/2))
    y_jump = margin
    dist = int(round(dist))
    # define the 6 corner points of the hexagon shaped mask.
    point1 = [x_cell+dist, y_cell]
    point2 = [x_cell+x_jump, y_cell+y_jump]
    point3 = [x_cell-x_jump, y_cell+y_jump]
    point4 = [x_cell-dist, y_cell]
    point5 = [x_cell-x_jump, y_cell-y_jump]
    point6 = [x_cell+x_jump, y_cell-y_jump]
    # create the mask.
    pts = np.array([point1, point2, point3, point4, point5, point6], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # create folders to store individual cells if wanted.
    if debug:
        dir_path = join(path, 'image_processing_files')
        try:
            mkdir(dir_path)
            print("Directory ", dir_path, " created.")
        except FileExistsError:
            print("Directory ", dir_path,
                  " already exists, overwriting files.")
        red_path = join(dir_path, 'red_rois')
        try:
            mkdir(red_path)
            print("Directory ", red_path, " created.")
        except FileExistsError:
            print("Directory ", red_path,
                  " already exists, overwriting files.")
        blue_path = join(dir_path, 'blue_rois')
        try:
            mkdir(blue_path)
            print("Directory ", blue_path, " created.")
        except FileExistsError:
            print("Directory ", blue_path,
                  " already exists, overwriting files.")
    
    # for loop that analyzes all grid cells.
    for i, feature in enumerate(features.features):
        # some adjustments to adjust for the distance of the camera to the side. This is not a necessity, but increases
        # robustness. Alternative would be to improve the perspective warp.
        if feature.properties["ghost_hexagon"]:
            continue
        geom = shape(feature.geometry)
        x = int(round(geom.centroid.x, 0))
        y = int(round(geom.centroid.y, 0))
        if i < 10:
            x = x - 7
        elif i < 19:
            x = x - 5
        elif i < 29:
            x = x - 3
        elif i < 38:
            x = x - 2
        elif i < 48:
            x = x - 1
        elif i < 95:
            pass
        elif i < 105:
            x = x + 1
        elif i < 114:
            x = x + 2
        elif i < 124:
            x = x + 3
        elif i < 133:
            x = x + 5
        else:
            x = x + 7

        # get region of interest (ROI) for red (geometry) and analyse number
        # of contours (which identifies the markers) found.
        roi_red = red_dilate[y-y_cell:y+y_cell, x-x_cell:x+x_cell]
        roi_blue = blue_dilate[y-y_cell:y+y_cell, x-x_cell:x+x_cell]

        # mask the ROI to eliminate adjacent grid cells.
        masked_img_red = cv2.bitwise_and(roi_red, roi_red, mask=mask)
        masked_img_blue = cv2.bitwise_and(roi_blue, roi_blue, mask=mask)

        # find contours within masked ROI.
        if old_opencv:
            im1, contours_red, h1 = cv2.findContours(masked_img_red, cv2.RETR_CCOMP,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            im2, contours_blue, h2 = cv2.findContours(masked_img_blue, cv2.RETR_CCOMP,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours_red, h1 = cv2.findContours(masked_img_red, cv2.RETR_CCOMP,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            contours_blue, h2 = cv2.findContours(masked_img_blue, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # store each analyzed ROI, not necessary for the script to work.
        if debug:
            filename_red = 'red_roi_%i.jpg'%i
            filename_blue = 'blue_roi_%i.jpg'%i
            cv2.imwrite(join(red_path, filename_red), masked_img_red)
            cv2.imwrite(join(blue_path, filename_blue), masked_img_blue)

        # sort the contours in such a way that the biggest contours are first for both red and blue
        if not old_opencv:
            contours_red = list(contours_red)
            contours_blue = list(contours_blue)
        contours_red.sort(key = cv2.contourArea, reverse = True)

        contours_blue.sort(key = cv2.contourArea, reverse = True)
        no_red = False
        no_blue = False
        try:
            largest_red = cv2.contourArea(contours_red[0])
        except IndexError:
            no_red = True
        try:
            largest_blue = cv2.contourArea(contours_blue[0])
        except IndexError:
            no_blue = True
        
        count_red = 0
        count_blue = 0
        margin = (1 / 3)
        
        if not no_red:
            margin_red = largest_red * margin
            for contour in contours_red:
                area = cv2.contourArea(contour)
                if area > margin_red:
                    count_red += 1
        if not no_blue:
            margin_blue = largest_blue * margin
            for contour in contours_blue:
                area = cv2.contourArea(contour)
                if area > margin_blue:
                    count_blue += 1

        feature.properties["red_markers"] = count_red
        feature.properties["blue_markers"] = count_blue
    return features



