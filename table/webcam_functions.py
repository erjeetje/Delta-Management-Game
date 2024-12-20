# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: HaanRJ
"""

import cv2
from os.path import join


def get_image(turn, path="", debug=False):
    """
    Method to get an image from the camera. Activates the webcam, makes sure the camera is set to fullHD and takes a
    picture. The picture is stored for reference and backup. i stands for turn.
    """
    # 0 or 1, always 0 if built-in webcam disabled. Change the value if it uses a wrong camera.
    cam = cv2.VideoCapture(0)  
    # TODO add some kind of check to determine this is the correct camera.
    
    # set correct width and height of the camera.
    cam.set(3, 1920)
    cam.set(4, 1080)
    ret_val, img = cam.read()
    if debug:
        # save the image taken, currently skipped. Can be removed later.
        filename = 'turn_picture%i.jpg' % turn
        cv2.imwrite(join(path, filename), img)
    return img


def main():
    get_image(0, debug=True)


if __name__ == '__main__':
    main()
