# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:14:34 2024

@author: HaanRJ
"""

import geojson
from table import grid_calibration_functions as cali
from table import image_processing_functions as detect
from table import webcam_functions as webcam
from table import compare_functions as compare
from table import transform_functions as transform
#import ghostcell_functions as ghosts
from copy import deepcopy
from cv2 import imread, flip
from os import mkdir
from os.path import join, realpath, dirname
from time import time


class Table():
    """
    Game table object. Run the get_board_state function to update the board state and subsequently retrieve the board
    state via the hexagons getter function.
    """
    def __init__(self, game, mirror=None, test=False, save=False, debug=False):
        super(Table, self).__init__()
        self.game = game
        self.initialized = False
        self.start_new_turn = False
        self.mirror = mirror
        self.test = test
        self.debug = debug
        self.turn = self.game.turn
        self.turn_count = self.game.turn_count
        self.update_count = 0
        self.save = save
        self.reloading = False
        self.reload_enabled = False
        # Memory variables
        self.turn_img = None
        self._hexagons = None
        self.hexagons_prev = None
        self.hexagons_model = None
        self.hexagons_flipped = None
        self.transforms = None
        self.pers = None
        # may not be necessary to store these, but some methods would need to
        # be updated (in gridCalibration and processImage).
        self.img_x = None
        self.img_y = None
        self.radius = None
        # list that tracks updated hexagons in case hexagons are placed back.
        self.set_paths()
        return


    def set_paths(self):
        # Defines and creates all the path locations used troughout the script.
        # In case changes were made to file locations (with this file as
        # the reference point), these need to be updated. 
        self.dir_path = dirname(realpath(__file__))
        self.input_path = join(self.dir_path, "input_files")
        # Set a local path where you want to store intermediate output (optional in code)
        self.local_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\07 DMG design\coding (notebooks)\game to IMSIDE"
        self.store_path = join(self.local_path, 'storing_files')
        self.processing_path = join(self.local_path, 'processing_files')
        self.config_path = join(self.local_path, 'config_files')
        try:
            mkdir(self.store_path)
            print("Directory ", self.store_path, " Created.")
        except FileExistsError:
            print("Directory ", self.store_path,
                  " already exists, overwriting files.")
        try:
            mkdir(self.processing_path)
            print("Directory ", self.processing_path, " Created.")
        except FileExistsError:
            print("Directory ", self.processing_path,
                  " already exists, overwriting files.")
        try:
            mkdir(self.config_path)
            print("Directory ", self.config_path, " Created.")
        except FileExistsError:
            print("Directory ", self.config_path,
                  " already exists, overwriting files.")
        return

    @property
    def hexagons(self):
        return self._hexagons

    def get_board_state(self):
        """
        Function that handles configuring and calibrating the game board and retrieving the number of markers in each
        hexagon location.
        """
        ping = time()
        self.turn = self.game.turn
        self.turn_count = self.game.turn_count
        if self.initialized:
            self.hexagons_prev = deepcopy(self._hexagons)
            self.start_new_turn = False
        # get the current board as an image from the webcam, if self.test == True it loads an image instead.
        self.get_image()
        continue_code = self.calibrate_camera()
        if not continue_code:
            print("failed to calibrate camera and not testing, aborting rest of the function")
            return
        found_hexagons = self.get_hexagons()
        if not found_hexagons:
            print("failed to get hexagons, aborting rest of method")
            return
        """
        in case of transform the features from the board coordinates to other coordinates, run the function below
        with the correct transform input (run multiple times for multiple reprojections). The "transform_to" has to be
        declared in the transform_features function in transform_functions.py, linking it to a correct transformation
        included in the self.transforms created in the create_calibration_file function. 
        """
        #self.transform_hexagons(transform_to="img_flip")
        if self.initialized:
            self._hexagons = compare.compare_hex(self._hexagons, self.hexagons_prev)
        #self.turn_count += 1
        pong = time()
        if self.save:
            self.save_files(end_round=False)
        if self.initialized:
            print("Calibration and board processing time:", str(round(pong - ping, 2)), "seconds.")
        else:
            print("Calibration, creating features and board processing time:", str(round(pong - ping, 2)), "seconds.")
        self.initialized = True
        # always reset the reloading variables in case reloading is used.
        self.reloading = False
        self.reload_enabled = False
        return

    def end_round(self):
        """
        This function ends a game round, saves the board and resets variables.
        """
        if not self.initialized:
            print("the game table is not yet calibrated, please first run once to initialize")
            return
        if self.start_new_turn:
            print("It appears as if you have pressed end_round twice,",
                  "there has been no update from the previous board state so far.")
            return
        print("Ending round", str(self.turn), "and applying all the changes.",
              "Make sure to save the files for this turn!")
        # if self.save is defined as True, the end of turn files are automatically stored.
        if self.save:
            self.save_files(end_round=True)
        self.start_new_turn = True
        #self.turn += 1
        #self.turn_count = 1
        return

    def reload(self):
        """
        Note: the reload function has not been tested yet, which is a to do.
        """
        if not self.reload_enabled:
            print("Are you sure you want to iniate a reload? If you intended "
                  "to press reload, press reload again to engage the reload.")
            self.reload_enabled = True
            return
        # this function needs to be checked, to load previously stored files instead.
        filename = 'hexagons%d.geojson' % self.turn
        with open(join(self.store_path, filename)) as f:
            self._hexagons = geojson.load(f)
        self.initialized = True
        self.start_new_turn = True
        #self.turn += 1
        #self.turn_count = 0
        return

    def get_image(self):
        """
        Get a camera image.
        """
        if self.test:
            filename = 'DMG_table%s_%d.jpg' % (self.turn, self.turn_count)
            self.turn_img = imread(join(self.input_path, filename))
        else:
            self.turn_img = webcam.get_image(self.turn, path=self.processing_path, debug=self.debug)
        if self.mirror is not None:
            self.turn_img = flip(self.turn_img, self.mirror)
        print("Retrieved initial board image")
        return

    def calibrate_camera(self):
        """
        Calibrate the camera/board.
        
        try - except TypeError --> if nothing returned by method, then go to
        # test mode.
        """
        try:
            canvas = cali.detect_corners(self.turn_img, method='adaptive', debug=self.debug,
                                                 path=self.processing_path)
        except TypeError:
            print("No camera detected or these is something wrong with the image captured, aborting")
            return False
        try:
            self.pers, self.img_x, self.img_y, cut_points = cali.rotate_grid(canvas)
        except AttributeError:
            print("did not find all four calibration corners, aborting initialization.",
                  "Check webcam picture (run with self.debug=True) to find the problem.")
            return False
        print("Calibrated camera.")
        # create the calibration file for use by other methods and store it
        if not self.initialized:
            self._hexagons, self.radius = cali.create_features(
                    self.img_y, self.img_x)
            self.transforms = transform.create_calibration_file(self.img_x, self.img_y, path=self.config_path,
                                                                debug=self.debug)
        return True

    def transform_hexagons(self, transform_to="model"):
        """
        Function that transforms the hexagons to the coordinates that the SandBox / Tygron uses.
        """
        if not self.reloading:
            # update the hexagons to initial board state.
            self.hexagons_flipped = transform.transform_features(self._hexagons, self.transforms, export=transform_to)
        print("Transformed hexagons suitable for model.")
        return

    def get_hexagons(self):
        """
        Function that creates/gets the new hexagons. Gets them from either the
        camera (live mode) or image file (test mode).
        """
        self._hexagons = detect.detect_markers(
            self.turn_img, self.pers, self.img_x, self.img_y, self.radius, self._hexagons, method='LAB',
            path=self.processing_path, debug=self.debug)
        """
        TODO: add flexible ghost cell option. 
        """
        #if not self.initialized:
        #    self._hexagons = ghosts.set_values(self._hexagons)
        print("Retrieved board state.")
        return True

    def save_files(self, end_round=True):
        """
        This function save the game board, both intermediate and end of round.
        """
        if end_round:
            filename = 'hexagons%d.geojson' % self.turn
        else:
            filename = ('hexagons%s_%d' % (self.turn, self.turn_count)) + '.geojson'
        with open(join(self.store_path, filename), 'w') as f:
            geojson.dump(
                    self._hexagons, f, sort_keys=True, indent=2)
        print("Saved hexagon file for turn " + str(self.turn) + ".")
        return

def main():
    """
    mirror determines if the image from the webcam is flipped. Declare no value for no flip. 0 = flip along x-axis,
    1 along y-axis, -1 along both x- and y-axis.
    test determines if you run the board live (False) or use test images from the webcam (True).
    save determines if board states are saved (currently as geojsons).
    debug determines if intermediate steps are saved, like red and blue detection and images of each hexagon tile.
    """
    table = Table(mirror=1, test=True, save=True, debug=False)
    for x in range(5):
        table.get_board_state()
        hexagons = table.hexagons
        table.end_round()
        #if hexagons is not None:
            #for feature in hexagons.features:
                #print("id:", feature.id, "red:", feature.properties["red_markers"], "blue:", feature.properties["blue_markers"])
    print("completed code")


if __name__ == '__main__':
    main()
