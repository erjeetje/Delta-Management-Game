This repository is the code base for the Delta Management Game, a serious game on salt intrusion in the Dutch Rhine-Meuse estuary, which is developed as part of the Salti Solutions research program (grant number P18-32), which is (partly) financed by the Dutch Research Council (NWO) and the Dutch Ministry of Economic Affairs. It is work in progress, with the code currently showing a demonstrator of changes to the Rhine-Meuse estuary (boundary conditions, bathymetry changes) and the effects on salinity in the estuary.

To run the demonstrator, the shape files of the channels in the estuary are required as input files that are currently not included in this repository. These may be added later.

The game integrates the 4.3.7 network version of the IMSIDE model, an idealized model on salt intrusion in Deltas and Estuaries. The model is developed in a separate project of the Salti Solutions research program, the repository with all versions is here: https://github.com/nietBouke/IMSIDE/. the IMSIDE model version used is adapted to interface with other parts of the game.

You can use the DMGclean.yml file or requirements.txt to create a Python environment with all used libraries. 
