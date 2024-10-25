<h2>Introduction</h2>
This repository is the code base for the Delta Management Game, a serious game on salt intrusion in the Dutch Rhine-Meuse estuary, which is developed as part of the Salti Solutions research program (grant number P18-32), which is (partly) financed by the Dutch Research Council (NWO) and the Dutch Ministry of Economic Affairs. It is work in progress, with the code currently showing a demonstrator of changes to the Rhine-Meuse estuary (boundary conditions, bathymetry changes) and the effects on salinity in the estuary.

The game integrates the 4.3.7 network version of the IMSIDE model, an idealized model on salt intrusion in Deltas and Estuaries. The model is developed in a separate project of the Salti Solutions research program, the repository with all versions is here: https://github.com/nietBouke/IMSIDE/. the IMSIDE model version used is adapted to interface with other parts of the game.

You can use the DMGclean.yml file or requirements.txt to create a Python environment with all used libraries. 

<h2>Using the demonstrator</h2>
The demonstrator runs via the "game_runfile". At the bottom of that file, you can find a dictionary that provides settings. At the moment this concerns what input scenarios to take and what the corresponding sea level rise for these is (both work in progress and will be updated). The game is based around four turns. If you run the file, the game will set up with an initial run of the model for turn 1. Once done, a GUI pops up and the output of turn 1 is shown. At the bottom right you can find buttons for the other three turns, which will not yet show any output. At the top you can switch between visualizing salinity concentration as values or as categorized data.

Interacting with the delta's bathymetry and boundary conditions currently goes via the text field and the "change delta" and "run model" buttons. Actual changes are added in the text field, following the structure of python dictionaries. The menu of possible changes are an:

<table>
  <tr>
    <th>Action</th>
    <th>Textfield entry</th>
    <th>Example entry</th>
  </tr>
  <tr>
    <td>widen channel</td>
    <td>polygon_id: "widen"</td>
    <td>59: "widen"</td>
  </tr>
  <tr>
    <td>narrow channel</td>
    <td>polygon_id: "narrow"</td>
    <td>59: "narrow"</td>
  </tr>
  <tr>
    <td>deepen channel</td>
    <td>polygon_id: "deepen"</td>
    <td>59: "deepen"</td>
  </tr>
  <tr>
    <td>undeepen channel</td>
    <td>polygon_id: "undeepen"</td>
    <td>59: "undeepen"</td>
  </tr>
  <tr>
    <td>split channel (permanently block)</td>
    <td>polygon_id: "split"</td>
    <td>59: "split"</td>
  </tr>
  <tr>
    <td>change Haringvliet discharge</td>
    <td>"Qhar": value</td>
    <td>"Qhar": 120</td>
  </tr>
  <tr>
    <td>change Hagestein weir (Lek) discharge</td>
    <td>"Qlek": value</td>
    <td>"Qlek": 80</td>
  </tr>
  <tr>
    <td>change Hollandsche IJssel (KWA) discharge</td>
    <td>"Qhij": value</td>
    <td>"Qhij": 5</td>
  </tr>
</table>

You can either add multiple changes (e.g. add multiple of the example entries separately) or combine them as one entry like typing ' 59: "deepen", 49: "deepen", 39: "deepen", 29: "deepen" ' (without the ') in the textfield. Hitting the "change delta" button while process these changes (print statements will show this). The polygon_ids (like 59 or 49 above) and their location in the delta and its channels can be found in the "polygon_id_reference_figure.png" figure or the "polygon_id_reference_table.png" in the repository. For example, the above entry with polygon_ids 59, 49, 39 and 29 all concern the Nieuwe Waterweg channel. Note that the code does not hardcode these polygon_ids, rather it processes them based on polygon references. Making changes to this input would thus change the polygon_ids.

Once you are done with changes, you can hit the "run model" to see have the model run with the delta changes in place. Once done (you will see the progress as print statements), you can view the output under the corresponding turn button. For example, after running the model an additional time after the initial game run adds output that can be accessed under the "turn 2" button.
