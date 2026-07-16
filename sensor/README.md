# /readme - in depth analysis of existing code on made for the proforge

# /model - a pipeline to show how to extract segments that may need reprinting. 

# /tests - a large archive of code used to understand the laser, gcode, and imaging 

# Some constants of the proforge4
The pixel to gcode global coordinates ratio is around 26.13 with the height it's at. The few mm the camera moves up and down should not affect this significantly. 

The laser moves by around 26 pixels when z=1 on the docking motion, frame 2419 in the second scan of the disk. 

The laser is rotated by around 26.2 degrees, but this can change easily when messing with the laser. 