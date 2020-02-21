# antennaTracking

Right now the program is optimized for the video footage obtained from a previous test.

The program is expecting an Antenna Cascade Classifier for the object detection. It finds the 
two small antennas on the pole in order to reference the bigger antenna on the left. On line
126 of the antennaDetector.py program I offset the location of the detected object by 60 pixels
in order to align to the center of the bigger antenna.

// more instructions to follow
