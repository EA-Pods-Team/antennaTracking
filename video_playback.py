# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:33:06 2019

@author: Joel Miller
"""

import cv2
from areaSelect import areaSelector

def playVid(vidLoc):
    cap = cv2.VideoCapture(vidLoc)  
    class Found(Exception): pass
    try:
        while True:
            success, frame = cap.read()
			frame = frame[mask]
            key = cv2.waitKey(50) & 0xFF
            if not success:
                print('didnt find video')
                break
    
            # Pause the Video
            if key == 32:
                while True:
                    key2 = cv2.waitKey(1) or 0xFF
                    vidTime = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,0))
                    cv2.putText(frame,str(vidTime)+' seconds',(5,10),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,0),1)
                    cv2.imshow('Video',frame)
                    # Play the Video
                    if key2 == 32:
                        break
                    if key2 == 27 or key2 == 113:
                        raise Found
					# Select region of interest
					if key2 == ord('w'):
						cap.release()
						cv2.destroyAllWindows()
						objLoc = areaSelector(frame,'Select Region of Interest')
						return objLoc, frame
                        
            # Skip forward 3 seconds
            if key == ord('d'):
                skip = cap.get(cv2.CAP_PROP_POS_MSEC) + 3000
                cap.set(cv2.CAP_PROP_POS_MSEC,skip)
                success, frame = cap.read()
				frame = frame[mask]
            
            # Skip Back 3 seconds
            if key == ord('a'):
                skip = cap.get(cv2.CAP_PROP_POS_MSEC) - 3000
                cap.set(cv2.CAP_PROP_POS_MSEC,skip)
                success, frame = cap.read()
				frame = frame[mask]
                
            # Quit Video Playback by pressing 'q' or ESC
            if key == 113 or key == 27:
                break

            vidTime = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,0))
            cv2.putText(frame,str(vidTime)+' seconds',(5,10),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,0),1)
            cv2.imshow('Video',frame)
            
    except Found:
        print('done')
    
    cap.release()
    cv2.destroyAllWindows()
    
