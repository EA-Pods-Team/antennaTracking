import cv2,collections,math,argparse,pytesseract,os
from video_playback import playVid
from textRecog import textRecog
import numpy as np
from areaSelect import areaSelector, mask2Rect, mask2Center

parser = argparse.ArgumentParser(description='Track Antenna')
parser.add_argument('vidname', metavar='video', type=str, nargs='?',
					help='video for analysis')
					
args = parser.parse_args()
vidName = args.vidname
if(not vidName):
	vidName = '../vids/vid7.mp4'
	
def textRecognize(mask1,mask2,angleError,truAz,truEl):
	# crop region of interest
	roi1 = frame[mask1]
	roi2 = frame[mask2]
	# Convert to grayscale
	roi1 = cv2.cv2Color(roi1, cv2.COLOR_BGR2GRAY)
	roi2 = cv2.cv2Color(roi2, cv2.COLOR_BGR2GRAY)
	# double image size, linear interpolation
	roi1 = cv2.resize(roi1,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
	roi2 = cv2.resize(roi2,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
	# perform thresholding
	level,roi1 = cv2.threshold(roi1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	level,roi2 = cv2.threshold(roi2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# Take ROI and use tesseract to convert to string
	config = '--oem 1 --psm 7'
	text1 = pytesseract.image_to_string(roi1,config=config)
	text2 = pytesseract.image_to_string(roi2,config=config)
	try:
		textNum1 = int(text1[1:3]) + int(text1[4:6])/60 - truAz
		textNum2 = int(text2[1:2]) + int(text1[3:5])/60 - truEl
		textDiff = round(math.sqrt(textNum1**2 + textNum2**2),2)
		pixel2truth = round(textDiff - angleError,2)
		print(pixel2truth)
	except:
		print('failed')
	
def cal(vidName):
	cap = cv2.VideoCapture(vidName)
	success,frame = cap.read()
	if not success:
		print('Video not found')
	mask = areaSelector(frame,"select the Az text region")
	frame = frame[mask]
	boresightPixelLoc = mask2Center(areaSelector(frame,'Highlight the box crosshair'))
	azMask = areaSelector(frame,'select the Az text region')
	elMask = areaSelector(frame,'select the El text region')
	
	print('Select an object as a reference point with no angle error')
	objLoc1,frame = playVid(vidName,mask)
	truAz = textrecog(frame[azMask])
	truEl = textRecog(frame[elMask])
	truAz = round(int(truAz[1:3]) + float(truAz[4:6])/60,2)
	truEl = round(int(truEl[1:2]) + float(truEl[3:5])/60,2)
	refPixel = mask2Rect(objLoc1)[0]
	
	print('Select the same reference point with error')
	objLoc2,frame = playVid(vidName,mask)
	errAz = textrecog(frame[azMask])
	errEl = textRecog(frame[elMask])
	errAz = round(int(errAz[1:3]) + float(errAz[4:6])/60,2)
	errEl = round(int(errEl[1:2]) + float(errEl[3:5])/60,2)
	errPixel = mask2Rect(objLoc2)[0]
	
	degreeSeparation = round(abs(math.sqrt(truAz**2 + truEl**2) - math.sqrt(errAz**2 + errEl**2)),2)
	print(degreeSeparation)
	pixelSeparation = round(abs(math.sqrt(refPixel[0]**2 + refPixel[1]**2) - math.sqrt(errPixel[0]**2 + errPixel[1]**2)),0)
	print(pixelSeparation)
	degrees2pixels = degreeSeparation/pixelSeparation
	
	calName = vidName[vidName.rfind('/')+1:-4]
	calFile = open(f'cal_{calName}_data.txt','w')
	calFile.write(f'{mask}\n')
	calFile.write(f'{azMask}\n')
	calFile.write(f'{elMask}\n')
	calFile.write(f'{degrees2pixels}\n')
	calFile.write(f'{truAz}\n')
	calFile.write(f'{truEl}\n')
	calFile.write(f'{boresightPixelLoc}')
	calFile.close()
	return mask,azMask,elMask,degrees2pixels,truAz,truEl,boresightPixelLoc

def calCheck(vidName):
	calName = vidName[vidName.rfind('/')+1:-4]
	pathName = f'cal_{calName}_data.txt'
	if os.path.isfile(pathName) == True:
		calFile = open(f'cal_{calName}_data.txt','r')
		cals = []
		[cals.append(i) for i in calFile]
		mask = eval(cals[0].rstrip())
		azMask = eval(cals[1].rstrip())
		elMask = eval(cals[2].rstrip())
		degrees2pixels = float(cals[3].rstrip())
		truAz = float(cals[4].rstrip())
		truEl = float(cals[5].rstrip())
		boresightPixelLoc = eval(cals[6])
		calFile.close()
		return mask,azMask,elMask,degrees2pixels,truAz,truEl,boresightPixelLoc
	else:
		mask,azMask,elMask,degrees2pixels,truAz,truEl,boresightPixelLoc = cal(vidName)
		return mask,azMask,elMask,degrees2pixels,truAz,truEl,boresightPixelLoc

antenna_cascade = cv2.CascadeClassifier('antennaCascade.xml')
cap = cv2.VideoCapture(vidName)
success,frame = cap.read()
mask,azMask,elMask,degrees2pixels,truAz,truEl,boresightPixelLoc = calCheck(vidName)

# This will change how much of the trace persistance there is
bufSize = 40
xBuf = collections.deque([frame.shape[1]//2]*bufSize,maxlen=bufSize)
yBuf = collections.deque([frame.shape[0]//2]*bufSize,maxlen=bufSize)
[xBuf.append(boresightPixelLoc[0]) for i in range(bufSize)]
[yBuf.append(boresightPixelLoc[1]) for i in range(bufSize)]

while success:
	success, frame = cap.read()
	if not success: break
	frame = frame.copy()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	dish = antenna_cascade.detectMultiScale(gray)
	for (x,y,w,h) in dish:
		xBuf.append(x-60)
		yBuf.append(y)
		
		pixelError = round(math.sqrt((boresightPixelLoc[0] - xBuf[-1])**2 + (boresightPixelLoc[1] - yBuf[-1])**2))
		angleError = round(degrees2pixels*pixelError,2)
		cv2.putText(frame,'Angle error:' + str(angleError), (int(frame.shape[1]//(3/2)),20),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		
		a = np.vstack((list(xBuf),list(yBuf))).astype(np.int32).T
		for i in list(range(bufSize-2),0,-1)):
			cv2.line(refFrame,(xBuf[i],yBuf[i]),(xBuf[i+1],yBuf[i+1]),(0,0,255),4,4)
			alpha = i/bufSize
			cv2.addWeighted(refFrame,alpha,frame,1-alpha,0,frame)
		
	if type(dish) is not np.ndarray:
		cv2.putText(frame,'object not detected',(5,10),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)
		
	key = cv2.waitKey(60) & 0xFF
	if key == 32:
		while True:
			key2 = cv2.waitKey(1) & 0xFF
			vidTime = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,0))
			cv2.putText(refFrame,'angle error: ' + str(angleError),(int(frame.shape[1]//(3/2)),20),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
			cv2.imshow('antenna tracking', frame)
			if key2 == 32:
				break
	if success: cv2.imshow('antenna tracking', frame)
	
	if key == ord('q') or key == 27:
		break
		
cap.release()
cv2.destroyAllWindows()			

	
