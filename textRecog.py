import pytesseract,cv2

def textRecog(frame):
	roi1 = frame.copy()
	roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
	
	roi1 = cv2.resize(roi1,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
	level,roi1 = cv2.threshold(roi1,0,255,c2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	config = '--oem 1 --psm 7'
	text = pytesseract.image_to_string(roi1, config=config)
	
	try:
		return text
	except:
		return 'failed'
