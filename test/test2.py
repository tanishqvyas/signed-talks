# Importing Modules
import cv2
import numpy as np
import math

# Configuring camera

# Open camera
capture = cv2.VideoCapture(0)


# Exits as soon as live feed is stopped
while capture.isOpened():

	# Capture frames from camera
	# frame stores output frames
	ret, frame = capture.read()

	# Get hand data from the rectangle sub window
	cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 0)
	crop_image = frame[100:300, 100:300]

	# Applying Gaussian Blurrrr
	# Blurring video///////////
	blur = cv2.GaussianBlur(crop_image, (3, 3), 0) 

	# Change color-space from BGR -> HSV
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	# Create a binary image with where white will be skin colors and rest black
	mask2 = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))

	# Filtering out noise by dialation and erosion

	# Kernel for Morphological transformastion
	kernel = np.ones((5,5))

	# Apply Morphologial func to filter out bg noise
	dilation = cv2.dilate(mask2, kernel, iterations=1)
	erosion = cv2.erode(dilation, kernel, iterations=1)

	# Removing extra pixel values using Gaussian blur and Threshold
	filtered = cv2.GaussianBlur(erosion, (3,3), 0)
	ret, thresh = cv2.threshold(filtered, 127, 255, 0)

	# Show threshold image
	cv2.imshow("Threshold", thresh)

	# Find Contours
	image,contours,heirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	try:

		# Find max contour area
		contour = max(contours, key=lambda x: cv2.contourArea(x))

		# Create bounding rect around contour
		# x,y co-ordinates
		x,y,w,h = cv2.boundingRect(contour)
		cv2.rectangle(crop_image, (x,y), (x+w, y+h),(0, 0, 255), 0)

		# Find convex hull
		hull = cv2.convexHull(contour)

		# Drawing cntour
		drawing = np.zeros(crop_image.shape, np.uint8)
		cv2.drawContours(drawing, [contour], -1, (0,255,0), 0)
		cv2.drawContours(drawing, [hull], -1, (0,0,255), 0)

		# Convexity defect >>>>>>>>>read
		# Finding em
		hull = cv2.convexHull(contour, returnPoints=False)
		defects = cv2.convexityDefects(contour, hull)

		# Use cosine rule to find angle of the far point from start and end point
		#basically convex tips for all such defects
		count_defects = 0

		for i in range(defects.shape[0]):

			s, e, f, d = defects[i, 0] #start end farthest distance
			start = tuple(contour[s][0])
			end = tuple(contour[e][0])
			far = tuple(contour[f][0])


			a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 )
			b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2 )
			c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2 )

			angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 *b*c)) * 180)

			
			# if angle > 90 draw a circle at farthest point
			#debug
			if angle <= 90:
				count_defects += 1
				cv2.circle(crop_image, far, 1, [0,0,255], -1)

			cv2.line(crop_image, start, end, [0,255,0], 2)


			# Print num of fingers via num of defect

			if count_defects == 0:
				cv2.putText(frame, "ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
			elif count_defects == 1:
				cv2.putText(frame, "TWO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
			elif count_defects == 2:
				cv2.putText(frame, "THREE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)	
			elif count_defects == 3:
				cv2.putText(frame, "FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)	
			elif count_defects == 4:
				cv2.putText(frame, "FIVE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)	
			else:
				cv2.putText(frame, "NONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)	


	except:
		pass


	# Show required images
	cv2.imshow("GESTURES",frame)
	all_image = np.hstack((drawing, crop_image))
	#cv2.imshow('Contours', all_image)

	# Close the camera if 'c' is pressed

	if cv2.waitKey(1) == ord('c'):
		break
capture.release()
cv2.destroyAllWindows()