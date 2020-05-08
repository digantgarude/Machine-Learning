import cv2

cap = cv2.VideoCapture(0)

# Create the haar cascade
# Make sure the haarcascade_frontalface_default.xml file is in the same directory
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Read frame
	ret, frame = cap.read()

	# Convert frame to Black and White image (For speeding up the processing)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	# Returns the rectangles over image.
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
	# Therefore number of rectangles == No. of faces detected.
	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	# Here (x,y) is the top left corner in the image.
	# w - width, h - height is added to get other corners in the image.

	# Display the resulting frame
	cv2.imshow('frame', frame)

	# Press 'q' key to exit.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture object.
cap.release()
cv2.destroyAllWindows()
