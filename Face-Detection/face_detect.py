import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
# Make sure the haarcascade_frontalface_default.xml file is in the same directory
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
# Convert image to Black and White image (For speeding up the processing)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# Returns the rectangles over image.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)
# Therefore number of rectangles == No. of faces detected.
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Here (x,y) is the top left corner in the image.
# w - width, h - height is added to get other corners in the image.


cv2.imshow("Faces found", image)
cv2.waitKey(0)
