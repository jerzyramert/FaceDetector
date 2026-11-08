```python
import cv2
import numpy as np

def detect_face(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_faces(image, faces):
    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Load the video
video = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = video.read()
    if not ret:
        break
    # Detect faces in the frame
    faces = detect_face(frame)
    # Draw the faces on the frame
    frame = draw_faces(frame, faces)
    # Display the frame
    cv2.imshow('Face Detection', frame)
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video.release()
# Close all windows
cv2.destroyAllWindows()
```