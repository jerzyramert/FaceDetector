```python
import cv2
import numpy as np
from math import cos, sin

def detect_face(frame):
    # Implement the face detection algorithm here
    # For example, using Haar Cascade Classifier
    # This is a placeholder for the actual implementation
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate the angle of rotation
        angle = np.arctan2(y - center_y, x - center_x)
        
        # Draw a circle at the center of the face
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 2)
        
        # Draw the angle
        cv2.putText(frame, f"{int(angle * 180 / np.pi)}°", (center_x, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
```

**Pozwolenia:**
- Możesz używać bibliotek `opencv-python` i `numpy`.
- Nie musisz implementować własnej funkcji obliczającej kąt obracania.
- Możesz używać klasycznych metod obliczających kąt obracania (np. `np.arctan2`).
- Musisz używać biblioteki `cv2.CascadeClassifier` do wykrywania twarzy.
- Musisz używać funkcji `cv2.rectangle` do rysowania prostokątów wokół wykrytych twarzy.
- Musisz używać funkcji `cv2.putText` do dodawania tekstu na klatce.
- Musisz używać funkcji `cv2.circle` do rysowania kół na klatce.