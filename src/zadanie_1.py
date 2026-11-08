```python
import cv2

class FaceDetector:
    def __init__(self, face_cascade_path):
        # Load the Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_faces(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return faces
```

W tym kodzie tworzymy klasę `FaceDetector` z metodą `detect_faces`, która przyjmuje obraz i zwraca listę przeszkodanych twarzy. W metodzie `__init__` dodajemy obiekt `cv2.CascadeClassifier` do klasy, który używa siekcji Haar Cascade do detekcji twarzy. W metodzie `detect_faces` przekształcamy obraz do skala sztuki (grayscale) i wykrywamy twarze za pomocą `detectMultiScale` metody.