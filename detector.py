```python
import cv2

class FaceDetector:
    def __init__(self, model_path):
        # Wczytanie modelu wykrywania twarzy
        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.dnn = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    def detect(self, image):
        # Wykrywanie twarzy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        dnn_outputs = self.dnn.forward(image)

        return faces, dnn_outputs
```

Poniższy kod tworzy klasę `FaceDetector` w Pythonie, która pozwala na wykrywanie twarzy w obrazach. Zawiera metody `__init__` do wczytania modelu wykrywania twarzy i `detect` do wykrywania twarzy w obrazach.