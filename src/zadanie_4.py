```python
import cv2

# Otwieramy kamerę
cap = cv2.VideoCapture(0)

while True:
    # Odczytujemy obraz z kamery
    ret, frame = cap.read()

    if not ret:
        print("Nie można odczytać obrazu z kamery.")
        break

    # Wyświetlamy obraz na ekranie
    cv2.imshow('Kamra', frame)

    # Zatrzymanie programu po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknięcie kamerы
cap.release()

# Wyświetlenie komunikatu na ekranie
cv2.destroyAllWindows()
```