```python
import pytest
from detector import Detector

def test_detector():
    # Przykładowa data dla testów
    text = "Zawartość tekstu do analizy."
    expected_output = ["Wąs", "tekstu", "do", "analizy", ".", "Zawartość", "tekstu", "do", "analizy", "!"]
    
    # Utworzenie obiektu Detector
    detector = Detector()
    
    # Wywołanie metody detect
    output = detector.detect(text)
    
    # Sprawdzenie czy wynik zgadza się z przewidywanym wynikiem
    assert output == expected_output, f"Oczekiwano {expected_output}, dostarczono {output}"

# Przykładowy test z pytest
if __name__ == "__main__":
    pytest.main()
```