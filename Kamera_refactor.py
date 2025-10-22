import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import traceback # Do śledzenia błędów
import os # Do operacji na plikach i katalogach
import json # Dodano do obsługi JSON

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.photo = None 
        self.canvas_image_item = None

        # --- Sekcja Konfigurowalnych Parametrów (Wartości Domyślne) ---
        self.target_face_width = 800
        self.target_plate_width = 720
        self.face_save_padding = 50
        self.plate_save_padding = 1
        self.image_save_interval_seconds = 1.0
        self.min_face_size = (100, 100)
        self.min_plate_size = (50, 20)
        self.face_detection_scale_factor = 1.1
        self.face_detection_min_neighbors = 5
        self.plate_detection_scale_factor = 1.1
        self.plate_detection_min_neighbors = 5
        self.face_confidence_threshold = 5.0
        self.plate_confidence_threshold = 1.0
        self.roi_size_percentage = 0.9
        self.font_face_val = "FONT_HERSHEY_SIMPLEX"
        self.font_scale_info = 0.6
        self.font_scale_confidence = 0.5
        self.font_color_info = (255, 255, 255)
        self.confidence_text_color = (255, 255, 0)
        self.line_type_info = 2
        self.text_y_offset = 25
        self.line_spacing = 20
        self.rect_roi_color_default = (0, 255, 0)
        self.rect_roi_color_face_detected = (255, 0, 0)
        self.rect_roi_thickness = 2
        self.rect_face_color = (0, 0, 255)
        self.rect_face_thickness = 2
        self.rect_plate_color = (0, 255, 255)
        self.rect_plate_thickness = 2
        self.max_camera_check_index = 3
        
        self.is_batch_processing = False
        self.camera_index_to_resume = -1 

        # Wczytaj parametry z pliku, jeśli istnieje
        self._load_parameters_from_file() # WAŻNE: przed dynamicznym self.font_face

        # Dynamiczne przypisanie wartości cv2.FONT_HERSHEY_SIMPLEX
        # Musi być po _load_parameters_from_file(), aby użyć potencjalnie wczytanej wartości font_face_val
        self.font_face = getattr(cv2, self.font_face_val, cv2.FONT_HERSHEY_SIMPLEX)
        # --- Koniec Sekcji Konfigurowalnych Parametrów ---


        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.camera_name_info = "N/A"
        self.camera_index_used = -1 

        self.last_face_save_time = 0
        self.last_plate_save_time = 0

        os.makedirs("faces", exist_ok=True)
        os.makedirs("plates", exist_ok=True)
        os.makedirs("images", exist_ok=True) 
        print("Utworzono/sprawdzono katalogi 'faces', 'plates' i 'images'.")

        self.menubar = tk.Menu(self.window)
        self.camera_menu = tk.Menu(self.menubar, tearoff=0)
        
        self.available_cameras = []
        print("Wykrywanie dostępnych kamer dla menu...")
        for i in range(self.max_camera_check_index + 1): 
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                self.available_cameras.append(i)
                self.camera_menu.add_command(label=f"Kamera {i}", command=lambda idx=i: self.switch_camera(idx))
                cap_test.release()
        
        if not self.available_cameras:
            self.camera_menu.add_command(label="Brak wykrytych kamer", state="disabled")
        
        self.camera_menu.add_separator()
        self.camera_menu.add_command(label="Przetwórz folder 'images'", command=self.start_batch_processing)
        self.camera_menu.add_separator()
        self.camera_menu.add_command(label="Edytuj Parametry", command=self._open_settings_dialog) 
        self.camera_menu.add_command(label="Zapisz Parametry do Pliku", command=self._save_parameters_to_file) 

        self.menubar.add_cascade(label="Opcje", menu=self.camera_menu) 
        self.window.config(menu=self.menubar)

        self.face_cascade = None
        face_cascade_path = 'haarcascade_frontalface_default.xml'
        if os.path.exists(face_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self.face_cascade.empty():
                messagebox.showerror("Błąd modelu", f"Nie można załadować kaskady dla twarzy: {face_cascade_path}")
                self.face_cascade = None
        else:
            messagebox.showwarning("Brak modelu", f"Nie znaleziono pliku dla detekcji twarzy: {face_cascade_path}\nUmieść go w katalogu ze skryptem.")

        self.plate_cascade = None
        plate_cascade_path = 'haarcascade_russian_plate_number.xml'
        if os.path.exists(plate_cascade_path):
            self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
            if self.plate_cascade.empty():
                messagebox.showerror("Błąd modelu", f"Nie można załadować kaskady dla tablic rej.: {plate_cascade_path}")
                self.plate_cascade = None
        else:
            messagebox.showwarning("Brak modelu", f"Nie znaleziono pliku dla detekcji tablic rej.: {plate_cascade_path}\nUmieść go w katalogu ze skryptem.")

        initial_cam_idx_to_try = self.available_cameras[0] if self.available_cameras else 0 
        
        print(f"Inicjalizacja kamery startowej (indeks: {initial_cam_idx_to_try})...")
        self.vid = cv2.VideoCapture(initial_cam_idx_to_try)
        if not self.vid.isOpened() and initial_cam_idx_to_try == 0 and 1 in self.available_cameras: 
            print("Nie udało się otworzyć kamery 0, próbuję kamery 1...")
            initial_cam_idx_to_try = 1
            self.vid = cv2.VideoCapture(initial_cam_idx_to_try)

        if self.available_cameras and not self.vid.isOpened(): 
             messagebox.showwarning("Błąd kamery", "Nie można otworzyć domyślnej kamery. Spróbuj wybrać inną z menu.")
        elif not self.available_cameras: 
            messagebox.showerror("Błąd kamery", "Nie wykryto żadnych kamer. Podłącz kamerę i uruchom program ponownie.")
            self.window.destroy()
            return

        if self.vid.isOpened():
            self.camera_index_used = initial_cam_idx_to_try
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else: 
            self.camera_index_used = -1 
            self.width = 640 
            self.height = 480
        
        self._update_camera_info_string()

        if self.width == 0 or self.height == 0 and self.vid.isOpened(): 
            messagebox.showerror("Błąd kamery", f"Kamera {self.camera_index_used} zgłasza nieprawidłową rozdzielczość.")
            if self.vid.isOpened(): self.vid.release()
            self.camera_index_used = -1 
            self._update_camera_info_string()

        self._setup_ui_elements() 

        self.running = True 
        if self.vid and self.vid.isOpened(): 
            self.capture_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.info_label_text.set("Brak aktywnej kamery. Wybierz z menu lub podłącz kamerę.")
            
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)

    def _load_parameters_from_file(self):
        """Wczytuje parametry konfiguracyjne z pliku config.json."""
        config_filepath = "config.json"
        try:
            if os.path.exists(config_filepath):
                with open(config_filepath, "r", encoding="utf-8") as f:
                    loaded_params = json.load(f)
                
                print(f"Znaleziono plik konfiguracyjny: {config_filepath}. Wczytywanie parametrów...")
                
                # Lista parametrów, które oczekujemy i ich typy (do konwersji z listy na krotkę)
                param_type_map = {
                    "min_face_size": tuple, "min_plate_size": tuple,
                    "font_color_info": tuple, "confidence_text_color": tuple,
                    "rect_roi_color_default": tuple, "rect_roi_color_face_detected": tuple,
                    "rect_face_color": tuple, "rect_plate_color": tuple
                }

                for key, value in loaded_params.items():
                    if hasattr(self, key):
                        try:
                            # Konwersja list z JSON na krotki dla określonych parametrów
                            if key in param_type_map and isinstance(value, list):
                                setattr(self, key, tuple(value))
                                print(f"  Wczytano '{key}': {tuple(value)}")
                            else: # Dla innych typów, spróbuj bezpośrednio ustawić
                                # Można dodać bardziej szczegółową walidację typów, jeśli potrzeba
                                current_type = type(getattr(self, key))
                                if current_type == bool: # Obsługa bool
                                     setattr(self, key, bool(value))
                                else:
                                     setattr(self, key, current_type(value)) # Próba konwersji do typu domyślnego atrybutu
                                print(f"  Wczytano '{key}': {getattr(self, key)}")
                        except (ValueError, TypeError) as e_type:
                            print(f"  Ostrzeżenie: Niepoprawny typ/wartość dla parametru '{key}' w config.json: {value}. Używanie wartości domyślnej. Błąd: {e_type}")
                    else:
                        print(f"  Ostrzeżenie: Nieznany parametr '{key}' w config.json. Pomijanie.")
                print("Wczytywanie parametrów zakończone.")
            else:
                print(f"Plik konfiguracyjny {config_filepath} nie znaleziony. Używanie wartości domyślnych.")
        except json.JSONDecodeError as e_json:
            print(f"Błąd dekodowania pliku JSON {config_filepath}: {e_json}. Używanie wartości domyślnych.")
        except Exception as e:
            print(f"Nieoczekiwany błąd podczas wczytywania parametrów z {config_filepath}: {e}. Używanie wartości domyślnych.")
            traceback.print_exc()


    def _get_configurable_params_dict(self):
        """Zwraca słownik z aktualnymi wartościami konfigurowalnych parametrów."""
        return {
            "target_face_width": self.target_face_width,
            "target_plate_width": self.target_plate_width,
            "face_save_padding": self.face_save_padding,
            "plate_save_padding": self.plate_save_padding,
            "image_save_interval_seconds": self.image_save_interval_seconds,
            "min_face_size": list(self.min_face_size), 
            "min_plate_size": list(self.min_plate_size), 
            "face_detection_scale_factor": self.face_detection_scale_factor,
            "face_detection_min_neighbors": self.face_detection_min_neighbors,
            "plate_detection_scale_factor": self.plate_detection_scale_factor,
            "plate_detection_min_neighbors": self.plate_detection_min_neighbors,
            "face_confidence_threshold": self.face_confidence_threshold,
            "plate_confidence_threshold": self.plate_confidence_threshold,
            "roi_size_percentage": self.roi_size_percentage,
            "font_face_val": self.font_face_val, 
            "font_scale_info": self.font_scale_info,
            "font_scale_confidence": self.font_scale_confidence,
            "font_color_info": list(self.font_color_info),
            "confidence_text_color": list(self.confidence_text_color),
            "line_type_info": self.line_type_info,
            "text_y_offset": self.text_y_offset,
            "line_spacing": self.line_spacing,
            "rect_roi_color_default": list(self.rect_roi_color_default),
            "rect_roi_color_face_detected": list(self.rect_roi_color_face_detected),
            "rect_roi_thickness": self.rect_roi_thickness,
            "rect_face_color": list(self.rect_face_color),
            "rect_face_thickness": self.rect_face_thickness,
            "rect_plate_color": list(self.rect_plate_color),
            "rect_plate_thickness": self.rect_plate_thickness,
            "max_camera_check_index": self.max_camera_check_index
        }

    def _save_parameters_to_file(self):
        """Zapisuje aktualne parametry konfiguracyjne do pliku config.json."""
        if self.is_batch_processing:
            messagebox.showwarning("Przetwarzanie", "Nie można zapisać parametrów podczas przetwarzania folderu obrazów.")
            return
        
        params_to_save = self._get_configurable_params_dict()
        
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(params_to_save, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Zapisano", "Parametry zostały zapisane do pliku config.json.")
            print("Parametry zapisane do config.json")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie można zapisać parametrów do pliku: {e}")
            print(f"Błąd zapisu parametrów do config.json: {e}")


    def _open_settings_dialog(self):
        if self.is_batch_processing:
            messagebox.showwarning("Przetwarzanie", "Nie można edytować parametrów podczas przetwarzania folderu obrazów.")
            return

        settings_window = tk.Toplevel(self.window)
        settings_window.title("Edytuj Parametry")
        settings_window.geometry("450x650") 
        settings_window.transient(self.window) 
        settings_window.grab_set() 

        main_frame = ttk.Frame(settings_window, padding="10")
        main_frame.pack(expand=True, fill="both")
        
        canvas_settings = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_settings.yview)
        scrollable_frame = ttk.Frame(canvas_settings)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_settings.configure(
                scrollregion=canvas_settings.bbox("all")
            )
        )

        canvas_settings.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_settings.configure(yscrollcommand=scrollbar.set)

        canvas_settings.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        params_to_edit = [
            ("target_face_width", "Docelowa szer. twarzy (px)", "int"),
            ("target_plate_width", "Docelowa szer. tablicy (px)", "int"),
            ("face_save_padding", "Padding zapisu twarzy (px)", "int"),
            ("plate_save_padding", "Padding zapisu tablicy (px)", "int"),
            ("image_save_interval_seconds", "Interwał zapisu (s)", "float"),
            ("min_face_size", "Min. rozmiar twarzy (szer,wys)", "tuple_int"),
            ("min_plate_size", "Min. rozmiar tablicy (szer,wys)", "tuple_int"),
            ("face_detection_scale_factor", "Skala det. twarzy (np. 1.1)", "float"),
            ("face_detection_min_neighbors", "Sąsiedzi det. twarzy (int)", "int"),
            ("plate_detection_scale_factor", "Skala det. tablicy (np. 1.1)", "float"),
            ("plate_detection_min_neighbors", "Sąsiedzi det. tablicy (int)", "int"),
            ("face_confidence_threshold", "Próg pewności twarzy (float)", "float"),
            ("plate_confidence_threshold", "Próg pewności tablicy (float)", "float"),
            ("roi_size_percentage", "Rozmiar ROI (% całości, 0.1-1.0)", "float"),
        ]

        param_entries = {} 

        for i, (attr_name, label_text, data_type) in enumerate(params_to_edit):
            ttk.Label(scrollable_frame, text=label_text + ":").grid(row=i, column=0, sticky="w", pady=2, padx=5)
            entry = ttk.Entry(scrollable_frame, width=20) 
            entry.grid(row=i, column=1, sticky="ew", pady=2, padx=5)
            
            current_value = getattr(self, attr_name)
            if data_type == "tuple_int":
                entry.insert(0, f"{current_value[0]},{current_value[1]}")
            else:
                entry.insert(0, str(current_value))
            param_entries[attr_name] = (entry, data_type)

        scrollable_frame.columnconfigure(1, weight=1) 

        button_frame = ttk.Frame(main_frame) 
        button_frame.pack(fill="x", pady=10)


        save_button = ttk.Button(button_frame, text="Zapisz", 
                                 command=lambda: self._save_settings(param_entries, settings_window))
        save_button.pack(side="left", padx=10, expand=True)

        cancel_button = ttk.Button(button_frame, text="Anuluj", command=settings_window.destroy)
        cancel_button.pack(side="right", padx=10, expand=True)
        
        settings_window.wait_window() 

    def _save_settings(self, param_entries, dialog_window):
        """Zapisuje nowe wartości parametrów."""
        new_settings = {}
        try:
            for attr_name, (entry_widget, data_type) in param_entries.items():
                value_str = entry_widget.get()
                if data_type == "int":
                    val = int(value_str)
                    if "min_neighbors" in attr_name and val < 0:
                        raise ValueError(f"Wartość dla '{attr_name}' musi być nieujemna.")
                    new_settings[attr_name] = val
                elif data_type == "float":
                    val = float(value_str)
                    if "scale_factor" in attr_name and val <= 1.0:
                         raise ValueError(f"Wartość dla '{attr_name}' musi być większa niż 1.0.")
                    if "roi_size_percentage" in attr_name and not (0.0 < val <= 1.0):
                        raise ValueError(f"Wartość dla '{attr_name}' musi być między 0.0 a 1.0.")
                    new_settings[attr_name] = val
                elif data_type == "tuple_int":
                    parts = value_str.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"Nieprawidłowy format dla '{attr_name}'. Oczekiwano 'liczba,liczba'.")
                    val1 = int(parts[0].strip())
                    val2 = int(parts[1].strip())
                    if val1 <=0 or val2 <=0:
                        raise ValueError(f"Wymiary dla '{attr_name}' muszą być dodatnie.")
                    new_settings[attr_name] = (val1, val2)
                else: 
                    new_settings[attr_name] = value_str
            
            for attr_name, value in new_settings.items():
                setattr(self, attr_name, value)
                print(f"Zaktualizowano parametr '{attr_name}' na: {value}")
            
            # Po udanej aktualizacji parametrów, zaktualizuj font_face
            self.font_face = getattr(cv2, self.font_face_val, cv2.FONT_HERSHEY_SIMPLEX)


            messagebox.showinfo("Zapisano", "Parametry zostały zaktualizowane.", parent=dialog_window)
            dialog_window.destroy()

        except ValueError as e:
            messagebox.showerror("Błąd wartości", f"Nieprawidłowa wartość dla jednego z pól: {e}", parent=dialog_window)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił nieoczekiwany błąd: {e}", parent=dialog_window)


    def _update_camera_info_string(self):
        if self.vid and self.vid.isOpened():
            try:
                backend_name = self.vid.getBackendName()
                self.camera_name_info = f"Kamera (Indeks: {self.camera_index_used}, Backend: {backend_name})"
            except Exception: 
                self.camera_name_info = f"Kamera (Indeks: {self.camera_index_used})"
        else:
            self.camera_name_info = "Kamera nieaktywna"


    def _setup_ui_elements(self):
        window_width = self.width + 40 
        window_height = self.height + 140 
        self.window.geometry(f"{window_width}x{window_height}")

        if not hasattr(self, 'canvas') or self.canvas is None:
            self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg="black", relief="sunken", borderwidth=2)
            self.canvas.pack(padx=10, pady=10)
        else:
            self.canvas.config(width=self.width, height=self.height)

        if not hasattr(self, 'info_label_text') or self.info_label_text is None:
            self.info_label_text = tk.StringVar()
            self.info_label = ttk.Label(self.window, textvariable=self.info_label_text, font=("Helvetica", 10))
            self.info_label.pack(pady=(0, 5), fill=tk.X, padx=10)
        
        self.info_label.config(wraplength=self.width if self.width > 0 else 600) 
        self.info_label_text.set(f"Ładowanie... {self.camera_name_info} @ {self.width}x{self.height}")


        if not hasattr(self, 'btn_quit') or self.btn_quit is None:
            self.btn_quit = ttk.Button(self.window, text="Zamknij", command=self.quit_app, style="Accent.TButton")
            self.btn_quit.pack(pady=10)
            style = ttk.Style() 
            style.configure("Accent.TButton", foreground="white", background="#007AFF", font=("Helvetica", 12, "bold"), padding=10)
            style.map("Accent.TButton", foreground=[('active', 'white')], background=[('active', '#0056b3')])
        
        if self.canvas_image_item:
            self.canvas.delete(self.canvas_image_item)
        self.canvas_image_item = None
        self.photo = None


    def switch_camera(self, new_camera_index):
        print(f"Próba przełączenia na kamerę o indeksie: {new_camera_index}")
        if self.is_batch_processing:
            messagebox.showwarning("Przetwarzanie", "Nie można zmienić kamery podczas przetwarzania folderu obrazów.")
            return

        if new_camera_index == self.camera_index_used and self.vid and self.vid.isOpened():
            print("Wybrana kamera jest już aktywna.")
            return

        self.running = False 
        if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            print("Oczekiwanie na zakończenie bieżącego wątku kamery...")
            self.capture_thread.join(timeout=2.0) 
            if self.capture_thread.is_alive():
                print("Ostrzeżenie: Poprzedni wątek kamery nie zakończył się w wyznaczonym czasie.")
        self.capture_thread = None 

        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            print(f"Zwalnianie kamery {self.camera_index_used}...")
            self.vid.release()
            self.vid = None 

        print(f"Otwieranie kamery {new_camera_index}...")
        new_vid_temp = cv2.VideoCapture(new_camera_index)
        if new_vid_temp.isOpened():
            self.vid = new_vid_temp
            self.camera_index_used = new_camera_index
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.width == 0 or self.height == 0:
                messagebox.showerror("Błąd kamery", f"Nowa kamera (indeks {new_camera_index}) zgłasza nieprawidłową rozdzielczość.")
                self.vid.release()
                self.vid = None
                self.camera_index_used = -1 
                self._update_camera_info_string()
                self.info_label_text.set(f"Błąd kamery {new_camera_index}. Wybierz inną.")
                return

            self._update_camera_info_string()
            self._setup_ui_elements() 

            self.fps_start_time = time.time()
            self.fps_counter = 0
            self.current_fps = 0
            self.last_face_save_time = 0
            self.last_plate_save_time = 0
            self.face_detected_in_roi_flag = False
            self.plate_detected_flag = False
            
            self.running = True
            self.capture_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.capture_thread.start()
            print(f"Pomyślnie przełączono na kamerę {new_camera_index}.")
        else:
            messagebox.showerror("Błąd zmiany kamery", f"Nie można otworzyć kamery o indeksie {new_camera_index}.")
            print(f"Nie udało się otworzyć kamery {new_camera_index}.")
            self.camera_index_used = -1 
            self._update_camera_info_string()
            self.info_label_text.set(f"Nie udało się otworzyć kamery {new_camera_index}. Wybierz inną.")

    def start_batch_processing(self):
        if self.is_batch_processing:
            messagebox.showinfo("Informacja", "Przetwarzanie folderu obrazów już trwa.")
            return

        images_folder_path = "images"
        if not os.path.isdir(images_folder_path):
            messagebox.showerror("Błąd folderu", f"Folder '{images_folder_path}' nie istnieje. Utwórz go i dodaj obrazy.")
            return
        
        image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            messagebox.showinfo("Informacja", f"Brak plików .png lub .jpg w folderze '{images_folder_path}'.")
            return

        messagebox.showinfo("Rozpoczęto przetwarzanie", f"Rozpoczynam przetwarzanie obrazów z folderu '{images_folder_path}'.\nTo może chwilę potrwać. Strumień z kamery zostanie wstrzymany.")
        
        self.camera_index_to_resume = self.camera_index_used 
        
        if self.running and hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            print("Wstrzymywanie kamery na żywo na czas przetwarzania wsadowego...")
            self.running = False
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                print("Ostrzeżenie: Wątek kamery na żywo nie zakończył się poprawnie przed wsadowym.")
            self.capture_thread = None 
        
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            print(f"Zwalnianie kamery {self.camera_index_used} przed przetwarzaniem wsadowym.")
            self.vid.release()
            self.vid = None
        
        self.camera_index_used = -1 
        self._update_camera_info_string()

        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.delete("all") 
            canvas_center_x = self.width / 2 if self.width > 0 else 320
            canvas_center_y = self.height / 2 if self.height > 0 else 240
            self.canvas.create_text(canvas_center_x, canvas_center_y, 
                                   text="Przetwarzanie folderu obrazów...", fill="white", font=("Helvetica", 16))
        if hasattr(self, 'info_label_text'):
             self.info_label_text.set("Przetwarzanie folderu obrazów...")

        self.is_batch_processing = True
        self.camera_menu.entryconfig("Przetwórz folder 'images'", state="disabled")
        if self.available_cameras: 
            for i in self.available_cameras:
                 self.camera_menu.entryconfig(f"Kamera {i}", state="disabled")

        batch_thread = threading.Thread(target=self._process_image_folder_thread_worker, daemon=True)
        batch_thread.start()

    def _process_and_draw_detections(self, frame, frame_for_saving, current_time, is_live_feed, source_details=None):
        current_frame_height, current_frame_width = frame.shape[:2]
        
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, current_frame_width, current_frame_height 
        offset_x_roi, offset_y_roi = 0, 0
        frame_roi_for_face_detection = frame 

        if self.face_cascade:
            if is_live_feed: 
                temp_roi_w = int(current_frame_width * self.roi_size_percentage)
                temp_roi_h = int(current_frame_height * self.roi_size_percentage)
                
                if temp_roi_w > 0 and temp_roi_h > 0 and current_frame_width > 0 and current_frame_height > 0:
                    roi_x1 = int((current_frame_width - temp_roi_w) / 2)
                    roi_y1 = int((current_frame_height - temp_roi_h) / 2)
                    roi_x2 = roi_x1 + temp_roi_w 
                    roi_y2 = roi_y1 + temp_roi_h
                    offset_x_roi, offset_y_roi = roi_x1, roi_y1                     

                    slice_roi_x1 = max(0, roi_x1)
                    slice_roi_y1 = max(0, roi_y1)
                    slice_roi_x2 = min(current_frame_width, roi_x2)
                    slice_roi_y2 = min(current_frame_height, roi_y2)

                    if slice_roi_x2 > slice_roi_x1 and slice_roi_y2 > slice_roi_y1:
                        frame_roi_for_face_detection = frame[slice_roi_y1:slice_roi_y2, slice_roi_x1:slice_roi_x2]
                    else:
                        frame_roi_for_face_detection = frame[0:0, 0:0] 
                else: 
                    frame_roi_for_face_detection = frame[0:0, 0:0]
            
            if frame_roi_for_face_detection.size > 0 and \
               frame_roi_for_face_detection.shape[0] >= self.min_face_size[1] and \
               frame_roi_for_face_detection.shape[1] >= self.min_face_size[0]:
                gray_roi_for_face = cv2.cvtColor(frame_roi_for_face_detection, cv2.COLOR_BGR2GRAY)
                try:
                    faces, _, level_weights = self.face_cascade.detectMultiScale3(
                        gray_roi_for_face,
                        scaleFactor=self.face_detection_scale_factor,
                        minNeighbors=self.face_detection_min_neighbors,
                        minSize=self.min_face_size,
                        outputRejectLevels=True
                    )
                    if faces is not None and len(faces) > 0:
                        self.face_detected_in_roi_flag = True 
                        processed_one_save_this_cycle = False
                        for i, (x, y, w, h) in enumerate(faces):
                            face_x1_global = offset_x_roi + x
                            face_y1_global = offset_y_roi + y
                            face_x2_global = offset_x_roi + x + w
                            face_y2_global = offset_y_roi + y + h
                            cv2.rectangle(frame, (face_x1_global, face_y1_global), (face_x2_global, face_y2_global), self.rect_face_color, self.rect_face_thickness)
                            
                            confidence = 0.0
                            if level_weights is not None and i < len(level_weights):
                                confidence = level_weights[i]
                                cv2.putText(frame, f"{confidence:.2f}", (face_x1_global, face_y1_global - 10), self.font_face, self.font_scale_confidence, self.confidence_text_color, self.line_type_info)

                            can_save_time = not is_live_feed or (current_time - self.last_face_save_time > self.image_save_interval_seconds)
                            if not processed_one_save_this_cycle and can_save_time and confidence >= self.face_confidence_threshold:
                                fs_x1 = max(0, face_x1_global - self.face_save_padding)
                                fs_y1 = max(0, face_y1_global - self.face_save_padding)
                                fs_x2 = min(current_frame_width, face_x2_global + self.face_save_padding)
                                fs_y2 = min(current_frame_height, face_y2_global + self.face_save_padding)
                                image_crop_to_save = frame_for_saving[fs_y1:fs_y2, fs_x1:fs_x2]

                                if image_crop_to_save.size > 0:
                                    orig_h_crop, orig_w_crop = image_crop_to_save.shape[:2]
                                    resized_img = image_crop_to_save
                                    saved_w, saved_h = orig_w_crop, orig_h_crop
                                    if orig_w_crop > 0 and orig_h_crop > 0:
                                        ratio = self.target_face_width / float(orig_w_crop)
                                        new_h = int(orig_h_crop * ratio)
                                        if new_h > 0:
                                            resized_img = cv2.resize(image_crop_to_save, (self.target_face_width, new_h), interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR)
                                            saved_w, saved_h = resized_img.shape[1], resized_img.shape[0]
                                    
                                    base_name = os.path.splitext(source_details["original_filename"])[0] if source_details else f"live"
                                    png_filename = f"{base_name}_face_{i if source_details else self.camera_index_used}_{int(current_time)}.png"
                                    png_filepath = os.path.join("faces", png_filename)
                                    json_filepath = os.path.join("faces", os.path.splitext(png_filename)[0] + ".json")

                                    try:
                                        cv2.imwrite(png_filepath, resized_img)
                                        print(f"  Zapisano twarz: {png_filepath} (pewność: {confidence:.2f})")
                                        if is_live_feed: self.last_face_save_time = current_time
                                        processed_one_save_this_cycle = True
                                        if not is_live_feed and 'saved_detections_count_ref' in source_details: source_details['saved_detections_count_ref'][0] +=1


                                        json_data = {
                                            "detection_type": "face", "confidence_score": float(f"{confidence:.2f}"),
                                            "original_detected_object": {"width": int(w), "height": int(h)},
                                            "saved_image_details": {"png_filename": png_filename, "saved_width": int(saved_w), "saved_height": int(saved_h), "padding_applied": self.face_save_padding},
                                            "source_info": source_details if source_details else {"source_type": "live_camera", "timestamp": int(current_time), "camera_index": self.camera_index_used},
                                            "normalization": {"target_width": self.target_face_width}
                                        }
                                        self._save_detection_json(json_data, json_filepath)
                                    except Exception as e_save: print(f"  Błąd zapisu twarzy: {e_save}")
                except cv2.error as e_cv:
                    print(f"  Błąd OpenCV (twarz) {'w pliku ' + source_details['original_filename'] if source_details else 'na żywo'}: {e_cv}")
            
            if is_live_feed: 
                current_roi_color_display = self.rect_roi_color_face_detected if self.face_detected_in_roi_flag else self.rect_roi_color_default
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), current_roi_color_display, self.rect_roi_thickness)


        if self.plate_cascade:
            gray_frame_for_plate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            if gray_frame_for_plate.shape[0] >= self.min_plate_size[1] and gray_frame_for_plate.shape[1] >= self.min_plate_size[0]:
                try:
                    plates, _, level_weights_plate = self.plate_cascade.detectMultiScale3(
                        gray_frame_for_plate,
                        scaleFactor=self.plate_detection_scale_factor,
                        minNeighbors=self.plate_detection_min_neighbors,
                        minSize=self.min_plate_size,
                        outputRejectLevels=True
                    )
                    if plates is not None and len(plates) > 0:
                        self.plate_detected_flag = True
                        processed_one_save_this_cycle = False
                        for i, (x_p, y_p, w_p, h_p) in enumerate(plates):
                            cv2.rectangle(frame, (x_p, y_p), (x_p + w_p, y_p + h_p), self.rect_plate_color, self.rect_plate_thickness)
                            confidence_plate = 0.0
                            if level_weights_plate is not None and i < len(level_weights_plate):
                                confidence_plate = level_weights_plate[i]
                                cv2.putText(frame, f"{confidence_plate:.2f}", (x_p, y_p - 10), self.font_face, self.font_scale_confidence, self.confidence_text_color, self.line_type_info)

                            can_save_time_plate = not is_live_feed or (current_time - self.last_plate_save_time > self.image_save_interval_seconds)
                            if not processed_one_save_this_cycle and can_save_time_plate and confidence_plate >= self.plate_confidence_threshold:
                                ps_x1 = max(0, x_p - self.plate_save_padding)
                                ps_y1 = max(0, y_p - self.plate_save_padding)
                                ps_x2 = min(current_frame_width, x_p + w_p + self.plate_save_padding)
                                ps_y2 = min(current_frame_height, y_p + h_p + self.plate_save_padding)
                                plate_image_to_save = frame_for_saving[ps_y1:ps_y2, ps_x1:ps_x2]

                                if plate_image_to_save.size > 0:
                                    orig_h_p, orig_w_p = plate_image_to_save.shape[:2]
                                    resized_plate = plate_image_to_save
                                    saved_w_p, saved_h_p = orig_w_p, orig_h_p
                                    if orig_w_p > 0 and orig_h_p > 0:
                                        ratio_p = self.target_plate_width / float(orig_w_p)
                                        new_h_p = int(orig_h_p * ratio_p)
                                        if new_h_p > 0:
                                            resized_plate = cv2.resize(plate_image_to_save, (self.target_plate_width, new_h_p), interpolation=cv2.INTER_AREA if ratio_p < 1.0 else cv2.INTER_LINEAR)
                                            saved_w_p, saved_h_p = resized_plate.shape[1], resized_plate.shape[0]

                                    base_name_p = os.path.splitext(source_details["original_filename"])[0] if source_details else f"live"
                                    png_filename_p = f"{base_name_p}_plate_{i if source_details else self.camera_index_used}_{int(current_time)}.png"
                                    png_filepath_p = os.path.join("plates", png_filename_p)
                                    json_filepath_p = os.path.join("plates", os.path.splitext(png_filename_p)[0] + ".json")
                                    
                                    try:
                                        cv2.imwrite(png_filepath_p, resized_plate)
                                        print(f"  Zapisano tablicę: {png_filepath_p} (pewność: {confidence_plate:.2f})")
                                        if is_live_feed: self.last_plate_save_time = current_time
                                        processed_one_save_this_cycle = True
                                        if not is_live_feed and 'saved_detections_count_ref' in source_details: source_details['saved_detections_count_ref'][0] +=1


                                        json_data_p = {
                                            "detection_type": "plate", "confidence_score": float(f"{confidence_score_plate:.2f}"),
                                            "original_detected_object": {"width": int(w_p), "height": int(h_p)},
                                            "saved_image_details": {"png_filename": png_filename_p, "saved_width": int(saved_w_p), "saved_height": int(saved_h_p), "padding_applied": self.plate_save_padding},
                                            "source_info": source_details if source_details else {"source_type": "live_camera", "timestamp": int(current_time), "camera_index": self.camera_index_used},
                                            "normalization": {"target_width": self.target_plate_width}
                                        }
                                        self._save_detection_json(json_data_p, json_filepath_p)
                                    except Exception as e_save: print(f"  Błąd zapisu tablicy: {e_save}")
                except cv2.error as e_cv_plate:
                     print(f"  Błąd OpenCV (tablica) {'w pliku ' + source_details['original_filename'] if source_details else 'na żywo'}: {e_cv_plate}")

        if is_live_feed:
            text_lines = [f"FPS: {self.current_fps:.1f}", f"{self.camera_name_info}", f"Rozdz: {self.width}x{self.height}"]
            if self.face_detected_in_roi_flag: text_lines.append("TWARZ W ROI!")
            if self.plate_detected_flag: text_lines.append("TABLICA REJ.!")
            
            current_y_text = self.text_y_offset
            for i, line in enumerate(text_lines):
                color = self.font_color_info
                if "TWARZ W ROI!" in line: color = (0, 255, 255) 
                if "TABLICA REJ.!" in line: color = (0, 255, 255) 
                cv2.putText(frame, line, (10, current_y_text + i * self.line_spacing), self.font_face, self.font_scale_info, color, self.line_type_info)
        
        return frame 


    def _process_image_folder_thread_worker(self):
        print("Rozpoczęto wątek przetwarzania folderu obrazów.")
        images_folder_path = "images"
        processed_files_count = 0 
        saved_detections_count_batch = [0] 
        e_batch_to_report = None 

        try:
            image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)
            print(f"Znaleziono {total_files} obrazów do przetworzenia.")

            for file_idx, filename in enumerate(image_files):
                try: 
                    if not self.is_batch_processing: 
                        print("Przetwarzanie wsadowe przerwane.")
                        break
                    
                    image_path = os.path.join(images_folder_path, filename)
                    print(f"\nPrzetwarzanie obrazu ({file_idx+1}/{total_files}): {filename}")
                    
                    frame = cv2.imread(image_path)
                    if frame is None or frame.size == 0: 
                        print(f"  Nie można wczytać obrazu lub obraz jest pusty: {filename}")
                        continue 
                    
                    frame_for_saving = frame.copy()
                    current_image_height, current_image_width = frame.shape[:2]
                    current_processing_time = time.time() 

                    source_details_batch = {
                        "source_type": "image_file", 
                        "original_filename": filename,
                        "original_image_width": current_image_width,
                        "original_image_height": current_image_height,
                        "timestamp": int(current_processing_time), 
                        "camera_index": -1,
                        "saved_detections_count_ref": saved_detections_count_batch 
                    }
                    
                    self.face_detected_in_roi_flag = False 
                    self.plate_detected_flag = False
                    
                    _ = self._process_and_draw_detections(frame, frame_for_saving, current_processing_time, is_live_feed=False, source_details=source_details_batch)
                    
                except Exception as e_file_processing: 
                    print(f"  !! Ogólny błąd podczas przetwarzania pliku {filename}: {e_file_processing}")
                    traceback.print_exc() 
                finally:
                    processed_files_count +=1 
        
        except Exception as e_batch_outer: 
            e_batch_to_report = e_batch_outer 
            print(f"Poważny błąd podczas przetwarzania wsadowego: {e_batch_to_report}")
            traceback.print_exc()
        finally:
            self.is_batch_processing = False
            final_message = f"Zakończono przetwarzanie folderu 'images'.\nPrzetworzono plików: {processed_files_count}.\nZapisano detekcji: {saved_detections_count_batch[0]}." 
            print(final_message)
            
            if hasattr(self, 'window') and self.window.winfo_exists():
                self.window.after(0, lambda: self.camera_menu.entryconfig("Przetwórz folder 'images'", state="normal"))
                if self.available_cameras:
                     for i_cam in self.available_cameras:
                        self.window.after(0, lambda cam_idx=i_cam: self.camera_menu.entryconfig(f"Kamera {cam_idx}", state="normal"))
                
                if e_batch_to_report: 
                     self.window.after(0, lambda err=e_batch_to_report: messagebox.showerror("Błąd przetwarzania", f"Wystąpił błąd: {err}"))
                
                self.window.after(0, lambda: messagebox.showinfo("Zakończono", final_message))

                if self.camera_index_to_resume != -1:
                    print(f"Wznawianie kamery {self.camera_index_to_resume}...")
                    self.window.after(100, lambda idx=self.camera_index_to_resume: self.switch_camera(idx))
                else:
                    if hasattr(self, 'info_label_text'):
                        self.info_label_text.set("Przetwarzanie zakończone. Wybierz kamerę z menu.")
                    if hasattr(self, 'canvas') and self.canvas: 
                        self.canvas.delete("all")
                        self.canvas.config(bg="black") 


    def _update_canvas(self, cv2image_rgb_with_info): 
        if not self.running or not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
            return
        try:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image_rgb_with_info))
            if self.canvas_image_item is None: 
                self.canvas_image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.canvas.itemconfig(self.canvas_image_item, image=self.photo)
            self.canvas.update_idletasks()
            
            info_str_label = f"FPS: {self.current_fps:.1f} | {self.camera_name_info} | Rozdz: {self.width}x{self.height}"
            detection_info = []
            if hasattr(self, 'face_detected_in_roi_flag') and self.face_detected_in_roi_flag:
                detection_info.append("TWARZ W ROI")
            if hasattr(self, 'plate_detected_flag') and self.plate_detected_flag:
                detection_info.append("TABLICA REJ.")
            
            if detection_info:
                info_str_label += " | " + " & ".join(detection_info) + "!"
            self.info_label_text.set(info_str_label)

        except Exception as e:
            if isinstance(e, tk.TclError) and "invalid command name" in str(e):
                print("Próba aktualizacji zniszczonego widgetu Tkinter.")
            else:
                print(f"BŁĄD w _update_canvas: {e}")
                traceback.print_exc()


    def video_capture_loop(self):
        print(f"Rozpoczęto pętlę przechwytywania wideo dla kamery {self.camera_index_used}.")
        
        self.face_detected_in_roi_flag = False
        self.plate_detected_flag = False
        
        try:
            while self.running: 
                if not self.vid or not self.vid.isOpened():
                    if self.running: 
                        if hasattr(self, 'window') and self.window.winfo_exists():
                             self.window.after(0, self.info_label_text.set, f"Kamera {self.camera_index_used} nieaktywna. Wybierz z menu.")
                    time.sleep(0.5) 
                    continue 

                ret, frame = self.vid.read()
                if not ret: 
                    if self.running:
                        print(f"Błąd odczytu klatki z kamery {self.camera_index_used} (ret=False).")
                        if hasattr(self, 'window') and self.window.winfo_exists():
                             self.window.after(0, self.info_label_text.set, f"Błąd odczytu klatki (kamera {self.camera_index_used}).")
                    time.sleep(0.1)
                    continue

                frame_for_saving = frame.copy() 
                current_time_for_saving = time.time()
                
                self.fps_counter += 1
                self.face_detected_in_roi_flag = False 
                self.plate_detected_flag = False


                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time >= 1.0:
                    self.current_fps = self.fps_counter / elapsed_time
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                frame = self._process_and_draw_detections(frame, frame_for_saving, current_time_for_saving, is_live_feed=True)

                cv2image_rgb_with_info = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.running and hasattr(self, 'window') and self.window.winfo_exists():
                    self.window.after(0, self._update_canvas, cv2image_rgb_with_info.copy())
            
            print(f"Pętla przechwytywania wideo dla kamery {self.camera_index_used} zakończona (self.running={self.running}).")

        except Exception as e:
            print(f"Wystąpił krytyczny błąd w wątku kamery {self.camera_index_used}: {e}")
            traceback.print_exc()
            if self.running and hasattr(self, 'window') and self.window.winfo_exists(): 
                 self.window.after(0, messagebox.showerror, "Błąd wątku kamery", f"Krytyczny błąd w wątku kamery: {e}")

    def quit_app(self):
        print("Zamykanie aplikacji...")
        self.running = False 
        self.is_batch_processing = False 
        if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            print("Oczekiwanie na zakończenie wątku kamery przy zamykaniu...")
            self.capture_thread.join(timeout=1.0) 
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened(): 
            print("Zwalnianie kamery przy zamykaniu...")
            self.vid.release()
        if hasattr(self, 'window') and self.window.winfo_exists():
            print("Niszczenie okna głównego...")
            self.window.destroy()
        print("Aplikacja zamknięta.")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraApp(root, "Podgląd kamery - macOS v2.36") 
        root.mainloop()
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd główny: {e}")
        traceback.print_exc()
