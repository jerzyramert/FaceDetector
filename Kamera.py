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

        # --- Sekcja Konfigurowalnych Parametrów ---
        # Parametry zapisu obrazów
        self.target_face_width = 800  # Docelowa szerokość zapisywanej twarzy
        self.target_plate_width = 720 # Docelowa szerokość zapisywanej tablicy
        self.face_save_padding = 50   # Otoczenie w pikselach do zapisu obrazu twarzy
        self.plate_save_padding = 1  # Otoczenie w pikselach do zapisu obrazu tablicy
        self.image_save_interval_seconds = 1.0 # Minimalny odstęp czasu między zapisami tego samego typu obiektu (dla kamery na żywo)

        # Parametry detekcji
        self.min_face_size = (100, 100) # Minimalny rozmiar wykrywanej twarzy (szerokość, wysokość)
        self.min_plate_size = (50, 20)  # Minimalny rozmiar wykrywanej tablicy (szerokość, wysokość)
        self.face_detection_scale_factor = 1.1 # Współczynnik skalowania dla detekcji twarzy
        self.face_detection_min_neighbors = 5  # Minimalna liczba sąsiadów dla detekcji twarzy
        self.plate_detection_scale_factor = 1.1 # Współczynnik skalowania dla detekcji tablic
        self.plate_detection_min_neighbors = 5   # Minimalna liczba sąsiadów dla detekcji tablic
        self.face_confidence_threshold = 5.0  # Minimalna pewność detekcji twarzy do zapisu
        self.plate_confidence_threshold = 1.0 # Minimalna pewność detekcji tablicy do zapisu


        # Parametry ROI (Region of Interest) dla detekcji twarzy
        self.roi_size_percentage = 0.9 # Rozmiar ROI jako procent wymiarów klatki

        # Parametry rysowania na obrazie
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_info = 0.6          # Skala czcionki dla ogólnych informacji
        self.font_scale_confidence = 0.5    # Skala czcionki dla pewności detekcji
        self.font_color_info = (255, 255, 255) # Biały
        self.confidence_text_color = (255, 255, 0) # Jasnoniebieski/Cyan dla pewności
        self.line_type_info = 2
        self.text_y_offset = 25         # Przesunięcie pionowe pierwszej linii tekstu info
        self.line_spacing = 20          # Odstęp między liniami tekstu info
        
        self.rect_roi_color_default = (0, 255, 0)   # Zielony dla ROI
        self.rect_roi_color_face_detected = (255, 0, 0) # Niebieski dla ROI z wykrytą twarzą
        self.rect_roi_thickness = 2
        self.rect_face_color = (0, 0, 255)    # Czerwony dla wykrytej twarzy
        self.rect_face_thickness = 2
        self.rect_plate_color = (0, 255, 255) # Żółty dla wykrytej tablicy
        self.rect_plate_thickness = 2

        # Parametry kamery
        self.max_camera_check_index = 3 # Sprawdź kamery od indeksu 0 do tego (włącznie)
        
        # Status przetwarzania wsadowego
        self.is_batch_processing = False
        self.camera_index_to_resume = -1 # Indeks kamery do wznowienia po batch processingu


        # --- Koniec Sekcji Konfigurowalnych Parametrów ---

        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.camera_name_info = "N/A"
        self.camera_index_used = -1 

        # Throttling for saving images
        self.last_face_save_time = 0
        self.last_plate_save_time = 0

        os.makedirs("faces", exist_ok=True)
        os.makedirs("plates", exist_ok=True)
        os.makedirs("images", exist_ok=True) 
        print("Utworzono/sprawdzono katalogi 'faces', 'plates' i 'images'.")

        # --- Menu Bar ---
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

        self.menubar.add_cascade(label="Opcje", menu=self.camera_menu) 
        self.window.config(menu=self.menubar)

        # --- Initialize Cascades ---
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

        # --- Initialize First Camera ---
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

        if hasattr(self, '_update_canvas'):
            print("DEBUG: __init__ - self HAS _update_canvas")
        else:
            print("DEBUG: __init__ - self DOES NOT HAVE _update_canvas")


        self.running = True 
        if self.vid and self.vid.isOpened(): 
            self.capture_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.info_label_text.set("Brak aktywnej kamery. Wybierz z menu lub podłącz kamerę.")
            
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)

    def _save_detection_json(self, data_dict, json_filepath):
        """Zapisuje dane detekcji do pliku JSON."""
        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False) 
            print(f"Zapisano metadane JSON: {json_filepath}")
        except Exception as e:
            print(f"Błąd zapisu pliku JSON {json_filepath}: {e}")


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
        self.capture_thread = None # Upewnij się, że stary wątek jest 'zapomniany'

        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            print(f"Zwalnianie kamery {self.camera_index_used}...")
            self.vid.release()
            self.vid = None # Jawne ustawienie na None po zwolnieniu

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
        """Rozpoczyna przetwarzanie obrazów z folderu 'images' w osobnym wątku."""
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
        
        if self.running and hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            print("Wstrzymywanie kamery na żywo na czas przetwarzania wsadowego...")
            self.running = False
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                print("Ostrzeżenie: Wątek kamery na żywo nie zakończył się poprawnie przed wsadowym.")
            self.capture_thread = None # Zapomnij o starym wątku
        
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            print(f"Zwalnianie kamery {self.camera_index_used} przed przetwarzaniem wsadowym.")
            self.vid.release()
            self.vid = None
        
        self.camera_index_used = -1 # Brak aktywnej kamery podczas batch
        self._update_camera_info_string()


        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.delete("all") 
            self.canvas.create_text(self.width/2 if self.width > 0 else 320, 
                                   self.height/2 if self.height > 0 else 240, 
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

    def _process_image_folder_thread_worker(self):
        """Metoda robocza dla wątku przetwarzającego obrazy z folderu."""
        print("Rozpoczęto wątek przetwarzania folderu obrazów.")
        images_folder_path = "images"
        processed_files_count = 0 
        saved_detections_count = 0
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

                    # Detekcja twarzy w ROI
                    if self.face_cascade:
                        roi_w_img = int(current_image_width * self.roi_size_percentage)
                        roi_h_img = int(current_image_height * self.roi_size_percentage)
                        
                        if roi_w_img > 0 and roi_h_img > 0: 
                            roi_x1_img = int((current_image_width - roi_w_img) / 2)
                            roi_y1_img = int((current_image_height - roi_h_img) / 2)
                            frame_roi_for_face_detection = frame[roi_y1_img : roi_y1_img + roi_h_img, roi_x1_img : roi_x1_img + roi_w_img]

                            if frame_roi_for_face_detection.size > 0:
                                gray_roi_for_face = cv2.cvtColor(frame_roi_for_face_detection, cv2.COLOR_BGR2GRAY)
                                
                                if gray_roi_for_face.shape[0] >= self.min_face_size[1] and gray_roi_for_face.shape[1] >= self.min_face_size[0]:
                                    try:
                                        faces_in_roi, _, level_weights_face = self.face_cascade.detectMultiScale3(
                                            gray_roi_for_face, scaleFactor=self.face_detection_scale_factor,
                                            minNeighbors=self.face_detection_min_neighbors, minSize=self.min_face_size, outputRejectLevels=True)
                                        
                                        if faces_in_roi is not None and len(faces_in_roi) > 0: 
                                            for i, (x,y,w,h) in enumerate(faces_in_roi):
                                                confidence_score = 0.0
                                                if level_weights_face is not None and i < len(level_weights_face):
                                                     confidence_score = level_weights_face[i] 
                                                
                                                if confidence_score >= self.face_confidence_threshold:
                                                    face_x1_global = roi_x1_img + x; face_y1_global = roi_y1_img + y
                                                    face_x2_global = roi_x1_img + x + w; face_y2_global = roi_y1_img + y + h
                                                    
                                                    fs_x1 = max(0, face_x1_global - self.face_save_padding) 
                                                    fs_y1 = max(0, face_y1_global - self.face_save_padding) 
                                                    fs_x2 = min(current_image_width, face_x2_global + self.face_save_padding) 
                                                    fs_y2 = min(current_image_height, face_y2_global + self.face_save_padding) 
                                                    face_image_to_save = frame_for_saving[fs_y1:fs_y2, fs_x1:fs_x2]

                                                    if face_image_to_save.size > 0:
                                                        orig_h_f, orig_w_f = face_image_to_save.shape[:2]
                                                        resized_face = face_image_to_save 
                                                        saved_w_f, saved_h_f = orig_w_f, orig_h_f
                                                        if orig_w_f > 0 and orig_h_f > 0: 
                                                            ratio_f = self.target_face_width / float(orig_w_f)
                                                            new_face_h_f = int(orig_h_f * ratio_f)
                                                            if new_face_h_f > 0: 
                                                                resized_face = cv2.resize(face_image_to_save, (self.target_face_width, new_face_h_f), interpolation=cv2.INTER_AREA if ratio_f < 1.0 else cv2.INTER_LINEAR)
                                                                saved_w_f, saved_h_f = resized_face.shape[1], resized_face.shape[0]
                                                        
                                                        base_filename, _ = os.path.splitext(filename)
                                                        png_s_filename = f"{base_filename}_face_{i}_{int(current_processing_time)}.png"
                                                        png_s_filepath = os.path.join("faces", png_s_filename)
                                                        json_s_filepath = os.path.join("faces", os.path.splitext(png_s_filename)[0] + ".json")
                                                        
                                                        try:
                                                            cv2.imwrite(png_s_filepath, resized_face)
                                                            print(f"  Zapisano twarz: {png_s_filepath} (pewność: {confidence_score:.2f})")
                                                            saved_detections_count+=1
                                                            json_data_f = {
                                                                "detection_type": "face", "confidence_score": float(f"{confidence_score:.2f}"),
                                                                "original_detected_object": {"width": int(w), "height": int(h)},
                                                                "saved_image_details": {"png_filename": png_s_filename, "saved_width": int(saved_w_f), "saved_height": int(saved_h_f), "padding_applied": self.face_save_padding}, 
                                                                "source_info": {"source_type": "image_file", "original_image_filename": filename, "original_image_width": current_image_width, "original_image_height": current_image_height, "timestamp": int(current_processing_time), "camera_index": -1 },
                                                                "normalization": {"target_width": self.target_face_width}
                                                            }
                                                            self._save_detection_json(json_data_f, json_s_filepath)
                                                        except Exception as e_save: print(f"  Błąd zapisu twarzy z obrazu {filename}: {e_save}")
                                    except cv2.error as e_cv_face_batch:
                                        print(f"  Błąd OpenCV (twarz) podczas przetwarzania pliku {filename}: {e_cv_face_batch}")
                                else:
                                    print(f"  ROI dla detekcji twarzy w {filename} jest zbyt małe. Wymiary ROI: {gray_roi_for_face.shape}, Wymagane min.: {self.min_face_size}")
                        else:
                             print(f"  ROI dla detekcji twarzy w pliku {filename} ma nieprawidłowe wymiary ({roi_w_img}x{roi_h_img}). Pomijanie.")
                    
                    if self.plate_cascade:
                        gray_frame_for_plate_detection = cv2.cvtColor(frame_for_saving, cv2.COLOR_BGR2GRAY)
                        if gray_frame_for_plate_detection.shape[0] >= self.min_plate_size[1] and gray_frame_for_plate_detection.shape[1] >= self.min_plate_size[0]:
                            try:
                                plates, _, level_weights_plate = self.plate_cascade.detectMultiScale3(
                                    gray_frame_for_plate_detection, scaleFactor=self.plate_detection_scale_factor,
                                    minNeighbors=self.plate_detection_min_neighbors, minSize=self.min_plate_size, outputRejectLevels=True)

                                if plates is not None and len(plates) > 0: 
                                    for i, (x_p,y_p,w_p,h_p) in enumerate(plates):
                                        confidence_score_plate = 0.0
                                        if level_weights_plate is not None and i < len(level_weights_plate):
                                            confidence_score_plate = level_weights_plate[i] 
                                        
                                        if confidence_score_plate >= self.plate_confidence_threshold:
                                            ps_x1 = max(0, x_p - self.plate_save_padding) 
                                            ps_y1 = max(0, y_p - self.plate_save_padding) 
                                            ps_x2 = min(current_image_width, x_p + w_p + self.plate_save_padding) 
                                            ps_y2 = min(current_image_height, y_p + h_p + self.plate_save_padding) 
                                            plate_image_to_save = frame_for_saving[ps_y1:ps_y2, ps_x1:ps_x2]

                                            if plate_image_to_save.size > 0:
                                                orig_h_p, orig_w_p = plate_image_to_save.shape[:2]
                                                resized_plate = plate_image_to_save
                                                saved_w_p, saved_h_p = orig_w_p, orig_h_p
                                                if orig_w_p > 0 and orig_h_p > 0:
                                                    ratio_p = self.target_plate_width / float(orig_w_p)
                                                    new_plate_h_p = int(orig_h_p * ratio_p)
                                                    if new_plate_h_p > 0:
                                                        resized_plate = cv2.resize(plate_image_to_save, (self.target_plate_width, new_plate_h_p), interpolation=cv2.INTER_AREA if ratio_p < 1.0 else cv2.INTER_LINEAR)
                                                        saved_w_p, saved_h_p = resized_plate.shape[1], resized_plate.shape[0]

                                                base_filename, _ = os.path.splitext(filename)
                                                png_s_filename_p = f"{base_filename}_plate_{i}_{int(current_processing_time)}.png"
                                                png_s_filepath_p = os.path.join("plates", png_s_filename_p)
                                                json_s_filepath_p = os.path.join("plates", os.path.splitext(png_s_filename_p)[0] + ".json")

                                                try:
                                                    cv2.imwrite(png_s_filepath_p, resized_plate)
                                                    print(f"  Zapisano tablicę: {png_s_filepath_p} (pewność: {confidence_score_plate:.2f})")
                                                    saved_detections_count+=1
                                                    json_data_p = {
                                                        "detection_type": "plate", "confidence_score": float(f"{confidence_score_plate:.2f}"),
                                                        "original_detected_object": {"width": int(w_p), "height": int(h_p)},
                                                        "saved_image_details": {"png_filename": png_s_filename_p, "saved_width": int(saved_w_p), "saved_height": int(saved_h_p), "padding_applied": self.plate_save_padding}, 
                                                        "source_info": {"source_type": "image_file", "original_image_filename": filename, "original_image_width": current_image_width, "original_image_height": current_image_height, "timestamp": int(current_processing_time), "camera_index": -1},
                                                        "normalization": {"target_width": self.target_plate_width}
                                                    }
                                                    self._save_detection_json(json_data_p, json_s_filepath_p)
                                                except Exception as e_save: print(f"  Błąd zapisu tablicy z obrazu {filename}: {e_save}")
                            except cv2.error as e_cv_plate_batch:
                                 print(f"  Błąd OpenCV (tablica) podczas przetwarzania pliku {filename}: {e_cv_plate_batch}")
                        else:
                            print(f"  Obraz {filename} jest zbyt mały dla detekcji tablic. Wymiary: {gray_frame_for_plate_detection.shape}, Wymagane min.: {self.min_plate_size}")
                
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
            final_message = f"Zakończono przetwarzanie folderu 'images'.\nPrzetworzono plików: {processed_files_count}.\nZapisano detekcji: {saved_detections_count}."
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
        if hasattr(self, '_update_canvas'):
            pass 
        else:
            print(f"CRITICAL DEBUG: video_capture_loop - self DOES NOT HAVE _update_canvas AT START. dir(self): {dir(self)}")
        
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

                roi_w = int(self.width * self.roi_size_percentage)
                roi_h = int(self.height * self.roi_size_percentage)
                roi_x1 = int((self.width - roi_w) / 2); roi_y1 = int((self.height - roi_h) / 2)
                roi_x2 = roi_x1 + roi_w; roi_y2 = roi_y1 + roi_h
                
                if self.face_cascade:
                    frame_roi_for_face_detection = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if frame_roi_for_face_detection.size > 0 : 
                        gray_roi_for_face = cv2.cvtColor(frame_roi_for_face_detection, cv2.COLOR_BGR2GRAY)
                        if gray_roi_for_face.shape[0] >= self.min_face_size[1] and gray_roi_for_face.shape[1] >= self.min_face_size[0]:
                            try:
                                faces_in_roi, reject_levels_face, level_weights_face = self.face_cascade.detectMultiScale3(
                                    gray_roi_for_face, 
                                    scaleFactor=self.face_detection_scale_factor, 
                                    minNeighbors=self.face_detection_min_neighbors, 
                                    minSize=self.min_face_size, 
                                    outputRejectLevels = True 
                                )
                                processed_one_face_save_this_cycle = False
                                if faces_in_roi is not None and len(faces_in_roi) > 0: 
                                    for i, (x, y, w, h) in enumerate(faces_in_roi):
                                        self.face_detected_in_roi_flag = True
                                        face_x1_global = roi_x1 + x; face_y1_global = roi_y1 + y
                                        face_x2_global = roi_x1 + x + w; face_y2_global = roi_y1 + y + h
                                        cv2.rectangle(frame, (face_x1_global, face_y1_global), (face_x2_global, face_y2_global), self.rect_face_color, self.rect_face_thickness)
                                        
                                        confidence_score = 0.0
                                        if level_weights_face is not None and i < len(level_weights_face): 
                                            confidence_score = level_weights_face[i] 
                                            cv2.putText(frame, f"{confidence_score:.2f}", (face_x1_global, face_y1_global - 10), self.font_face, self.font_scale_confidence, self.confidence_text_color, self.line_type_info)

                                        if not processed_one_face_save_this_cycle and \
                                           (current_time_for_saving - self.last_face_save_time > self.image_save_interval_seconds) and \
                                           (confidence_score >= self.face_confidence_threshold): 
                                            fs_x1 = max(0, face_x1_global - self.face_save_padding) 
                                            fs_y1 = max(0, face_y1_global - self.face_save_padding) 
                                            fs_x2 = min(self.width, face_x2_global + self.face_save_padding) 
                                            fs_y2 = min(self.height, face_y2_global + self.face_save_padding) 
                                            face_image_to_save = frame_for_saving[fs_y1:fs_y2, fs_x1:fs_x2]

                                            if face_image_to_save.size > 0:
                                                orig_h_f, orig_w_f = face_image_to_save.shape[:2]
                                                resized_face = face_image_to_save 
                                                saved_w_f, saved_h_f = orig_w_f, orig_h_f
                                                if orig_w_f > 0 and orig_h_f > 0: 
                                                    ratio_f = self.target_face_width / float(orig_w_f)
                                                    new_face_h_f = int(orig_h_f * ratio_f)
                                                    if new_face_h_f > 0: 
                                                        resized_face = cv2.resize(face_image_to_save, (self.target_face_width, new_face_h_f), interpolation=cv2.INTER_AREA if ratio_f < 1.0 else cv2.INTER_LINEAR)
                                                        saved_w_f, saved_h_f = resized_face.shape[1], resized_face.shape[0]

                                                    png_filename = f"face_{int(current_time_for_saving)}_{self.camera_index_used}.png"
                                                    png_filepath = os.path.join("faces", png_filename)
                                                    json_filepath = os.path.join("faces", os.path.splitext(png_filename)[0] + ".json")
                                                    
                                                    try:
                                                        cv2.imwrite(png_filepath, resized_face)
                                                        print(f"Zapisano znormalizowaną twarz (pewność {confidence_score:.2f}): {png_filepath}")
                                                        self.last_face_save_time = current_time_for_saving
                                                        processed_one_face_save_this_cycle = True

                                                        json_data = {
                                                            "detection_type": "face",
                                                            "confidence_score": float(f"{confidence_score:.2f}"), 
                                                            "original_detected_object": {"width": int(w), "height": int(h)}, 
                                                            "saved_image_details": {
                                                                "png_filename": png_filename,
                                                                "saved_width": int(saved_w_f), 
                                                                "saved_height": int(saved_h_f), 
                                                                "padding_applied": self.face_save_padding 
                                                            },
                                                            "source_info": {
                                                                "source_type": "live_camera", 
                                                                "timestamp": int(current_time_for_saving),
                                                                "camera_index": self.camera_index_used
                                                            },
                                                            "normalization": {"target_width": self.target_face_width}
                                                        }
                                                        self._save_detection_json(json_data, json_filepath)

                                                    except Exception as e_save: print(f"Błąd zapisu twarzy lub JSON {png_filepath}: {e_save}")
                                                else: print("Błąd: Obliczona nowa wysokość twarzy jest niepoprawna.")
                                            else: print("Błąd: Oryginalne wymiary wyciętej twarzy są niepoprawne.")
                            except cv2.error as e_cv_face:
                                print(f"Błąd OpenCV (twarz) w pętli na żywo: {e_cv_face}")
                    
                current_roi_color_display = self.rect_roi_color_face_detected if self.face_detected_in_roi_flag else self.rect_roi_color_default
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), current_roi_color_display, self.rect_roi_thickness)

                if self.plate_cascade:
                    gray_frame_for_plate_detection = cv2.cvtColor(frame_for_saving, cv2.COLOR_BGR2GRAY) 
                    if gray_frame_for_plate_detection.shape[0] >= self.min_plate_size[1] and gray_frame_for_plate_detection.shape[1] >= self.min_plate_size[0]:
                        try:
                            plates, reject_levels_plate, level_weights_plate = self.plate_cascade.detectMultiScale3(
                                gray_frame_for_plate_detection, 
                                scaleFactor=self.plate_detection_scale_factor, 
                                minNeighbors=self.plate_detection_min_neighbors, 
                                minSize=self.min_plate_size, 
                                outputRejectLevels = True
                            ) 
                            processed_one_plate_save_this_cycle = False
                            if plates is not None and len(plates) > 0: 
                                for i, (x_p, y_p, w_p, h_p) in enumerate(plates):
                                    self.plate_detected_flag = True
                                    cv2.rectangle(frame,(x_p,y_p),(x_p+w_p,y_p+h_p), self.rect_plate_color, self.rect_plate_thickness)
                                    
                                    confidence_score_plate = 0.0
                                    if level_weights_plate is not None and i < len(level_weights_plate): 
                                        confidence_score_plate = level_weights_plate[i] 
                                        cv2.putText(frame, f"{confidence_score_plate:.2f}", (x_p, y_p - 10), self.font_face, self.font_scale_confidence, self.confidence_text_color, self.line_type_info)

                                    if not processed_one_plate_save_this_cycle and \
                                       (current_time_for_saving - self.last_plate_save_time > self.image_save_interval_seconds) and \
                                       (confidence_score_plate >= self.plate_confidence_threshold): 
                                        ps_x1 = max(0, x_p - self.plate_save_padding) 
                                        ps_y1 = max(0, y_p - self.plate_save_padding) 
                                        ps_x2 = min(self.width, x_p + w_p + self.plate_save_padding) 
                                        ps_y2 = min(self.height, y_p + h_p + self.plate_save_padding) 
                                        plate_image_to_save = frame_for_saving[ps_y1:ps_y2, ps_x1:ps_x2]

                                        if plate_image_to_save.size > 0:
                                            orig_h_p, orig_w_p = plate_image_to_save.shape[:2]
                                            resized_plate = plate_image_to_save 
                                            saved_w_p, saved_h_p = orig_w_p, orig_h_p
                                            if orig_w_p > 0 and orig_h_p > 0: 
                                                ratio_p = self.target_plate_width / float(orig_w_p)
                                                new_plate_h_p = int(orig_h_p * ratio_p)
                                                if new_plate_h_p > 0: 
                                                    resized_plate = cv2.resize(plate_image_to_save, (self.target_plate_width, new_plate_h_p), interpolation=cv2.INTER_AREA if ratio_p < 1.0 else cv2.INTER_LINEAR)
                                                    saved_w_p, saved_h_p = resized_plate.shape[1], resized_plate.shape[0]
                                                
                                                png_filename = f"plate_{int(current_time_for_saving)}_{self.camera_index_used}.png"
                                                png_filepath = os.path.join("plates", png_filename)
                                                json_filepath = os.path.join("plates", os.path.splitext(png_filename)[0] + ".json")

                                                try:
                                                    cv2.imwrite(png_filepath, resized_plate)
                                                    print(f"Zapisano znormalizowaną tablicę (pewność {confidence_score_plate:.2f}): {png_filepath}")
                                                    self.last_plate_save_time = current_time_for_saving
                                                    processed_one_plate_save_this_cycle = True

                                                    json_data = {
                                                        "detection_type": "plate",
                                                        "confidence_score": float(f"{confidence_score_plate:.2f}"), 
                                                        "original_detected_object": {"width": int(w_p), "height": int(h_p)}, 
                                                        "saved_image_details": {
                                                            "png_filename": png_filename,
                                                            "saved_width": int(saved_w_p), 
                                                            "saved_height": int(saved_h_p), 
                                                            "padding_applied": self.plate_save_padding 
                                                        },
                                                        "source_info": {
                                                            "source_type": "live_camera", 
                                                            "timestamp": int(current_time_for_saving),
                                                            "camera_index": self.camera_index_used
                                                        },
                                                        "normalization": {"target_width": self.target_plate_width}
                                                    }
                                                    self._save_detection_json(json_data, json_filepath)

                                                except Exception as e_save: print(f"Błąd zapisu tablicy lub JSON {png_filepath}: {e_save}")
                                            else: print("Błąd: Obliczona nowa wysokość tablicy jest niepoprawna.")
                                        else: print("Błąd: Oryginalne wymiary wyciętej tablicy są niepoprawne.")
                        except cv2.error as e_cv_plate:
                            print(f"Błąd OpenCV (tablica) w pętli na żywo: {e_cv_plate}")


                text_lines = [f"FPS: {self.current_fps:.1f}", f"{self.camera_name_info}", f"Rozdz: {self.width}x{self.height}"]
                if self.face_detected_in_roi_flag: text_lines.append("TWARZ W ROI!")
                if self.plate_detected_flag: text_lines.append("TABLICA REJ.!")
                
                current_y_text = self.text_y_offset
                for i, line in enumerate(text_lines):
                    color = self.font_color_info
                    if "TWARZ W ROI!" in line: color = (0, 255, 255) 
                    if "TABLICA REJ.!" in line: color = (0, 255, 255) 
                    cv2.putText(frame, line, (10, current_y_text + i * self.line_spacing), self.font_face, self.font_scale_info, color, self.line_type_info)

                cv2image_rgb_with_info = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if not hasattr(self, '_update_canvas'):
                    print(f"CRITICAL DEBUG: Right before self.window.after, _update_canvas is MISSING on self!")
                elif not callable(self._update_canvas):
                     print(f"CRITICAL DEBUG: Right before self.window.after, _update_canvas is NOT CALLABLE! It is: {self._update_canvas}")


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
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            print("Oczekiwanie na zakończenie wątku kamery przy zamykaniu...")
            self.capture_thread.join(timeout=1.0) 
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened(): # Dodano sprawdzenie self.vid
            print("Zwalnianie kamery przy zamykaniu...")
            self.vid.release()
        if hasattr(self, 'window') and self.window.winfo_exists():
            print("Niszczenie okna głównego...")
            self.window.destroy()
        print("Aplikacja zamknięta.")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraApp(root, "Podgląd kamery - macOS v2.31") 
        root.mainloop()
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd główny: {e}")
        traceback.print_exc()
