from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
# Nilai default awal, sesuaikan setelah tuning pertama menggunakan slider.
manual_threshold_value = 100 

# --- Callback Function untuk Trackbar (Slider) ---
# Fungsi ini akan dipanggil setiap kali slider Threshold digeser.
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
# Membantu menghaluskan nilai error dari kamera, mengurangi 'jitter' robot.
class ErrorFilter:
    def __init__(self, window_size=3): # Ukuran jendela rata-rata untuk stabilitas.
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Kelas untuk Pemulihan Garis (Strategi "Hold Last Direction") ---
# Mengatur perilaku robot ketika garis hilang dari pandangan kamera.
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang.
        self.last_valid_error_direction = 0 # Menyimpan arah error terakhir. Positif: kanan, Negatif: kiri.
        self.search_speed = 40 # Kecepatan PWM saat berputar mencari garis.
        self.hold_speed_base_pwm = 40 # Kecepatan dasar saat mode "hold direction" (maju).
        self.hold_direction_factor = 0.5 # Seberapa agresif belok saat mode "hold direction".

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # Tahap 1: Coba Lanjutkan dengan arah terakhir yang diketahui (maju sambil belok)
        # Selama lost_count masih di bawah ambang batas (misal 15 frame), robot akan mencoba 'menebak' arah garis.
        if self.lost_count < 15: 
            # Menghitung PWM berdasarkan arah error terakhir.
            if self.last_valid_error_direction > 0: # Garis terakhir di kanan, belok ke kiri (motor kiri > kanan)
                pwm_kiri_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
            elif self.last_valid_error_direction < 0: # Garis terakhir di kiri, belok ke kanan (motor kiri < kanan)
                pwm_kiri_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
            else: # Terakhir lurus, coba lurus terus
                pwm_kiri_rec = self.hold_speed_base_pwm
                pwm_kanan_rec = self.hold_speed_base_pwm
            
            # Memastikan nilai PWM tidak keluar batas aman.
            pwm_kiri_rec = max(20, min(85, pwm_kiri_rec))
            pwm_kanan_rec = max(20, min(85, pwm_kanan_rec))

            send_motor_commands(ser_instance, int(pwm_kiri_rec), int(pwm_kanan_rec))
            return f"HOLD_DIR ({int(pwm_kiri_rec)},{int(pwm_kanan_rec)})"
        
        # Tahap 2: Jika garis masih hilang setelah batas 'hold direction', beralih ke strategi putar di tempat (fallback)
        else:
            if self.last_valid_error_direction > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
                send_motor_commands(ser_instance, self.search_speed, -self.search_speed)
                return "SEARCH_RIGHT_SPIN"
            else: # Garis terakhir di kiri, putar ke kiri (motor kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed)
                return "SEARCH_LEFT_SPIN"
        
    def line_found(self, current_error):
        self.lost_count = 0 # Reset hitungan frame hilang.
        # Hanya simpan arah error jika cukup signifikan, hindari menyimpan 'noise' saat hampir lurus.
        if abs(current_error) > 5: 
            self.last_valid_error_direction = current_error 
        else: # Jika error sangat kecil, anggap arahnya lurus.
            self.last_valid_error_direction = 0 

# --- Setup Logika Fuzzy Control (FLC) ---
# Mendefinisikan variabel input (antecedents) dan output (consequents) serta aturan fuzzy.
def setup_fuzzy_logic():
    # Rentang nilai input error dan delta (sesuai lebar frame dan perubahan error).
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error') 
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta') 
    # Rentang nilai output kontrol dari FLC.
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output') 

    # --- CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan) ---
    # Mendefinisikan 'fuzzy set' untuk setiap variabel.

    # ERROR: Posisi garis relatif terhadap pusat kamera (pusat: 0).
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])  # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])     # Zero (Zona Lurus yang stabil, diperlebar)
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])     # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])   # Positive Large

    # DELTA: Perubahan error antar frame (kecepatan error).
    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])   # Dikurangi sensitivitas terhadap noise.
    delta['Z']  = fuzz.trimf(delta.universe, [-7, 0, 7])       # Dikurangi sensitivitas terhadap noise.
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    # OUTPUT: Nilai kontrol yang akan mengubah PWM motor.
    output['L']  = fuzz.trimf(output.universe, [-150, -100, -50]) # Full Left
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   # Light-Medium Left
    output['Z']  = fuzz.trimf(output.universe, [-3, 0, 3])      # Straight (SANGAT SEMPIT untuk lurus sempurna)
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])     # Light-Medium Right
    output['R']  = fuzz.trimf(output.universe, [50, 100, 150])   # Full Right

    # --- Rule Base (Aturan Fuzzy) ---
    # Menentukan bagaimana kombinasi error dan delta menghasilkan output kontrol.
    rules = [
        # Error Negative Large (Garis di kiri jauh dari pusat) -> Belok Kanan Kuat
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), 
        ctrl.Rule(error['NL'] & delta['NS'], output['L']), 
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']), 
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']), # Pencegah Overshoot (mulai luruskan)
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']), # Pencegah Overshoot

        # Error Negative Small (Garis di kiri sedikit dari pusat) -> Belok Kanan Ringan
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']), # Mulai Luruskan
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),  # Target Lurus
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        # Error Zero (Garis di tengah) -> Prioritas Utama: Lurus Sempurna
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']), # Koreksi Halus
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),  
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),   # IDEAL: Lurus Sempurna
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),  
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']), # Koreksi Halus

        # Error Positive Small (Garis di kanan sedikit dari pusat) -> Belok Kiri Ringan
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),  # Target Lurus
        ctrl.Rule(error['PS'] & delta['PS'], output['Z']), # Mulai Luruskan
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        # Error Positive Large (Garis di kanan jauh dari pusat) -> Belok Kiri Kuat
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']), # Pencegah Overshoot
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']), 
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']), 
        ctrl.Rule(error['PL'] & delta['PL'], output['R']), 
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera Raspberry Pi ---
def setup_camera():
    picam2 = Picamera2()
    # Resolusi 320x240 adalah pilihan yang baik untuk Pi 4, fokus pada framerate.
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Beri waktu kamera untuk stabil.
    return picam2

# --- Setup Komunikasi Serial ---
# Menghubungkan Raspberry Pi dengan ESP32.
def setup_serial():
    try:
        # Sesuaikan port serial: '/dev/ttyAMA0' atau '/dev/ttyS0'.
        # Pastikan sudah diaktifkan di raspi-config.
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1) 
        print("[UART] Port serial berhasil dibuka.")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        print("Pastikan ESP32 terhubung dan port serial benar.")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
# Mengambil frame, mengubahnya menjadi gambar biner, dan menentukan Region of Interest (ROI).
def process_image(frame, display_mode=False):
    global manual_threshold_value 

    # Inisialisasi nilai ROI yang akan dikembalikan (default).
    # Ini penting agar roi_start_y dan roi_end_y selalu terdefinisi untuk visualisasi.
    roi_start_y_local = 120 # ROI dimulai dari baris 120 (melihat lebih jauh ke depan).
    roi_end_y_local = 240   # ROI berakhir di baris 240 (bawah frame).
    
    # Cek ukuran frame untuk memastikan valid.
    if frame.shape[0] < roi_end_y_local or frame.shape[1] == 0: # Tambah cek lebar frame juga
        print("[ERROR] Frame tidak valid atau terlalu kecil untuk ROI. Menggunakan default ROI untuk visualisasi.")
        return None, None, None, roi_start_y_local, roi_end_y_local 
        
    roi_color = frame[roi_start_y_local:roi_end_y_local, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # Menerapkan Fixed Thresholding menggunakan nilai dari slider.
    # THRESH_BINARY_INV: Piksel di bawah threshold jadi putih, di atas jadi hitam (untuk garis hitam di latar terang).
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Operasi morfologi untuk membersihkan gambar biner dari noise.
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) 

    if display_mode:
        # Konversi full frame untuk visualisasi.
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean, roi_start_y_local, roi_end_y_local 
    else:
        return None, None, binary_roi_clean, roi_start_y_local, roi_end_y_local 

# --- Menghitung Posisi Garis ---
# Menggunakan momen gambar untuk menemukan pusat garis (cx, cy).
def calculate_line_position(roi_binary, roi_start_y): 
    M = cv2.moments(roi_binary)
    # Ambang batas M['m00'] untuk mendeteksi garis yang valid (bukan noise).
    if M['m00'] > 100: 
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        # cx dan cy dikonversi ke koordinat absolut frame.
        return True, cx, cy_roi + roi_start_y 
    return False, 0, 0

# --- Menghitung Output Kontrol Fuzzy ---
# Menjalankan inferensi FLC berdasarkan error dan delta_error.
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error): 
    try:
        # Memastikan input dalam rentang universe FLC.
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) 
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) 
        fuzzy_ctrl.compute()
        # Memastikan output dalam rentang yang valid.
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        # Jika ada error FLC, kembalikan 0 (robot diam atau lurus).
        return 0.0

# --- Menghitung Nilai PWM Motor ---
# Mengkonversi output FLC menjadi nilai PWM untuk motor kiri dan kanan.
def calculate_motor_pwm(kontrol, base_pwm=45, scaling_factor=0.08): 
    # FLC Dead Zone: Jika output kontrol sangat kecil, robot dianggap harus lurus.
    # Membantu stabilitas di garis lurus, mencegah zigzag kecil.
    FLC_DEAD_ZONE = 10 
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 # Tidak ada koreksi, PWM kiri = kanan.
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Batas nilai PWM agar tidak melebihi rentang aman/efektif motor.
    MIN_PWM_OUTPUT = 35 
    MAX_PWM_OUTPUT = 55 

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
# Mengirim nilai PWM ke ESP32 melalui komunikasi serial.
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open: # Memastikan port serial terbuka.
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n" # Format perintah: "PWMKiri,PWMKanan\n".
            ser.write(cmd.encode()) # Mengirim data sebagai byte.
            ser.flush() # Memastikan semua data terkirim.
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

# --- Fungsi Utama Program (Loop Kontrol) ---
def main():
    global manual_threshold_value 

    # Inisialisasi semua sistem: FLC, Kamera, Serial, Filter Error, Pemulihan Garis.
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3) 
    line_recovery_handler = LineRecovery() 

    prev_error = 0 # Error dari frame sebelumnya.
    frame_count = 0 # Penghitung frame untuk debug.

    DISPLAY_GUI = True # Set ke False untuk performa maksimal (headless mode).

    # Setup jendela dan slider OpenCV jika GUI aktif.
    if DISPLAY_GUI:
        cv2.namedWindow("Threshold ROI")
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        cv2.namedWindow("Camera View")

    try:
        while True:
            # Mengambil frame dari kamera.
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2 

            # Mengambil nilai threshold terbaru dari slider jika GUI aktif.
            if DISPLAY_GUI:
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            # Memproses gambar untuk mendapatkan ROI biner.
            # roi_start_y dan roi_end_y juga dikembalikan untuk visualisasi.
            gray_full, binary_full, roi_binary, roi_start_y, roi_end_y = process_image(frame, display_mode=DISPLAY_GUI)
            
            # --- Penanganan Frame Tidak Valid atau Garis Hilang ---
            if roi_binary is None: # Ini terjadi jika frame dari kamera bermasalah atau ROI tidak valid.
                # Panggil logika pemulihan garis.
                recovery_action = line_recovery_handler.handle_line_lost(ser) 
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan kembali.
                
                if frame_count % 30 == 0: # Cetak debug setiap 30 frame.
                    print(f"[DEBUG] Gagal memproses frame: ROI tidak valid. Aksi pemulihan: {recovery_action}")
                frame_count += 1
                
                # Menampilkan visualisasi bahkan saat ada masalah frame.
                if DISPLAY_GUI:
                    frame_for_display = frame.copy()
                    cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                    cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 
                    flc_error_z_boundary = 15
                    cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    current_recovery_action = "STOP_SEARCH" if line_recovery_handler.lost_count < 10 else ("SEARCH_RIGHT_SPIN" if line_recovery_handler.last_valid_error_direction > 0 else "SEARCH_LEFT_SPIN")
                    cv2.putText(frame_for_display, f"ACTION: {current_recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.imshow("Camera View", frame_for_display)
                    if binary_full is not None:
                        cv2.imshow("Threshold ROI", binary_full)
                    else:
                        # Jika binary_full None, tampilkan dummy kosong agar jendela tidak crash.
                        try:
                            cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                        except Exception as e:
                            print(f"[VISUALIZATION ERROR] Could not display dummy ROI: {e}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue # Lanjutkan ke iterasi berikutnya tanpa memproses lebih lanjut.
            
            # --- Deteksi Garis dan Kontrol Motor ---
            line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
            
            if line_detected:
                line_recovery_handler.line_found(cx - center_x_frame) # Reset recovery state & simpan arah.
                
                error = cx - center_x_frame # Hitung error (penyimpangan dari pusat frame).
                error = error_filter.filter_error(error) # Haluskan error.
                delta_error = error - prev_error # Hitung perubahan error.
                prev_error = error # Simpan error saat ini untuk frame berikutnya.

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error) # Hitung output FLC.
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol) # Hitung PWM motor.
                send_motor_commands(ser, pwm_kiri, pwm_kanan) # Kirim perintah ke ESP32.

                if frame_count % 10 == 0: # Debug print ke konsol setiap 10 frame.
                    print(f"[DEBUG] Err:{error:4d}, ΔErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else: # Garis tidak terdeteksi dalam frame saat ini.
                recovery_action = line_recovery_handler.handle_line_lost(ser) # Panggil logika pemulihan.
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan.
                if frame_count % 20 == 0: # Debug print setiap 20 frame saat garis hilang.
                    print(f"[DEBUG] Garis tidak terdeteksi. Aksi pemulihan: {recovery_action}")

            # --- Bagian Tampilan GUI (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Gambar garis tengah acuan (hijau) di frame.
                cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                
                # Gambar kotak ROI (biru).
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                # Gambar garis bantu indikasi belok (kuning).
                flc_error_z_boundary = 15 # Sesuaikan dengan batas 'Z' pada error FLC.
                cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: # Tampilkan status pemulihan jika garis hilang.
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    current_recovery_action = "STOP_SEARCH" if line_recovery_handler.lost_count < 10 else ("SEARCH_RIGHT_SPIN" if line_recovery_handler.last_valid_error_direction > 0 else "SEARCH_LEFT_SPIN")
                    cv2.putText(frame_for_display, f"ACTION: {current_recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary) 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            # Loop berjalan secepat mungkin, tanpa time.sleep() tambahan.
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna.")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti saat program berakhir.
        if ser and ser.is_open:
            ser.close() # Tutup port serial.
        picam2.stop() # Hentikan kamera.
        if DISPLAY_GUI:
            cv2.destroyAllWindows() # Tutup semua jendela OpenCV.
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
