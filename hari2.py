from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
manual_threshold_value = 100 # Nilai default awal, sesuaikan setelah tuning pertama

# --- Callback Function untuk Trackbar (Slider) ---
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=3): # Sedikit tingkatkan window_size untuk stabilitas
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Kelas untuk Pemulihan Garis ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang
        self.last_valid_error_direction = 0 # Menyimpan arah error terakhir (+:kanan, -:kiri)
        self.search_speed = 40 # Kecepatan PWM saat berputar mencari garis

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # Tahap 1: Diam sebentar
        if self.lost_count < 10: # Diam selama 10 frame (sekitar 0.1 - 0.2 detik)
            send_motor_commands(ser_instance, 0, 0)
            return "STOP_SEARCH"
        # Tahap 2: Berputar mencari garis
        else:
            # Berputar ke arah terakhir garis terdeteksi (putar di tempat)
            if self.last_valid_error_direction > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
                send_motor_commands(ser_instance, self.search_speed, -self.search_speed)
                return "SEARCH_RIGHT_SPIN"
            else: # Garis terakhir di kiri, putar ke kiri (motor kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed)
                return "SEARCH_LEFT_SPIN"
            
    def line_found(self, current_error):
        self.lost_count = 0
        if abs(current_error) > 5: # Hanya simpan arah jika error cukup signifikan
            self.last_valid_error_direction = current_error # Simpan error terakhir yang valid

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Definisi Universe (Rentang Nilai Input/Output)
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error')  # Error dari -160 ke 160 (lebar frame 320)
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta')  # Perubahan error
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output') # Output kontrol

    # CUSTOM MEMBERSHIP FUNCTIONS
    # Tuning MF error
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])  # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])     # Zero
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])     # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])   # Positive Large

    # Tuning MF delta error
    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])   
    delta['Z']  = fuzz.trimf(delta.universe, [-7, 0, 7])       
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    # Tuning MF output (dibuat lebih agresif di ujung)
    output['L']  = fuzz.trimf(output.universe, [-150, -100, -30]) # Left (large negative)
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   # Left Small
    output['Z']  = fuzz.trimf(output.universe, [-3, 0, 3])       # Zero
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])     # Right Small
    output['R']  = fuzz.trimf(output.universe, [30, 100, 150])   # Right (large positive)

    # Rule Base
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), 
        ctrl.Rule(error['NL'] & delta['NS'], output['L']), 
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']), 
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']), 
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']), 

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']), 
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),   
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']), 
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),   
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),   
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']), 

        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),   
        ctrl.Rule(error['PS'] & delta['PS'], output['Z']), 
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']), 
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
    # Resolusi yang lebih kecil untuk performa lebih cepat
    config = picam2.create_still_configuration(main={"size": (320, 240)}) 
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Beri waktu kamera untuk stabil
    print("[Camera] Camera initialized.")
    return picam2

# --- Setup Komunikasi Serial ---
def setup_serial():
    try:
        # Ganti '/dev/serial0' jika port Anda berbeda
        ser = serial.Serial('/dev/serial0', 115200, timeout=0.1) 
        print("[UART] Serial port opened successfully.")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Failed to open serial port: {e}")
        print("Make sure ESP32 is connected and the serial port is correct.")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame, display_mode=False):
    global manual_threshold_value 

    # ROI yang lebih tinggi untuk 'melihat' belokan lebih awal
    roi_start_y_local = 80 # Disesuaikan dari 120
    roi_end_y_local = 240 
    
    # Cek ukuran frame sebelum memproses
    if frame.shape[0] < roi_end_y_local or frame.shape[1] == 0:
        print("[ERROR] Frame invalid or too small for ROI. Skipping frame processing.")
        return None, None, None, roi_start_y_local, roi_end_y_local 
        
    roi_color = frame[roi_start_y_local:roi_end_y_local, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # Invert binary threshold untuk garis gelap di latar terang
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV) 
    
    # Operasi morfologi untuk membersihkan noise
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) 

    if display_mode:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean, roi_start_y_local, roi_end_y_local 
    else:
        return None, None, binary_roi_clean, roi_start_y_local, roi_end_y_local 

# --- Menghitung Posisi Garis (Centroid) ---
def calculate_line_position(roi_binary, roi_start_y): 
    M = cv2.moments(roi_binary)
    # Filter area untuk memastikan garis terdeteksi (hindari noise kecil)
    if M['m00'] > 100: # Threshold m00 disesuaikan, mungkin perlu tuning
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + roi_start_y # Kembalikan cy relatif terhadap frame asli
    return False, 0, 0

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error): 
    try:
        # Clip input ke universe FLC agar tidak error
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) 
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) 
        fuzzy_ctrl.compute()
        # Clip output ke universe FLC
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        print(f"[FLC ERROR] Fuzzy computation error: {e}")
        return 0.0 # Kembali 0 jika ada masalah

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.12): # Base PWM 50, scaling_factor 0.12 (initial)
    FLC_DEAD_ZONE = 10 # Zona mati untuk kontrol FLC

    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 # Tidak ada koreksi jika kontrol di dalam dead zone
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Batas PWM sesuai kemampuan motor Anda
    MIN_PWM_OUTPUT = 40 # Minimum PWM motor dapat berjalan
    MAX_PWM_OUTPUT = 60 # Maximum PWM untuk belokan yang lebih tajam

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            # Format command: "pwm_kiri,pwm_kanan\n"
            cmd = f"{pwm_kiri},{pwm_kanan}\n" 
            ser.write(cmd.encode())
            ser.flush() # Pastikan data terkirim
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Failed to send data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")
    else:
        #print("[UART] Serial port not open or not initialized.") # Debugging, bisa dinonaktifkan
        pass # Diam saja jika serial tidak siap

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value 

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial() # ser bisa None jika gagal
    error_filter = ErrorFilter(window_size=3) 
    line_recovery_handler = LineRecovery() 

    prev_error = 0 # Error sebelumnya untuk menghitung delta
    frame_count = 0 # Menghitung frame untuk debug output

    DISPLAY_GUI = True # Set False jika tidak ada monitor/SSH X forwarding

    if DISPLAY_GUI:
        cv2.namedWindow("Threshold ROI")
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        cv2.namedWindow("Camera View")

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2 

            if DISPLAY_GUI:
                # Update manual_threshold_value dari trackbar
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            # Pemrosesan citra
            gray_full, binary_full, roi_binary, roi_start_y, roi_end_y = process_image(frame, display_mode=DISPLAY_GUI)
            
            # Cek apakah ROI binary valid (tidak None karena masalah frame/ROI)
            if roi_binary is None:
                # Jika frame tidak valid, anggap garis hilang dan masuk mode pemulihan
                recovery_action = line_recovery_handler.handle_line_lost(ser) 
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan kembali
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Failed to process frame: ROI not valid. Recovery action: {recovery_action}")
                
                # Tetap tampilkan visualisasi jika GUI aktif, gunakan frame asli
                if DISPLAY_GUI:
                    frame_for_display = frame.copy()
                    cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                    cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2)
                    
                    flc_error_z_boundary = 15
                    cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    current_recovery_action_text = "STOP" if line_recovery_handler.lost_count < 10 else ("SPIN_R" if line_recovery_handler.last_valid_error_direction > 0 else "SPIN_L")
                    cv2.putText(frame_for_display, f"ACTION: {current_recovery_action_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.imshow("Camera View", frame_for_display)
                    # Tampilkan dummy kosong jika binary_full atau roi_binary None
                    if binary_full is not None:
                        cv2.imshow("Threshold ROI", binary_full)
                    else:
                        cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue # Lanjut ke iterasi berikutnya tanpa memproses kontrol motor
            
            # Hitung posisi garis
            line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
            
            if line_detected:
                line_recovery_handler.line_found(cx - center_x_frame) # Reset recovery state
                
                error = cx - center_x_frame 
                error = error_filter.filter_error(error) # Filter error
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0: # Cetak debug setiap 10 frame
                    print(f"[DEBUG] Err:{error:4d}, ÎErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                recovery_action = line_recovery_handler.handle_line_lost(ser)
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan kembali
                if frame_count % 20 == 0: # Cetak debug setiap 20 frame jika garis hilang
                    print(f"[DEBUG] Line not detected. Recovery action: {recovery_action}")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Garis tengah acuan (hijau)
                cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                
                # Gambar kotak ROI (biru)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                # Garis bantu indikasi belok (kuning)
                flc_error_z_boundary = 15 # Dari MF FLC 'Z' error
                cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1) # Tampilkan centroid
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"PWM: L{pwm_kiri} R{pwm_kanan}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: # Tampilkan status pemulihan jika garis hilang
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    current_recovery_action_text = "STOP" if line_recovery_handler.lost_count < 10 else ("SPIN_R" if line_recovery_handler.last_valid_error_direction > 0 else "SPIN_L")
                    cv2.putText(frame_for_display, f"ACTION: {current_recovery_action_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary) # Ini seharusnya selalu valid sekarang
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            # Tidak ada time.sleep() di sini, biarkan loop berjalan secepat mungkin
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna.")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti
        if ser and ser.is_open:
            ser.close()
            print("[UART] Serial port closed.")
        picam2.stop()
        print("[Camera] Camera stopped.")
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
            print("[GUI] OpenCV windows closed.")
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
