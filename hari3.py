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

# --- Kelas untuk Pemulihan Garis (Baru Ditambahkan/Disesuaikan) ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang
        self.last_valid_error_direction = 0 # Menyimpan arah error terakhir (+:kanan, -:kiri)
        self.search_speed = 40 # Kecepatan PWM saat berputar mencari garis
        self.max_lost_frames_for_spin = 100 # Batasan berapa lama akan berputar di tempat sebelum mungkin berhenti total (jika terlalu lama hilang)

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # HILANGKAN TAHAP 1 (DIAM SEBENTAR)
        # Langsung masuk ke Tahap 2: Berputar mencari garis
        if self.last_valid_error_direction > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
            send_motor_commands(ser_instance, self.search_speed, -self.search_speed)
            return "SEARCH_RIGHT_SPIN"
        else: # Garis terakhir di kiri (atau belum ada arah valid), putar ke kiri (motor kiri mundur, kanan maju)
            # Jika belum ada arah valid, default ke salah satu arah, misal kiri
            if self.last_valid_error_direction == 0: 
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed) # Default ke kiri
                return "SEARCH_LEFT_SPIN_DEFAULT"
            else:
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed)
                return "SEARCH_LEFT_SPIN"
        
    def line_found(self, current_error):
        self.lost_count = 0
        # Simpan arah error terakhir yang valid, bahkan jika errornya kecil, 
        # asalkan garis terdeteksi. Ini membantu saat garis hilang tiba-tiba di jalur lurus.
        # if abs(current_error) > 5: # Hapus kondisi ini agar selalu update arah
        self.last_valid_error_direction = current_error # Simpan error terakhir yang valid

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Definisi Universe (Rentang Nilai Input/Output)
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error') 
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta') 
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output') 

    # CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan)
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) 
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10]) 
    error['Z'] = fuzz.trimf(error.universe, [-15, 0, 15])     
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])    
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])  

    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])   
    delta['Z'] = fuzz.trimf(delta.universe, [-7, 0, 7])       
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    output['L'] = fuzz.trimf(output.universe, [-150, -100, -50]) 
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   
    output['Z'] = fuzz.trimf(output.universe, [-3, 0, 3])      
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])    
    output['R'] = fuzz.trimf(output.universe, [50, 100, 150])   

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

# --- Raspberry Pi Camera Setup ---
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Give camera time to stabilize
    return picam2

# --- Serial Communication Setup ---
def setup_serial():
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=0.1) 
        print("[UART] Serial port opened successfully")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Failed to open serial port: {e}")
        print("Make sure ESP32 is connected and the serial port is correct.")
        return None

# --- Image Processing Using OpenCV ---
def process_image(frame, display_mode=False):
    global manual_threshold_value 

    # Inisialisasi ROI values to be returned (default)
    roi_start_y_local = 120 
    roi_end_y_local = 240 
    
    if frame.shape[0] < roi_end_y_local:
        print("[ERROR] Frame is too small for the defined ROI. Using default ROI for visualization.")
        return None, None, None, roi_start_y_local, roi_end_y_local 
        
    roi_color = frame[roi_start_y_local:roi_end_y_local, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
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

# --- Calculate Line Position ---
def calculate_line_position(roi_binary, roi_start_y): 
    M = cv2.moments(roi_binary)
    if M['m00'] > 100: 
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + roi_start_y 
    return False, 0, 0

# --- Compute Fuzzy Control Output ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error): 
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) 
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) 
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        return 0.0

# --- Calculate Motor PWM Values ---
def calculate_motor_pwm(kontrol, base_pwm=45, scaling_factor=0.08): 
    FLC_DEAD_ZONE = 10 
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    MIN_PWM_OUTPUT = 35 
    MAX_PWM_OUTPUT = 55 

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Send Motor Commands via Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush() 
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Failed to send data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value 

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3) 
    line_recovery_handler = LineRecovery() 

    prev_error = 0
    frame_count = 0

    DISPLAY_GUI = True 

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
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            gray_full, binary_full, roi_binary, roi_start_y, roi_end_y = process_image(frame, display_mode=DISPLAY_GUI)
            
            # --- START Logic Update for Line Detection and Recovery ---
            if roi_binary is None or cv2.countNonZero(roi_binary) < 50: # Tambahkan cek countNonZero untuk deteksi garis yang lebih robust
                # Garis tidak terdeteksi atau ROI tidak valid
                recovery_action = line_recovery_handler.handle_line_lost(ser)
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Garis tidak terdeteksi atau ROI tidak valid. Aksi pemulihan: {recovery_action}")
            else:
                # Garis terdeteksi
                line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
                if line_detected:
                    line_recovery_handler.line_found(cx - center_x_frame) # Reset recovery state
                    
                    error = cx - center_x_frame 
                    error = error_filter.filter_error(error)
                    delta_error = error - prev_error
                    prev_error = error

                    kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                    pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                    send_motor_commands(ser, pwm_kiri, pwm_kanan)

                    if frame_count % 10 == 0:
                        print(f"[DEBUG] Err:{error:4d}, ΔErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
                else:
                    # Garis tidak terdeteksi meski ROI valid (mungkin hanya ada sedikit noise)
                    recovery_action = line_recovery_handler.handle_line_lost(ser)
                    prev_error = 0 # Reset error
                    if frame_count % 20 == 0:
                        print(f"[DEBUG] Garis tidak terdeteksi (moment issue). Aksi pemulihan: {recovery_action}")

            # --- END Logic Update for Line Detection and Recovery ---

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Garis tengah acuan (hijau)
                cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                
                # Gambar kotak ROI (biru)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                # Garis bantu indikasi belok (kuning)
                flc_error_z_boundary = 15 
                cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                # Update tampilan teks berdasarkan status deteksi garis
                if 'line_detected' in locals() and line_detected: # Periksa apakah line_detected terdefinisi dan True
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: # Tampilkan status pemulihan jika garis hilang atau tidak terdeteksi
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # Perbaikan display recovery action
                    display_recovery_action = line_recovery_handler.handle_line_lost(None) # Panggil hanya untuk mendapatkan string aksi
                    cv2.putText(frame_for_display, f"ACTION: {display_recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                # Pastikan binary_full valid sebelum ditampilkan, jika tidak, tampilkan roi_binary
                if binary_full is not None:
                    cv2.imshow("Threshold ROI", binary_full)
                elif roi_binary is not None: # Tampilkan ROI binary jika full binary tidak tersedia
                    cv2.imshow("Threshold ROI", roi_binary)
                else: # Jika keduanya None, tampilkan dummy kosong
                    try:
                        cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                    except Exception as e:
                        print(f"[VISUALIZATION ERROR] Could not display dummy ROI: {e}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti saat program berakhir
        if ser and ser.is_open:
            ser.close()
        picam2.stop()
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
