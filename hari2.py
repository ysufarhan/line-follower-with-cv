from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
manual_threshold_value = 100 

# --- Callback Function untuk Trackbar (Slider) ---
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=5): 
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Definisi Universe (Rentang Nilai Input/Output)
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan)
    # ERROR: Posisi garis relatif terhadap pusat kamera (pusat: 0)
    error['NL'] = fuzz.trimf(error.universe, [-250, -180, -70]) 
    error['NS'] = fuzz.trimf(error.universe, [-90, -40, -15])  
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])     
    error['PS'] = fuzz.trimf(error.universe, [15, 40, 90])     
    error['PL'] = fuzz.trimf(error.universe, [70, 180, 250])   

    # DELTA: Perubahan error antar frame (kecepatan perubahan posisi garis)
    delta['NL'] = fuzz.trimf(delta.universe, [-180, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -20, -8])   
    delta['Z']  = fuzz.trimf(delta.universe, [-8, 0, 8])       
    delta['PS'] = fuzz.trimf(delta.universe, [8, 20, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 180])

    # OUTPUT: Nilai kontrol yang akan digunakan untuk menghitung PWM.
    # DIUBAH SANGAT PENTING UNTUK AGRESIVITAS BELOKAN!
    output['L']  = fuzz.trimf(output.universe, [-150, -150, -50]) # Cepat capai -150
    output['LS'] = fuzz.trimf(output.universe, [-70, -30, -10])  
    output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])       
    output['RS'] = fuzz.trimf(output.universe, [10, 30, 70])     
    output['R']  = fuzz.trimf(output.universe, [50, 150, 150])   # Cepat capai 150

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
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Beri waktu kamera untuk stabil
    print("[Camera] Camera initialized.")
    return picam2

# --- Setup Komunikasi Serial ---
def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1) 
        print("[UART] Serial port opened successfully.")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Failed to open serial port: {e}")
        print("Pastikan ESP32 terhubung dan port serial benar.")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame, display_mode=False):
    global manual_threshold_value 

    roi_start_y = 60 # ROI yang lebih tinggi untuk 'melihat' belokan lebih awal
    roi_end_y = 240
    
    if frame is None or frame.shape[0] < roi_end_y or frame.shape[1] == 0:
        print("[ERROR] Frame invalid or too small for ROI. Skipping frame processing.")
        return None, None, None, roi_start_y, roi_end_y 
        
    roi_color = frame[roi_start_y:roi_end_y, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV) 
    
    kernel = np.ones((5,5), np.uint8) # Kernel sedikit lebih besar
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) 

    if display_mode:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean, roi_start_y, roi_end_y 
    else:
        return None, None, binary_roi_clean, roi_start_y, roi_end_y 

# --- Menghitung Posisi Garis (Centroid) ---
def calculate_line_position(roi_binary, roi_start_y): 
    M = cv2.moments(roi_binary)
    if M['m00'] > 100: 
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + roi_start_y 
    return False, 0, 0

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error): 
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) 
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) 
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        return 0.0 

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.25): # DIUBAH: scaling_factor ditingkatkan
    FLC_DEAD_ZONE = 5 
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    MIN_PWM_OUTPUT = 40 
    MAX_PWM_OUTPUT = 60 

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
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
    else:
        pass 

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value 

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial() 
    error_filter = ErrorFilter(window_size=5) 

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
            
            if roi_binary is None:
                send_motor_commands(ser, 0, 0) 
                prev_error = 0 
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Failed to process frame: ROI not valid. STOPPING.")
                
                if DISPLAY_GUI:
                    frame_for_display = frame.copy()
                    cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                    cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2)
                    
                    flc_error_z_boundary = 15 
                    cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    
                    cv2.putText(frame_for_display, f"STATUS: LINE LOST - STOPPED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.imshow("Camera View", frame_for_display)
                    if binary_full is not None:
                        cv2.imshow("Threshold ROI", binary_full)
                    else:
                        cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue 
            
            line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
            
            if line_detected:
                error = cx - center_x_frame 
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0: 
                    print(f"[DEBUG] Line Detected! Err:{error:4d}, ΔErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0) 
                prev_error = 0 
                if frame_count % 20 == 0: 
                    print(f"[DEBUG] Line NOT Detected. STOPPING.")

            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                cv2.line(frame_for_display, (center_x_frame, 0), (center_x_frame, height), (0, 255, 0), 2)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                flc_error_z_boundary = 15 
                cv2.line(frame_for_display, (center_x_frame - flc_error_z_boundary, roi_start_y), (center_x_frame - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (center_x_frame + flc_error_z_boundary, roi_start_y), (center_x_frame + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1) 
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"PWM: L{pwm_kiri} R{pwm_kanan}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"STATUS: FOLLOWING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else: 
                    cv2.putText(frame_for_display, f"STATUS: LINE LOST - STOPPED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                if binary_full is not None:
                    cv2.imshow("Threshold ROI", binary_full)
                else:
                    cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna.")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) 
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
