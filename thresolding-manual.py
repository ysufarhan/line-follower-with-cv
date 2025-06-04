from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Global variable untuk menyimpan nilai threshold dari trackbar
# Akan diinisialisasi setelah jendela OpenCV dibuat
manual_threshold_value = 100 # Nilai default awal

# Callback function untuk trackbar (harus ada, meskipun kosong)
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val
    # print(f"Threshold value updated to: {manual_threshold_value}") # Opsional: untuk debug

class ErrorFilter:
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

def setup_fuzzy_logic():
    # Definisi universe yang diperluas
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS - Dioptimalkan untuk belokan 90 derajat
    error['NL'] = fuzz.trimf(error.universe, [-200, -180, -80])
    error['NS'] = fuzz.trimf(error.universe, [-120, -50, -10])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [10, 50, 120])
    error['PL'] = fuzz.trimf(error.universe, [80, 180, 200])

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -120, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-70, -30, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 30, 70])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 120, 150])

    output['L']  = fuzz.trimf(output.universe, [-150, -120, -70])
    output['LS'] = fuzz.trimf(output.universe, [-90, -40, -15])
    output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])
    output['RS'] = fuzz.trimf(output.universe, [15, 40, 90])
    output['R']  = fuzz.trimf(output.universe, [70, 120, 150])

    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame, display_mode=False):
    global manual_threshold_value # Akses nilai threshold dari global variable

    roi_start_y = 160
    roi_end_y = 240
    
    if frame.shape[0] < roi_end_y:
        print("[ERROR] Frame terlalu kecil untuk ROI yang ditentukan.")
        return None, None, None
        
    roi_color = frame[roi_start_y:roi_end_y, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # MENGGUNAKAN NILAI THRESHOLD DARI SLIDER
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    if display_mode:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        # Tampilan full frame juga pakai threshold dari slider
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean
    else:
        return None, None, binary_roi_clean

def calculate_line_position(roi_binary):
    M = cv2.moments(roi_binary)
    if M['m00'] > 50:
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + 160
    return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.35):
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    pwm_kiri = max(20, min(85, pwm_kiri))
    pwm_kanan = max(20, min(85, pwm_kanan))
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

def main():
    global manual_threshold_value # Akses nilai threshold dari global variable

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=2) 

    prev_error = 0
    frame_count = 0
    start_time = time.time()

    DISPLAY_GUI = True # WAJIB TRUE untuk menggunakan slider

    if DISPLAY_GUI:
        # Buat jendela untuk ROI dan slider
        cv2.namedWindow("Threshold ROI")
        # Tambahkan trackbar ke jendela "Threshold ROI"
        # Nama Trackbar, Nama Jendela, Nilai Awal, Nilai Maksimum, Fungsi Callback
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        # Jendela untuk tampilan kamera utama (opsional, tapi bagus untuk konteks)
        cv2.namedWindow("Camera View")

    try:
        while True:
            frame = picam2.capture_array()
            
            # Mendapatkan nilai threshold terbaru dari slider
            if DISPLAY_GUI:
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            _, _, roi_binary = process_image(frame, display_mode=DISPLAY_GUI)
            
            if roi_binary is None:
                send_motor_commands(ser, 0, 0)
                if frame_count % 30 == 0:
                    print("[DEBUG] Gagal memproses frame: ROI tidak valid.")
                frame_count += 1
                continue

            line_detected, cx, cy = calculate_line_position(roi_binary)
            
            if line_detected:
                error = cx - 160
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    print(f"[DEBUG] Error: {error:4d}, Delta: {delta_error:4d}, FLC: {kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                cv2.line(frame_for_display, (160, 160), (160, 240), (0, 255, 0), 2)
                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Tampilkan nilai threshold di layar
                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0)
        if ser and ser.is_open:
            ser.close()
        picam2.stop()
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
