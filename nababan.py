from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Konfigurasi Global ---
SERIAL_PORT = '/dev/ttyS0' 
BAUD_RATE = 115200

BASE_PWM = 45 
SCALING_FACTOR = 0.08 

MIN_PWM_OUTPUT = 25
MAX_PWM_OUTPUT = 65

FLC_DEAD_ZONE_ERROR = 15 

# --- KONFIGURASI BARU UNTUK KICK-START ---
INITIAL_PUSH_PWM = 50 # Nilai PWM untuk dorongan awal (lebih tinggi dari BASE_PWM untuk mengatasi static friction)
INITIAL_PUSH_DURATION = 1.0 # Durasi dorongan awal dalam detik (sesuaikan jika perlu)

# --- Kelas untuk Filter Error ---
class ErrorFilter:
    def __init__(self, window_size=3, alpha=0.7):
        self.window_size = window_size
        self.error_history = []
        self.alpha = alpha
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors)//2]
        
        if len(self.error_history) > 1:
            prev_median_history = self.error_history[:-1]
            prev_median = sorted(prev_median_history)[len(prev_median_history)//2] if len(prev_median_history) > 0 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_median
        else:
            smoothed_error = median_error
        
        return int(smoothed_error)

# --- Kelas untuk Pemulihan Garis ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.recovery_speed = 40

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        if self.lost_count < 10: 
            send_motor_commands(ser_instance, 0, 0)
            return "STOP"
        elif self.lost_count < 30: 
            send_motor_commands(ser_instance, -30, -30)
            return "REVERSE"
        else:
            if self.last_valid_error > 0: 
                send_motor_commands(ser_instance, self.recovery_speed, -self.recovery_speed)
                return "SEARCH_RIGHT"
            else: 
                send_motor_commands(ser_instance, -self.recovery_speed, self.recovery_speed)
                return "SEARCH_LEFT"
    
    def line_found(self, current_error):
        self.lost_count = 0
        self.last_valid_error = current_error

# --- Setup Fuzzy Logic Control (FLC) ---
def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    error['NL'] = fuzz.trimf(error.universe, [-350, -200, -60])
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])
    error['PL'] = fuzz.trimf(error.universe, [60, 200, 350])

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-5, 0, 5])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 150])

    output['L']  = fuzz.trimf(output.universe, [-100, -70, -40])
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -8])
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 50])
    output['R']  = fuzz.trimf(output.universe, [40, 70, 100])

    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),

        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
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
    config = picam2.create_still_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    picam2.start()
    return picam2

# --- Setup Komunikasi Serial dengan ESP32 ---
def setup_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
        print(f"[UART] Port serial {SERIAL_PORT} berhasil dibuka.")
        return ser
    except serial.SerialException as e:
        print(f"[UART ERROR] Gagal membuka serial port {SERIAL_PORT}: {e}")
        print("Pastikan ESP32 terhubung, port serial benar, dan tidak ada program lain yang menggunakannya.")
        print("Coba cek: 'ls /dev/tty*' dan 'sudo raspi-config' -> Interface Options -> P6 Serial Port.")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] Gagal membuka serial port: {e}")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    height, width = binary.shape
    roi_start = int(height * 0.3)
    roi_end = int(height * 0.95)
    roi_left = int(width * 0.05)
    roi_right = int(width * 0.95)
    
    roi_start = max(0, min(roi_start, height - 1))
    roi_end = max(0, min(roi_end, height))
    roi_left = max(0, min(roi_left, width - 1))
    roi_right = max(0, min(roi_right, width))

    roi = binary[roi_start:roi_end, roi_left:roi_right]
    return gray, binary, roi, roi_start, roi_left, roi_end, roi_right

# --- Menghitung Posisi Garis ---
def calculate_line_position(roi_image, roi_start_y, roi_start_x, frame_width):
    kernel = np.ones((5,5), np.uint8) 
    roi_clean = cv2.morphologyEx(roi_image, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 300: # Tune this value
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + roi_start_x
                cy = int(M['m01'] / M['m00']) + roi_start_y
                return True, cx, cy, largest_contour
    return False, 0, 0, None

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl_system, error_value, delta_error_value):
    try:
        fuzzy_ctrl_system.input['error'] = np.clip(error_value, -350, 350)
        fuzzy_ctrl_system.input['delta'] = np.clip(delta_error_value, -150, 150)
        fuzzy_ctrl_system.compute()
        return np.clip(fuzzy_ctrl_system.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}. Input Error: {error_value}, Delta Error: {delta_error_value}")
        return 0.0

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(control_output, base_pwm, scaling_factor):
    if abs(control_output) < FLC_DEAD_ZONE_ERROR: 
        control_scaled = 0
    else:
        control_scaled = control_output * scaling_factor

    pwm_kiri = base_pwm + control_scaled
    pwm_kanan = base_pwm - control_scaled

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))

    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser_instance, pwm_kiri, pwm_kanan):
    if ser_instance and ser_instance.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser_instance.write(cmd.encode())
            ser_instance.flush()
        except serial.SerialException as e:
            print(f"[SERIAL SEND ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL SEND ERROR] Error tidak dikenal saat mengirim: {e}")
    # else:
        # print("[SERIAL WARNING] Serial port tidak terinisialisasi atau tertutup.") # Uncomment for more verbose output

# --- Fungsi Utama Program ---
def main():
    fuzzy_controller = setup_fuzzy_logic()
    camera = setup_camera()
    serial_port = setup_serial()
    error_filter = ErrorFilter()
    line_recovery_handler = LineRecovery()

    prev_filtered_error = 0
    frame_counter = 0
    
    print("[INFO] Memulai robot Line Follower...")
    print("[INFO] Menunggu kamera stabil...")
    time.sleep(2) # Beri waktu kamera untuk auto-exposure

    # --- START-UP KICK-START ---
    if serial_port and serial_port.is_open:
        print(f"[STARTUP] Memberikan dorongan awal ({INITIAL_PUSH_PWM} PWM selama {INITIAL_PUSH_DURATION} detik)...")
        start_time = time.time()
        while time.time() - start_time < INITIAL_PUSH_DURATION:
            send_motor_commands(serial_port, INITIAL_PUSH_PWM, INITIAL_PUSH_PWM)
            time.sleep(0.05) # Jeda singkat agar perintah terkirim
        print("[STARTUP] Dorongan awal selesai.")
    else:
        print("[STARTUP WARNING] Serial port tidak aktif, tidak dapat memberikan dorongan awal.")

    try:
        while True:
            frame = camera.capture_array()
            height, width = frame.shape[:2]
            center_x = width // 2

            gray_frame, binary_frame, roi_frame, roi_y_start, roi_x_start, roi_y_end, roi_x_end = process_image(frame)
            line_found, line_cx, line_cy, line_contour = calculate_line_position(roi_frame, roi_y_start, roi_x_start, width)

            current_pwm_kiri = 0
            current_pwm_kanan = 0

            if line_found:
                line_recovery_handler.line_found(line_cx - center_x)
                
                current_error = line_cx - center_x
                filtered_error = error_filter.filter_error(current_error)
                
                delta_error = filtered_error - prev_filtered_error
                prev_filtered_error = filtered_error

                control_output = compute_fuzzy_control(fuzzy_controller, filtered_error, delta_error)
                current_pwm_kiri, current_pwm_kanan = calculate_motor_pwm(control_output, BASE_PWM, SCALING_FACTOR)
                
                send_motor_commands(serial_port, current_pwm_kiri, current_pwm_kanan)

                if frame_counter % 10 == 0:
                    status_text = "STRAIGHT" if abs(filtered_error) < FLC_DEAD_ZONE_ERROR else "TURNING"
                    print(f"[{status_text}] Err:{filtered_error:3d}, ΔErr:{delta_error:3d}, FLC:{control_output:5.1f}, PWM: L{current_pwm_kiri} R{current_pwm_kanan}")
                    
            else:
                recovery_action = line_recovery_handler.handle_line_lost(serial_port)
                prev_filtered_error = 0 
                if frame_counter % 15 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, lost_count: {line_recovery_handler.lost_count}")
            
            # --- Visualisasi (untuk debugging dan monitoring) ---
            frame_display = frame.copy()
            cv2.line(frame_display, (center_x, 0), (center_x, height), (0, 255, 0), 2)
            roi_color = (255, 255, 0)
            cv2.rectangle(frame_display, 
                          (roi_x_start, roi_y_start), 
                          (roi_x_end, roi_y_end), 
                          roi_color, 2)
            
            if line_found:
                cv2.circle(frame_display, (line_cx, line_cy), 8, (0, 0, 255), -1)
                if line_contour is not None:
                    adjusted_contour = line_contour.copy()
                    adjusted_contour[:, :, 0] += roi_x_start
                    adjusted_contour[:, :, 1] += roi_y_start
                    cv2.drawContours(frame_display, [adjusted_contour], -1, (255, 0, 255), 2)
                
                current_error_display = line_cx - center_x # Raw error for display
                cv2.putText(frame_display, f"Error: {current_error_display}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Filtered Err: {filtered_error}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame_display, f"Status: TRACKING", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_display, f"Status: LINE LOST ({line_recovery_handler.lost_count})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Line Follower Live", cv2.resize(frame_display, (640, 480)))
            cv2.imshow("Binary Processed ROI", cv2.resize(binary_frame, (320, 240)))
            cv2.imshow("Cleaned ROI for Contour", cv2.resize(roi_frame, (320, 100)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1
            time.sleep(0.02) # Control framerate
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna (KeyboardInterrupt).")
    except Exception as e:
        print(f"[CRITICAL ERROR] Terjadi error tak terduga: {e}")
    finally:
        print("[INFO] Menghentikan motor dan membersihkan sumber daya.")
        send_motor_commands(serial_port, 0, 0) # Pastikan motor berhenti total
        if serial_port and serial_port.is_open:
            serial_port.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
