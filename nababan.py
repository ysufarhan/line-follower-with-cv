from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Konfigurasi Global ---
# Sesuaikan port serial jika diperlukan. Umumnya '/dev/ttyS0' untuk RPi 4, atau '/dev/ttyAMA0'
# Pastikan sudah diaktifkan di raspi-config (Interface Options -> P6 Serial Port -> No for login shell, Yes for hardware serial)
SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 115200

# Kecepatan dasar robot saat lurus dan scaling factor untuk belokan
BASE_PWM = 40 # MODIFIKASI: Turunkan BASE_PWM sedikit untuk kontrol belok lebih baik. (Original: 45)
SCALING_FACTOR = 0.12 # MODIFIKASI: Tingkatkan scaling factor untuk belokan lebih agresif. (Original: 0.08)

# Batasan PWM untuk motor agar aman dan efektif (0-100)
MIN_PWM_OUTPUT = 10 # MODIFIKASI: Turunkan MIN_PWM_OUTPUT agar motor bisa lebih pelan/hampir berhenti. (Original: 25)
MAX_PWM_OUTPUT = 70 # MODIFIKASI: Tingkatkan MAX_PWM_OUTPUT untuk kekuatan belok lebih besar. (Original: 65)
MAX_REVERSE_PWM = 40 # MODIFIKASI: PWM maksimum saat mundur (untuk pivot turn). Sesuaikan dengan kekuatan motor Anda.

# Ukuran Dead Zone FLC: dalam piksel. Error di bawah nilai ini dianggap nol.
# Membantu stabilitas di garis lurus.
FLC_DEAD_ZONE_ERROR = 10 # MODIFIKASI: Kurangi dead zone agar robot lebih cepat merespons di awal tikungan. (Original: 15)

# Batas ambang output FLC untuk memicu pivot turn
PIVOT_THRESHOLD = 70 # Jika abs(control_output) > 70, coba pivot. Sesuaikan nilai ini!

# --- Kelas untuk Filter Error ---
class ErrorFilter:
    def __init__(self, window_size=5, alpha=0.6): # MODIFIKASI: Tingkatkan window_size & sesuaikan alpha
        self.window_size = window_size
        self.error_history = []
        self.alpha = alpha
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        median_error = error
        if len(self.error_history) > 0:
            sorted_errors = sorted(self.error_history)
            median_error = sorted_errors[len(sorted_errors)//2]
        
        if len(self.error_history) > 1:
            prev_smoothed_error = self.error_history[-2] if len(self.error_history) >= 2 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_smoothed_error
        else:
            smoothed_error = median_error
            
        return int(smoothed_error)

# --- Kelas untuk Pemulihan Garis ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.recovery_speed = 30 # MODIFIKASI: Kurangi kecepatan recovery untuk lebih stabil saat mencari. (Original: 40)

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # MODIFIKASI: Sesuaikan fase recovery agar lebih efektif
        # Fase 1: Diam sebentar
        if self.lost_count < 10:
            send_motor_commands(ser_instance, 0, 0)
            return "STOP"
        # Fase 2: Mundur sedikit
        elif self.lost_count < 30:
            send_motor_commands(ser_instance, -25, -25)
            return "REVERSE"
        # Fase 3: Berputar mencari garis
        else:
            turn_pwm = min(self.recovery_speed, int(abs(self.last_valid_error) * 0.1))
            if turn_pwm < 10: turn_pwm = 10
            
            if self.last_valid_error > 0: # Garis terakhir di kanan, putar ke kanan (kiri maju, kanan mundur)
                send_motor_commands(ser_instance, turn_pwm, -turn_pwm)
                return "SEARCH_RIGHT"
            else: # Garis terakhir di kiri, putar ke kiri (kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -turn_pwm, turn_pwm)
                return "SEARCH_LEFT"
        
    def line_found(self, current_error):
        self.lost_count = 0
        self.last_valid_error = current_error

# --- Setup Fuzzy Logic Control (FLC) ---
def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership Functions (Tuning untuk "mulus saat berbelok dan tetap mempertahankan titik tengah")
    # MODIFIKASI: Agresifkan respons di tepi, perlebar jangkauan NL/PL, persempit Z.
    # Error:
    error['NL'] = fuzz.trimf(error.universe, [-350, -180, -50])
    error['NS'] = fuzz.trimf(error.universe, [-70, -25, -5])
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])
    error['PS'] = fuzz.trimf(error.universe, [5, 25, 70])
    error['PL'] = fuzz.trimf(error.universe, [50, 180, 350])

    # Delta Error:
    delta['NL'] = fuzz.trimf(delta.universe, [-150, -80, -25])
    delta['NS'] = fuzz.trimf(delta.universe, [-35, -12, -2])
    delta['Z']  = fuzz.trimf(delta.universe, [-4, 0, 4])
    delta['PS'] = fuzz.trimf(delta.universe, [2, 12, 35])
    delta['PL'] = fuzz.trimf(delta.universe, [25, 80, 150])

    # Output:
    output['L']  = fuzz.trimf(output.universe, [-100, -75, -45])
    output['LS'] = fuzz.trimf(output.universe, [-55, -30, -10])
    output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])
    output['RS'] = fuzz.trimf(output.universe, [10, 30, 55])
    output['R']  = fuzz.trimf(output.universe, [45, 75, 100])

    # Rule Base (Disempurnakan untuk belokan mulus dan stabilitas)
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']), # MODIFIKASI: Jauh kiri & mendekat -> Tetap belok kiri sedang
        ctrl.Rule(error['NL'] & delta['PL'], output['LS']), # MODIFIKASI: Jauh kiri & cepat mendekat -> Tetap belok kiri sedang

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

        ctrl.Rule(error['PL'] & delta['NL'], output['RS']), # MODIFIKASI: Jauh kanan & cepat mendekat -> Tetap belok kanan sedang
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']), # MODIFIKASI: Jauh kanan & mendekat -> Tetap belok kanan sedang
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
    
    try:
        picam2.set_controls({"AeEnable": True}) # Aktifkan auto-exposure
        picam2.set_controls({"AwbEnable": True}) # Aktifkan auto white balance
    except Exception as e:
        print(f"[CAMERA CONTROL WARNING] Could not set camera controls: {e}")

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
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # MODIFIKASI: Kurangi kernel size Gaussian. (Original: (7,7))
    blurred = cv2.medianBlur(blurred, 3) # MODIFIKASI: Kurangi kernel size Median. (Original: 5)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    height, width = binary.shape
    
    # Region of Interest (ROI)
    # MODIFIKASI: Perlebar ROI ke atas (melihat lebih jauh ke depan).
    roi_start = int(height * 0.05)
    roi_end = int(height * 0.98)
    roi_left = int(width * 0.02)
    roi_right = int(width * 0.98)

    roi_start = max(0, min(roi_start, height - 1))
    roi_end = max(0, min(roi_end, height))
    roi_left = max(0, min(roi_left, width - 1))
    roi_right = max(0, min(roi_right, width))

    roi = binary[roi_start:roi_end, roi_left:roi_right]
    
    return gray, binary, roi, roi_start, roi_left, roi_end, roi_right

# --- Menghitung Posisi Garis ---
def calculate_line_position(roi_image, roi_start_y, roi_start_x, frame_width):
    kernel = np.ones((3,3), np.uint8) # MODIFIKASI: Kurangi kernel size morfologi. (Original: (5,5))
    roi_clean = cv2.morphologyEx(roi_image, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # MODIFIKASI: Turunkan threshold area minimum untuk deteksi garis. (Original: 300)
        if cv2.contourArea(largest_contour) > 150: # Coba 150; bisa 100-250
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
    # Menerapkan dead zone untuk stabilitas di garis lurus
    if abs(control_output) < FLC_DEAD_ZONE_ERROR: 
        control_scaled = 0 
    else:
        control_scaled = control_output * scaling_factor

    pwm_kiri = base_pwm - control_scaled
    pwm_kanan = base_pwm + control_scaled
    
    # MODIFIKASI: Logika untuk memungkinkan pivot turn (satu motor mundur)
    if control_output < -PIVOT_THRESHOLD: # Belok keras kiri (control_output sangat negatif)
        pwm_kiri = -min(MAX_REVERSE_PWM, int(abs(control_output) * (MAX_REVERSE_PWM / 100.0)))
        pwm_kanan = MAX_PWM_OUTPUT
    elif control_output > PIVOT_THRESHOLD: # Belok keras kanan (control_output sangat positif)
        pwm_kiri = MAX_PWM_OUTPUT
        pwm_kanan = -min(MAX_REVERSE_PWM, int(abs(control_output) * (MAX_REVERSE_PWM / 100.0)))
    else: # Normal tracking (maju kedua motor, variasi kecepatan)
        pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, int(pwm_kiri)))
        pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, int(pwm_kanan)))
        
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
    elif not ser_instance:
        pass

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
    time.sleep(2) # Beri waktu kamera untuk melakukan auto-exposure

    try:
        while True:
            frame = camera.capture_array()
            height, width = frame.shape[:2]
            center_x = width // 2

            gray_frame, binary_frame, roi_frame, roi_y_start, roi_x_start, roi_y_end, roi_x_end = process_image(frame)
            
            line_found, line_cx, line_cy, line_contour = calculate_line_position(roi_frame, roi_y_start, roi_x_start, width)

            if line_found:
                line_recovery_handler.line_found(line_cx - center_x)
                
                current_error = line_cx - center_x
                
                filtered_error = error_filter.filter_error(current_error)
                
                delta_error = filtered_error - prev_filtered_error
                prev_filtered_error = filtered_error

                control_output = compute_fuzzy_control(fuzzy_controller, filtered_error, delta_error)
                
                current_pwm_kiri, current_pwm_kanan = calculate_motor_pwm(control_output, BASE_PWM, SCALING_FACTOR)
                
                send_motor_commands(serial_port, current_pwm_kiri, current_pwm_kanan)

                if frame_counter % 5 == 0:
                    status_text = "STRAIGHT" if abs(filtered_error) < FLC_DEAD_ZONE_ERROR else "TURNING"
                    print(f"[{status_text}] Err:{filtered_error:3d}, ΔErr:{delta_error:3d}, FLC:{control_output:6.2f}, PWM: L{current_pwm_kiri:3d} R{current_pwm_kanan:3d}")
                    
            else:
                recovery_action = line_recovery_handler.handle_line_lost(serial_port)
                prev_filtered_error = 0
                if frame_counter % 10 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, lost_count: {line_recovery_handler.lost_count}")
                    
            # --- Visualisasi (untuk debugging dan monitoring jurnal) ---
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
                
                cv2.putText(frame_display, f"Err: {current_error}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Filtered Err: {filtered_error}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame_display, f"Status: TRACKING", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_display, f"Status: LINE LOST ({line_recovery_handler.lost_count})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # MODIFIKASI: Sesuaikan ukuran tampilan jendela untuk efisiensi rendering
            cv2.imshow("Line Follower Live", cv2.resize(frame_display, (320, 240)))
            cv2.imshow("Binary Processed ROI", cv2.resize(binary_frame, (320, 240)))
            cv2.imshow("Cleaned ROI for Contour", cv2.resize(roi_frame, (320, 100)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1
            # time.sleep(0.01) # Sesuaikan jika robot terlalu cepat/lambat atau CPU tinggi.
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna (KeyboardInterrupt).")
    except Exception as e:
        print(f"[CRITICAL ERROR] Terjadi error tak terduga: {e}")
    finally:
        print("[INFO] Menghentikan motor dan membersihkan sumber daya.")
        send_motor_commands(serial_port, 0, 0)
        if serial_port and serial_port.is_open:
            serial_port.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
