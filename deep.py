from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# import matplotlib.pyplot as plt # Bisa di-uncomment jika ingin visualisasi FLC membership functions

class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.7 # Alpha untuk Exponential smoothing
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        # Median filter untuk mengurangi noise, lalu Exponential smoothing
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors) // 2]
        
        smoothed_error = median_error # Default jika hanya ada 1 data
        if len(self.error_history) > 1:
            # Gunakan nilai terakhir yang di-smooth dari history untuk EMA
            # Ini lebih akurat sebagai EMA dibanding mengambil median dari history sebelumnya
            prev_smoothed = self.error_history[-2] # Ambil nilai sebelum yang terakhir di history (yang belum di-smooth)
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_smoothed
        
        return int(smoothed_error)

class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.recovery_speed = 45 # Sedikit lebih cepat dari 40

    def handle_line_lost(self, ser):
        self.lost_count += 1
        
        action_status = "stop"
        pwm_left, pwm_right = 0, 0
        
        if self.lost_count < 15: # Stop lebih lama sedikit
            pwm_left, pwm_right = 0, 0
            action_status = "stop"
        elif self.lost_count < 40: # Mundur lebih lama sedikit
            pwm_left, pwm_right = -35, -35 # Mundur dengan PWM -35
            action_status = "reverse"
        else: # Cari garis dengan berputar
            if self.last_valid_error > 0: # Garis terakhir di kanan (putar kanan)
                pwm_left, pwm_right = self.recovery_speed, -self.recovery_speed
                action_status = "search_right"
            else: # Garis terakhir di kiri (putar kiri)
                pwm_left, pwm_right = -self.recovery_speed, self.recovery_speed
                action_status = "search_left"
        
        # Panggil fungsi global send_motor_commands
        send_motor_commands(ser, pwm_left, pwm_right)
        return action_status
        
    def line_found(self, error):
        self.lost_count = 0
        self.last_valid_error = error

def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-320, 321, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions untuk ERROR
    error['NL'] = fuzz.trimf(error.universe, [-320, -200, -60])
    error['NS'] = fuzz.trimf(error.universe, [-70, -25, -5])
    error['Z']  = fuzz.trimf(error.universe, [-10, 0, 10])
    error['PS'] = fuzz.trimf(error.universe, [5, 25, 70])
    error['PL'] = fuzz.trimf(error.universe, [60, 200, 320])

    # Membership functions untuk DELTA ERROR
    delta['NL'] = fuzz.trimf(delta.universe, [-100, -60, -20])
    delta['NS'] = fuzz.trimf(delta.universe, [-25, -10, -1])
    delta['Z']  = fuzz.trimf(delta.universe, [-3, 0, 3])
    delta['PS'] = fuzz.trimf(delta.universe, [1, 10, 25])
    delta['PL'] = fuzz.trimf(delta.universe, [20, 60, 100])

    # Membership functions untuk OUTPUT (Kontrol PWM)
    output['L']  = fuzz.trimf(output.universe, [-100, -75, -30])
    output['LS'] = fuzz.trimf(output.universe, [-40, -15, -5])
    output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])
    output['RS'] = fuzz.trimf(output.universe, [5, 15, 40])
    output['R']  = fuzz.trimf(output.universe, [30, 75, 100])

    # Aturan Fuzzy
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
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
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    
    # if you want to visualize the membership functions:
    # error.view()
    # delta.view()
    # output.view()
    # plt.show()
    
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    try:
        # Gunakan ttyS0 untuk RPi 4, atau ttyAMA0 jika diaktifkan dan terhubung ke GPIO
        ser = serial.Serial('/dev/ttyS0', 115200, timeout=1) 
        print("[UART] Port serial berhasil dibuka")
        return ser
    except serial.SerialException as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}. Pastikan kabel terhubung dan port benar.")
        print("Coba cek: 'ls /dev/tty*' dan 'sudo raspi-config' -> Interface Options -> Serial Port.")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] Gagal membuka serial port: {e}")
        return None

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
    
    roi = binary[roi_start:roi_end, roi_left:roi_right]
    
    return binary, roi, roi_start, roi_left, height, width

def calculate_line_position(roi, roi_start, roi_left):
    kernel = np.ones((5,5), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 300: # Eksperimen dengan nilai ini!
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + roi_left
                cy = int(M['m01'] / M['m00']) + roi_start
                return True, cx, cy, largest_contour
    
    return False, 0, 0, None

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -320, 320) # Menyesuaikan dengan universe error
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -100, 100) # Menyesuaikan dengan universe delta
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}. Error_val: {error_val}, Delta_error: {delta_error}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.1): # base_pwm ditingkatkan, scaling_factor ditingkatkan
    if abs(kontrol) < 10: # Dead zone sedikit dipersempit dari 15 ke 10
        kontrol_scaled = 0 
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    pwm_kiri = max(25, min(70, pwm_kiri)) # Batasan PWM sedikit diperlebar (max 70)
    pwm_kanan = max(25, min(70, pwm_kanan))

    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL SEND ERROR] {e}")

def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter()
    line_recovery = LineRecovery()

    prev_error = 0
    frame_count = 0
    
    print("[INFO] Warming up camera...")
    time.sleep(2)

    try:
        while True:
            # Mengambil frame dan memproses gambar
            frame = picam2.capture_array()
            binary, roi, roi_start, roi_left, height, width = process_image(frame)
            center_x = width // 2
            
            line_detected, cx, cy, contour = calculate_line_position(roi, roi_start, roi_left)

            current_pwm_left, current_pwm_right = 0, 0 # Inisialisasi PWM untuk log

            if line_detected:
                line_recovery.line_found(cx - center_x)
                
                error = cx - center_x
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                current_pwm_left = pwm_kiri
                current_pwm_right = pwm_kanan

                if frame_count % 10 == 0:
                    status = "STRAIGHT" if abs(error) < 10 else "TURNING" # Menggunakan dead zone baru
                    print(f"[{status}] Err:{error:3d}, ΔErr:{delta_error:3d}, FLC:{kontrol:5.1f}, PWM L:{pwm_kiri} R:{pwm_kanan}")
                    
            else:
                recovery_action = line_recovery.handle_line_lost(ser)
                # PWM akan diset di dalam handle_line_lost, jadi ambil dari sana jika perlu
                # Untuk log, kita bisa tambahkan logika untuk mendapatkan PWM dari recovery action
                # Namun untuk menyederhanakan, kita asumsikan recovery_action akan mengirimkan sendiri
                # dan log utama hanya menampilkan status.
                
                if frame_count % 15 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, count: {line_recovery.lost_count}")
                
                prev_error = 0 # Reset error saat garis hilang

            # Visualisasi
            frame_with_line = frame.copy()
            cv2.line(frame_with_line, (center_x, 0), (center_x, height), (0, 255, 0), 2)
            
            # Gambar ROI
            roi_color = (255, 255, 0)
            cv2.rectangle(frame_with_line, 
                          (roi_left, roi_start), 
                          (roi_left + roi.shape[1], roi_start + roi.shape[0]), # Gunakan shape roi untuk end point
                          roi_color, 2)
            
            if line_detected:
                cv2.circle(frame_with_line, (cx, cy), 8, (0, 0, 255), -1)
                if contour is not None:
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += roi_left
                    adjusted_contour[:, :, 1] += roi_start
                    cv2.drawContours(frame_with_line, [adjusted_contour], -1, (255, 0, 255), 2)
                
                error_display = cx - center_x
                cv2.putText(frame_with_line, f"Error: {error_display}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_line, f"Status: TRACKING", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Tampilkan PWM di overlay (opsional, bisa bikin rame)
                # cv2.putText(frame_with_line, f"PWM L:{current_pwm_left} R:{current_pwm_right}",
                #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame_with_line, f"Status: LINE LOST ({line_recovery.lost_count})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Line Follower", cv2.resize(frame_with_line, (640, 480)))
            cv2.imshow("Binary", cv2.resize(binary, (320, 240)))
            cv2.imshow("ROI", cv2.resize(roi, (320, int(roi.shape[0]*320/roi.shape[1]) if roi.shape[1] > 0 else 100))) # Sesuaikan ukuran display ROI

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.02) # 50 FPS
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    finally:
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
