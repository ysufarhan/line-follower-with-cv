from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.7  # Parameter untuk exponential smoothing

    def filter_error(self, error):
        # Kombinasi moving average dan exponential smoothing
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Moving average
        avg_error = sum(self.error_history) / len(self.error_history)
        
        # Exponential smoothing untuk mengurangi noise
        if len(self.error_history) > 1:
            prev_avg = sum(self.error_history[:-1]) / len(self.error_history[:-1])
            smoothed_error = self.alpha * avg_error + (1 - self.alpha) * prev_avg
        else:
            smoothed_error = avg_error
            
        return int(smoothed_error)

def setup_fuzzy_logic():
    # Universe disesuaikan dengan kebutuhan obstacle avoidance
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # Membership functions yang lebih smooth untuk obstacle avoidance
    error['NL'] = fuzz.trimf(error.universe, [-200, -150, -60])
    error['NS'] = fuzz.trimf(error.universe, [-90, -35, -5])
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])      # Zona netral diperlebar
    error['PS'] = fuzz.trimf(error.universe, [5, 35, 90])
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 200])

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -80, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -20, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])      # Zona netral diperlebar
    delta['PS'] = fuzz.trimf(delta.universe, [3, 20, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 80, 150])

    output['L']  = fuzz.trimf(output.universe, [-150, -100, -60])
    output['LS'] = fuzz.trimf(output.universe, [-75, -35, -8])
    output['Z']  = fuzz.trimf(output.universe, [-12, 0, 12])    
    output['RS'] = fuzz.trimf(output.universe, [8, 35, 75])
    output['R']  = fuzz.trimf(output.universe, [60, 100, 150])

    # Rules yang sama
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
        # Sesuaikan dengan baudrate ESP32 (115200)
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka - ESP32 Ready")
        time.sleep(2)  # Beri waktu ESP32 untuk siap
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = binary[160:240, :]
    return gray, binary, roi

def calculate_line_position(roi):
    kernel = np.ones((3,3), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    M = cv2.moments(roi_clean)
    if M['m00'] > 100:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + 160
        return True, cx, cy
    return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.3):
    # Sesuaikan dengan ESP32 PWM range (0-100)
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # PWM range untuk ESP32 (0-100), dengan minimum untuk torsi
    pwm_kiri = max(30, min(80, pwm_kiri))  
    pwm_kanan = max(30, min(80, pwm_kanan))
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            # Format sesuai dengan ESP32: "pwm_kiri,pwm_kanan\n"
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode('utf-8'))
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def handle_emergency_stop(ser):
    """Fungsi untuk menghentikan motor dalam keadaan darurat"""
    if ser:
        try:
            ser.write("0,0\n".encode('utf-8'))
            ser.flush()
            print("[EMERGENCY] Motor dihentikan")
        except:
            pass

def main():
    print("[INFO] Inisialisasi sistem Line Following dengan Obstacle Avoidance...")
    
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)

    if not ser:
        print("[ERROR] Tidak dapat melanjutkan tanpa koneksi serial")
        return

    prev_error = 0
    frame_count = 0
    lost_line_count = 0
    max_lost_frames = 10  # Berapa frame boleh kehilangan garis

    try:
        print("[INFO] Sistem siap - Tekan 'q' untuk keluar")
        while True:
            frame = picam2.capture_array()
            gray, binary, roi = process_image(frame)

            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                lost_line_count = 0  # Reset counter
                error = cx - 160
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                # Debug output yang lebih informatif
                if frame_count % 15 == 0:
                    status = "STRAIGHT" if abs(error) < 15 else ("LEFT" if error < 0 else "RIGHT")
                    print(f"[{status}] E:{error:4d} | D:{delta_error:4d} | FLC:{kontrol:6.2f} | PWM: L{pwm_kiri} R{pwm_kanan}")
                    
            else:
                lost_line_count += 1
                if lost_line_count >= max_lost_frames:
                    # Berhenti jika terlalu lama kehilangan garis
                    send_motor_commands(ser, 0, 0)
                    if frame_count % 20 == 0:
                        print("[WARNING] Garis hilang - Motor dihentikan")
                else:
                    # Tetap gunakan kontrol terakhir untuk beberapa frame
                    if frame_count % 30 == 0:
                        print(f"[SEARCHING] Mencari garis... ({lost_line_count}/{max_lost_frames})")

            # Display untuk monitoring
            frame_with_overlay = frame.copy()
            cv2.line(frame_with_overlay, (160, 160), (160, 240), (0, 255, 0), 2)
            
            if line_detected:
                cv2.circle(frame_with_overlay, (cx, cy), 5, (0, 0, 255), -1)
                # Status indicator
                color = (0, 255, 0) if abs(error) < 15 else (0, 255, 255)
                cv2.putText(frame_with_overlay, f"Error: {error}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame_with_overlay, "LINE LOST!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Obstacle status (informasi dari ESP32 akan tampil di serial monitor ESP32)
            cv2.putText(frame_with_overlay, "Obstacle Check: ESP32", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imshow("Line Following + Obstacle Avoidance", frame_with_overlay)
            cv2.imshow("Binary ROI", roi)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.05)  # 20 FPS - sesuai dengan ESP32 sensor timing
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Membersihkan sistem...")
        handle_emergency_stop(ser)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai dengan aman")

if __name__ == "__main__":
    main()
