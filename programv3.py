from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ErrorFilter:
    def __init__(self, window_size=5):  # Diperbesar untuk stabilitas
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.6  # Dikurangi untuk respon lebih halus

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Median filter untuk mengurangi noise
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors)//2]
        
        # Exponential smoothing
        if len(self.error_history) > 1:
            prev_filtered = sorted_errors[-2] if len(sorted_errors) > 1 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_filtered
        else:
            smoothed_error = median_error
        return int(smoothed_error)

class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.search_direction = 1  # 1 untuk kanan, -1 untuk kiri
        self.recovery_speed = 40
        
    def handle_line_lost(self, ser):
        self.lost_count += 1
        
        if self.lost_count < 10:  # Coba diam dulu
            self.send_motor_commands(ser, 0, 0)
            return "stop"
        elif self.lost_count < 30:  # Mundur sedikit
            self.send_motor_commands(ser, -30, -30)
            return "reverse"
        else:  # Cari garis dengan berputar
            if self.last_valid_error > 0:  # Garis terakhir di kanan
                self.send_motor_commands(ser, self.recovery_speed, -self.recovery_speed)
                return "search_right"
            else:  # Garis terakhir di kiri
                self.send_motor_commands(ser, -self.recovery_speed, self.recovery_speed)
                return "search_left"
    
    def line_found(self, error):
        self.lost_count = 0
        self.last_valid_error = error
        
    def send_motor_commands(self, ser, pwm_kiri, pwm_kanan):
        if ser:
            try:
                cmd = f"{pwm_kiri},{pwm_kanan}\n"
                ser.write(cmd.encode())
                ser.flush()
            except Exception as e:
                print(f"[SERIAL ERROR] {e}")

def setup_fuzzy_logic():
    # Range error yang disesuaikan
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions dengan zona zero yang lebih lebar
    error['NL'] = fuzz.trimf(error.universe, [-200, -150, -50])
    error['NS'] = fuzz.trimf(error.universe, [-80, -35, -10])
    error['Z']  = fuzz.trimf(error.universe, [-25, 0, 25])  # Zona zero diperlebar
    error['PS'] = fuzz.trimf(error.universe, [10, 35, 80])
    error['PL'] = fuzz.trimf(error.universe, [50, 150, 200])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -60, -20])
    delta['NS'] = fuzz.trimf(delta.universe, [-35, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])  # Zona zero diperlebar
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 35])
    delta['PL'] = fuzz.trimf(delta.universe, [20, 60, 100])

    # Output dengan zona zero yang dominan
    output['L']  = fuzz.trimf(output.universe, [-100, -70, -40])
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -8])
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])  # Zona zero lebar
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 50])
    output['R']  = fuzz.trimf(output.universe, [40, 70, 100])

    # Rules yang lebih konservatif - prioritas zona zero
    rules = [
        # Error Negative Large
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),   # Lebih ke zero
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),   # Lebih ke zero

        # Error Negative Small
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']),   # Ke zero
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),    # Ke zero
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   # Ke zero
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),   # Ke zero

        # Error Zero - SEMUA KE ZERO untuk jalan lurus
        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

        # Error Positive Small
        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),   # Ke zero
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   # Ke zero
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),    # Ke zero
        ctrl.Rule(error['PS'] & delta['PS'], output['Z']),   # Ke zero
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        # Error Positive Large
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),   # Lebih ke zero
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),   # Lebih ke zero
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    # Resolusi diperbesar untuk detail lebih baik dari ketinggian
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adaptive histogram equalization untuk lighting yang tidak merata
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Multiple blur untuk mengurangi noise dari ketinggian
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 5)
    
    # OTSU thresholding - otomatis menentukan threshold optimal
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ROI lebih lebar dan lebih jauh untuk kamera tinggi
    height, width = binary.shape
    roi_start = int(height * 0.6)  # Mulai dari 60% tinggi frame
    roi = binary[roi_start:height, int(width*0.1):int(width*0.9)]  # 10% margin kiri-kanan
    
    return gray, binary, roi, roi_start

def calculate_line_position(roi, roi_start, frame_width):
    # Morphological operations yang lebih agresif
    kernel = np.ones((5,5), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    # Cari kontur untuk deteksi garis yang lebih akurat
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Cari kontur terbesar (kemungkinan garis)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 200:  # Threshold area minimum
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + int(frame_width*0.1)  # Kompensasi margin
                cy = int(M['m01'] / M['m00']) + roi_start
                return True, cx, cy, largest_contour
    
    return False, 0, 0, None

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -100, 100)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=45, scaling_factor=0.08):
    # Dead zone diperlebar untuk stabilitas di garis lurus
    if abs(kontrol) < 15:  # Zona mati lebih besar
        kontrol_scaled = 0  # PWM kiri = kanan (jalan lurus)
    else:
        # Kontrol yang lebih halus
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Range PWM yang lebih kecil dan aman
    pwm_kiri = max(25, min(65, pwm_kiri))
    pwm_kanan = max(25, min(65, pwm_kanan))

    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=5)
    line_recovery = LineRecovery()

    prev_error = 0
    frame_count = 0
    
    # Stabilization period
    print("[INFO] Warming up camera...")
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x = width // 2
            
            gray, binary, roi, roi_start = process_image(frame)
            line_detected, cx, cy, contour = calculate_line_position(roi, roi_start, width)

            if line_detected:
                line_recovery.line_found(cx - center_x)
                
                error = cx - center_x
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    status = "STRAIGHT" if abs(error) < 15 else "TURNING"
                    print(f"[{status}] Err:{error:3d}, ÎErr:{delta_error:3d}, FLC:{kontrol:5.1f}, PWM: L{pwm_kiri} R{pwm_kanan}")
                    
            else:
                recovery_action = line_recovery.handle_line_lost(ser)
                if frame_count % 15 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, count: {line_recovery.lost_count}")

            # Visualisasi yang lebih informatif
            frame_with_line = frame.copy()
            
            # Center line
            cv2.line(frame_with_line, (center_x, 0), (center_x, height), (0, 255, 0), 2)
            
            # ROI area
            roi_color = (255, 255, 0)
            cv2.rectangle(frame_with_line, 
                         (int(width*0.1), roi_start), 
                         (int(width*0.9), height), 
                         roi_color, 2)
            
            if line_detected:
                # Line position
                cv2.circle(frame_with_line, (cx, cy), 8, (0, 0, 255), -1)
                
                # Draw contour
                if contour is not None:
                    # Adjust contour coordinates
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += int(width*0.1)
                    adjusted_contour[:, :, 1] += roi_start
                    cv2.drawContours(frame_with_line, [adjusted_contour], -1, (255, 0, 255), 2)
                
                # Error display
                error_display = cx - center_x
                cv2.putText(frame_with_line, f"Error: {error_display}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_line, f"Status: TRACKING", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_with_line, f"Status: LINE LOST ({line_recovery.lost_count})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display images
            cv2.imshow("Line Follower", cv2.resize(frame_with_line, (640, 480)))
            cv2.imshow("Binary", cv2.resize(binary, (320, 240)))
            cv2.imshow("ROI", cv2.resize(roi, (320, 100)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.03)  # Slightly faster loop
            
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
