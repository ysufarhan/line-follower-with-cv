from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

class AdaptiveController:
    """Adaptive controller untuk mengatasi osilasi dan responsivitas"""
    def __init__(self):
        self.error_history = []
        self.oscillation_count = 0
        self.last_direction_change = 0
        self.current_mode = "normal"  # normal, curve, recovery
        self.curve_detection_threshold = 80
        self.straight_line_count = 0
        
    def detect_oscillation(self, error):
        """Deteksi osilasi berdasarkan perubahan tanda error"""
        self.error_history.append(error)
        if len(self.error_history) > 5:
            self.error_history.pop(0)
            
        # Hitung perubahan arah
        direction_changes = 0
        for i in range(1, len(self.error_history)):
            if (self.error_history[i] * self.error_history[i-1]) < 0:
                direction_changes += 1
                
        # Update oscillation detection
        if direction_changes >= 3:
            self.oscillation_count = min(self.oscillation_count + 1, 10)
        else:
            self.oscillation_count = max(self.oscillation_count - 1, 0)
            
        return self.oscillation_count > 3
    
    def detect_curve(self, error, line_area):
        """Deteksi belokan berdasarkan error dan area garis"""
        is_curve = False
        
        # Kriteria belokan: error besar + area garis memadai
        if abs(error) > self.curve_detection_threshold and line_area > 200:
            is_curve = True
            self.straight_line_count = 0
        elif abs(error) < 30:
            self.straight_line_count += 1
            
        # Update mode
        if is_curve:
            self.current_mode = "curve"
        elif self.oscillation_count > 3:
            self.current_mode = "recovery"
        elif self.straight_line_count > 10:
            self.current_mode = "normal"
            
        return is_curve
    
    def get_adaptive_params(self, error):
        """Dapatkan parameter adaptif berdasarkan kondisi"""
        if self.current_mode == "curve":
            # Mode belokan: lebih responsif
            return {
                'base_pwm': 45,  # Lebih lambat untuk kontrol lebih baik
                'scaling_factor': 0.35,  # Lebih responsif
                'max_pwm_diff': 35
            }
        elif self.current_mode == "recovery":
            # Mode recovery dari osilasi: lebih halus
            return {
                'base_pwm': 50,
                'scaling_factor': 0.15,  # Lebih halus
                'max_pwm_diff': 20
            }
        else:
            # Mode normal
            return {
                'base_pwm': 55,
                'scaling_factor': 0.22,
                'max_pwm_diff': 25
            }

def setup_fuzzy_logic_adaptive():
    """Setup FLC dengan rules yang lebih responsif untuk belokan"""
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions dengan zona dead yang lebih kecil
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -50])
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -8])
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])
    error['PS'] = fuzz.trimf(error.universe, [8, 30, 80])
    error['PL'] = fuzz.trimf(error.universe, [50, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 100, 100])

    # Output dengan respons yang lebih progresif
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -35])
    output['LS'] = fuzz.trimf(output.universe, [-50, -20, -5])
    output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 50])
    output['R']  = fuzz.trimf(output.universe, [35, 100, 100])

    # Rules yang lebih agresif untuk belokan
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['L']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PL'], output['LS']),

        ctrl.Rule(error['NS'] & delta['NL'], output['L']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['R']),

        ctrl.Rule(error['PL'] & delta['NL'], output['RS']),
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['R']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
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
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame, use_otsu=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if use_otsu:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    return gray, binary, binary[160:240, :]

def calculate_line_position_enhanced(roi):
    """Enhanced line detection dengan informasi area"""
    kernel = np.ones((3,3), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    M = cv2.moments(roi_clean)
    area = M['m00']
    
    if area > 100:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + 160
        return True, cx, cy, area
    return False, 0, 0, 0

def compute_fuzzy_control_adaptive(fuzzy_ctrl, error_val, delta_error):
    """Compute fuzzy control dengan bounds checking"""
    try:
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        
        kontrol = fuzzy_ctrl.output['output']
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm_adaptive(kontrol, params):
    """
    Adaptive PWM calculation dengan pembatasan maksimum
    """
    base_pwm = params['base_pwm']
    scaling_factor = params['scaling_factor']
    max_pwm_diff = params['max_pwm_diff']
    
    # Scale control signal
    kontrol_scaled = kontrol * scaling_factor
    
    # Limit maximum PWM difference untuk mencegah osilasi
    kontrol_scaled = max(-max_pwm_diff, min(max_pwm_diff, kontrol_scaled))
    
    # Calculate motor PWMs
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Ensure minimum speed untuk menghindari motor mati
    pwm_kiri = max(20, min(80, pwm_kiri))
    pwm_kanan = max(20, min(80, pwm_kanan))
    
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
    print("[SYSTEM] Starting Adaptive Line Following Robot")
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_adaptive()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)
    adaptive_ctrl = AdaptiveController()
    
    # Variabel kontrol
    prev_error = 0
    frame_count = 0
    no_line_count = 0
    
    print("[CONFIG] Adaptive control enabled")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            _, _, roi = process_image(frame)
            
            # Enhanced line detection
            line_detected, cx, cy, line_area = calculate_line_position_enhanced(roi)
            
            if line_detected:
                no_line_count = 0
                
                # Hitung error dan delta error
                error = cx - 160  # Setpoint di tengah (160)
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error
                
                # Adaptive control analysis
                is_oscillating = adaptive_ctrl.detect_oscillation(error)
                is_curve = adaptive_ctrl.detect_curve(error, line_area)
                
                # Get adaptive parameters
                adaptive_params = adaptive_ctrl.get_adaptive_params(error)
                
                # Compute FLC output
                kontrol = compute_fuzzy_control_adaptive(fuzzy_ctrl, error, delta_error)
                
                # Calculate adaptive PWM
                pwm_kiri, pwm_kanan = calculate_motor_pwm_adaptive(kontrol, adaptive_params)
                
                # Send commands
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Debug info
                if frame_count % 15 == 0:
                    mode_info = f"Mode: {adaptive_ctrl.current_mode}"
                    if is_oscillating:
                        mode_info += " [OSC]"
                    if is_curve:
                        mode_info += " [CURVE]"
                    
                    print(f"[DEBUG] {mode_info}")
                    print(f"        Error: {error:3d}, Delta: {delta_error:3d}, "
                          f"FLC: {kontrol:5.1f}, PWM: L={pwm_kiri}, R={pwm_kanan}")
                    print(f"        Base: {adaptive_params['base_pwm']}, "
                          f"Scale: {adaptive_params['scaling_factor']:.2f}")
                
            else:
                no_line_count += 1
                
                if no_line_count < 10:
                    # Garis hilang sebentar - keep moving dengan PWM rendah
                    send_motor_commands(ser, 35, 35)
                elif no_line_count < 30:
                    # Cari garis dengan berputar pelan
                    if prev_error > 0:
                        send_motor_commands(ser, 30, 45)  # Turn right
                    else:
                        send_motor_commands(ser, 45, 30)  # Turn left
                else:
                    # Stop jika garis tidak ditemukan terlalu lama
                    send_motor_commands(ser, 0, 0)
                
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Line not detected (count: {no_line_count})")
            
            # Adaptive delay berdasarkan mode
            if adaptive_ctrl.current_mode == "curve":
                time.sleep(0.03)  # Lebih cepat untuk belokan
            else:
                time.sleep(0.05)  # Normal speed
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Program dihentikan oleh user")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # Cleanup
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        print("[SYSTEM] Cleanup completed")

if __name__ == '__main__':
    main()
