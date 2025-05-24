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

def setup_fuzzy_logic_smooth():
    """Setup FLC dengan membership function yang lebih smooth untuk output kontinyu"""
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions dengan overlap lebih besar untuk transisi halus
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -60])
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, -5])
    error['Z']  = fuzz.trimf(error.universe, [-30, 0, 30])
    error['PS'] = fuzz.trimf(error.universe, [5, 40, 100])
    error['PL'] = fuzz.trimf(error.universe, [60, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -20, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 20, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 100])

    # Output dengan range yang lebih halus
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -40])
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -5])
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 60])
    output['R']  = fuzz.trimf(output.universe, [40, 100, 100])

    # Rules yang lebih balanced
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

def compute_fuzzy_control_smooth(fuzzy_ctrl, error_val, delta_error):
    """Compute fuzzy control tanpa dead zone untuk output kontinyu"""
    try:
        # Clamp input values
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        
        kontrol = fuzzy_ctrl.output['output']
        
        # Clip output untuk keamanan
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm_direct(kontrol, base_pwm=55, scaling_factor=0.2):
    """
    Direct PWM calculation dari FLC output tanpa if-else
    
    Args:
        kontrol: Output dari FLC (-100 to 100)
        base_pwm: Base speed untuk kedua motor
        scaling_factor: Faktor pengali untuk kontrol (0.1 - 0.3)
    """
    # Direct scaling tanpa kondisi if-else
    kontrol_scaled = kontrol * scaling_factor
    
    # Hitung PWM untuk masing-masing motor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Clamp PWM values ke range yang aman
    pwm_kiri = max(25, min(80, pwm_kiri))
    pwm_kanan = max(25, min(80, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def calculate_motor_pwm_sigmoid(kontrol, base_pwm=55, max_scale=0.25, steepness=0.05):
    """
    Alternatif: Sigmoid scaling untuk respons lebih halus
    """
    # Sigmoid function untuk scaling yang halus
    sigmoid_val = max_scale * (2 / (1 + np.exp(-steepness * abs(kontrol))) - 1)
    kontrol_scaled = np.sign(kontrol) * sigmoid_val * abs(kontrol)
    
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    pwm_kiri = max(25, min(80, pwm_kiri))
    pwm_kanan = max(25, min(80, pwm_kanan))
    
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
    print("[SYSTEM] Starting Line Following Robot - Smooth FLC Mode")
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_smooth()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)
    
    # Variabel kontrol
    prev_error = 0
    frame_count = 0
    
    # Parameter yang bisa disesuaikan
    BASE_PWM = 55           # Kecepatan dasar (30-70)
    SCALING_FACTOR = 0.2    # Faktor scaling kontrol (0.1-0.3)
    USE_SIGMOID = False     # True untuk menggunakan sigmoid scaling
    
    print(f"[CONFIG] Base PWM: {BASE_PWM}, Scaling: {SCALING_FACTOR}")
    print(f"[CONFIG] Scaling Method: {'Sigmoid' if USE_SIGMOID else 'Linear Direct'}")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            _, _, roi = process_image(frame)
            
            # Deteksi posisi garis
            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                # Hitung error dan delta error
                error = cx - 160  # Setpoint di tengah (160)
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error
                
                # Compute FLC output
                kontrol = compute_fuzzy_control_smooth(fuzzy_ctrl, error, delta_error)
                
                # Hitung PWM langsung dari FLC output
                if USE_SIGMOID:
                    pwm_kiri, pwm_kanan = calculate_motor_pwm_sigmoid(kontrol, BASE_PWM)
                else:
                    pwm_kiri, pwm_kanan = calculate_motor_pwm_direct(kontrol, BASE_PWM, SCALING_FACTOR)
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Debug info setiap 20 frame
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Error: {error:3d}, Delta: {delta_error:3d}, "
                          f"FLC: {kontrol:5.1f}, PWM: L={pwm_kiri}, R={pwm_kanan}")
                
            else:
                # Garis tidak terdeteksi - stop atau cari garis
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Line not detected - stopping")
            
            # Delay untuk stabilitas
            time.sleep(0.05)  # 20 FPS
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Program dihentikan oleh user")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # Cleanup
        send_motor_commands(ser, 0, 0)  # Stop motors
        if ser:
            ser.close()
        picam2.stop()
        print("[SYSTEM] Cleanup completed")

if __name__ == '__main__':
    main()
