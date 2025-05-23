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

def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -80])
    error['NS'] = fuzz.trimf(error.universe, [-120, -50, -10])
    error['Z']  = fuzz.trimf(error.universe, [-40, 0, 40])
    error['PS'] = fuzz.trimf(error.universe, [10, 50, 120])
    error['PL'] = fuzz.trimf(error.universe, [80, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-70, -25, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-20, 0, 20])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 25, 70])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 100, 100])

    output['L']  = fuzz.trimf(output.universe, [-100, -100, -50])
    output['LS'] = fuzz.trimf(output.universe, [-70, -35, -10])
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])
    output['RS'] = fuzz.trimf(output.universe, [10, 35, 70])
    output['R']  = fuzz.trimf(output.universe, [50, 100, 100])

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

        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

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

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error, dead_zone=25):
    try:
        if abs(error_val) <= dead_zone and abs(delta_error) <= 15:
            return 0.0
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = fuzzy_ctrl.output['output']
        if abs(kontrol) < 8:
            kontrol = 0.0
        return np.clip(kontrol, -80, 80)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=55):
    if abs(kontrol) <= 10:
        return base_pwm, base_pwm
    elif abs(kontrol) <= 30:
        kontrol_scaled = kontrol * 0.08
    elif abs(kontrol) <= 60:
        kontrol_scaled = kontrol * 0.12
    else:
        kontrol_scaled = kontrol * 0.15
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    pwm_kiri = max(30, min(75, pwm_kiri))
    pwm_kanan = max(30, min(75, pwm_kanan))
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
    error_filter = ErrorFilter()
    prev_error = 0
    
    while True:
        frame = picam2.capture_array()
        _, _, roi = process_image(frame)
        line_detected, cx, cy = calculate_line_position(roi)
        if line_detected:
            error = cx - 160
            error = error_filter.filter_error(error)
            delta_error = error - prev_error
            prev_error = error
            kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
            pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
            send_motor_commands(ser, pwm_kiri, pwm_kanan)
        else:
            send_motor_commands(ser, 0, 0)
        time.sleep(0.05)

if __name__ == '__main__':
    main()
