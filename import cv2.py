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

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

def setup_fuzzy_logic_mix():
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -60])
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, -5])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [5, 40, 100])
    error['PL'] = fuzz.trimf(error.universe, [60, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -20, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 20, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 100])

    output['L']  = fuzz.trimf(output.universe, [-100, -100, -40])
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -5])
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 60])
    output['R']  = fuzz.trimf(output.universe, [40, 100, 100])

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

        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
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
        print("[UART] Serial opened")
        return ser
    except Exception as e:
        print(f"[UART ERROR] {e}")
        return None

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary[160:240, :]

def calculate_line_position(roi):
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    M = cv2.moments(cleaned)
    if M['m00'] > 100:
        cx = int(M['m10'] / M['m00'])
        return True, cx
    return False, 0

def compute_fuzzy_output(fuzzy_ctrl, error, delta):
    try:
        error = max(-160, min(160, error))
        delta = max(-100, min(100, delta))
        fuzzy_ctrl.input['error'] = error
        fuzzy_ctrl.input['delta'] = delta
        fuzzy_ctrl.compute()
        output = fuzzy_ctrl.output['output']
        return 0 if abs(error) < 5 else np.clip(output, -100, 100)  # DEAD ZONE
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=55, scaling=0.25):
    k_scaled = kontrol * scaling
    pwm_left = max(25, min(80, base_pwm + k_scaled))
    pwm_right = max(25, min(80, base_pwm - k_scaled))
    return int(pwm_left), int(pwm_right)

def send_motor_commands(ser, left, right):
    if ser:
        try:
            cmd = f"{left},{right}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def main():
    print("[SYSTEM] Starting Mixed Fuzzy Line Follower")

    fuzzy_ctrl = setup_fuzzy_logic_mix()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)

    prev_error = 0
    frame_count = 0

    try:
        while True:
            frame_count += 1
            frame = picam2.capture_array()
            roi = process_image(frame)

            detected, cx = calculate_line_position(roi)
            if detected:
                error = cx - 160
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_output(fuzzy_ctrl, error, delta_error)
                pwm_left, pwm_right = calculate_motor_pwm(kontrol)

                send_motor_commands(ser, pwm_left, pwm_right)

                if frame_count % 20 == 0:
                    print(f"[DEBUG] Err: {error}, ΔErr: {delta_error}, FLC: {kontrol:.1f}, PWM: {pwm_left},{pwm_right}")
            else:
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Line not detected")

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by user")
    finally:
        send_motor_commands(ser, 0, 0)
        if ser: ser.close()
        picam2.stop()
        print("[SYSTEM] Shutdown complete")

if __name__ == '__main__':
    main()
