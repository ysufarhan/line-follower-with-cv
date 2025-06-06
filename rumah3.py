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

def setup_fuzzy_logic_stronger():
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # Membership functions
    error['NL'] = fuzz.trimf(error.universe, [-200, -200, -60])
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, -5])
    error['Z']  = fuzz.trimf(error.universe, [-30, 0, 30])
    error['PS'] = fuzz.trimf(error.universe, [5, 40, 100])
    error['PL'] = fuzz.trimf(error.universe, [60, 200, 200])

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -150, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -20, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 20, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 150, 150])

    output['L']  = fuzz.trimf(output.universe, [-150, -150, -60])
    output['LS'] = fuzz.trimf(output.universe, [-80, -30, -10])
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [10, 30, 80])
    output['R']  = fuzz.trimf(output.universe, [60, 150, 150])

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
        fuzzy_ctrl.input['error'] = max(-200, min(200, error_val))
        fuzzy_ctrl.input['delta'] = max(-150, min(150, delta_error))
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=55, scaling_factor=0.4):
    # Logika koreksi diperbaiki: kontrol (+) = garis kanan = motor kanan lebih cepat
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled   # motor kiri lambat saat belok kiri
    pwm_kanan = base_pwm - kontrol_scaled  # motor kanan lambat saat belok kanan
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

def draw_simple_overlay(frame, error, kontrol, cx):
    text = f"Err:{error:3d} | Ctrl: {kontrol:5.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    center_x = frame.shape[1] // 2
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255, 0, 0), 2)
    cv2.rectangle(frame, (0, 160), (frame.shape[1], 240), (0, 255, 255), 2)
    if cx > 0:
        cv2.circle(frame, (cx, 200), 6, (0, 0, 255), -1)
    return frame

def main():
    print("[SYSTEM] Starting Robot Line Follower v2")

    fuzzy_ctrl = setup_fuzzy_logic_stronger()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter()

    prev_error = 0
    frame_count = 0

    cv2.namedWindow('Line Follower', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame = picam2.capture_array()
            gray, binary, roi = process_image(frame)
            line_detected, cx, cy = calculate_line_position(roi)

            if line_detected:
                error = cx - 160
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)

                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 20 == 0:
                    print(f"[DEBUG] Error: {error}, Delta: {delta_error}, FLC: {kontrol:.2f}, PWM: {pwm_kiri},{pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            frame_overlay = draw_simple_overlay(frame.copy(), error if line_detected else 0, kontrol if line_detected else 0, cx if line_detected else 0)
            roi_display = cv2.resize(roi, (320, 160))
            cv2.imshow('Line Follower', frame_overlay)
            cv2.imshow('ROI', roi_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[SYSTEM] Keyboard interrupt")
    finally:
        send_motor_commands(ser, 0, 0)
        if ser: ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Shutdown complete")

if __name__ == '__main__':
    main()
