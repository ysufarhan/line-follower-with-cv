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

def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    error.automf(names=['NL', 'NS', 'Z', 'PS', 'PL'])
    delta.automf(names=['NL', 'NS', 'Z', 'PS', 'PL'])
    output.automf(names=['L', 'LS', 'Z', 'RS', 'R'])

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
        fuzzy_ctrl.input['error'] = np.clip(error_val, -160, 160)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -100, 100)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.2):
    kontrol_scaled = kontrol * scaling_factor
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
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)

    prev_error = 0
    frame_count = 0

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
                    print(f"[DEBUG] Error: {error}, Delta: {delta_error}, FLC: {kontrol:.2f}, PWM: {pwm_kiri}, {pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            # Tampilkan tampilan real-time untuk analisis
            frame_with_line = frame.copy()
            cv2.line(frame_with_line, (160, 160), (160, 240), (0, 255, 0), 2)
            cv2.circle(frame_with_line, (cx, cy), 5, (0, 0, 255), -1) if line_detected else None

            cv2.imshow("Camera View", frame_with_line)
            cv2.imshow("Threshold ROI", roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.05)
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
