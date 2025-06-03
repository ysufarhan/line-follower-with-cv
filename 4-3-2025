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
    # Definisi universe yang diperluas untuk responsivitas lebih baik
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS - Fine-tuned untuk mengurangi error
    # ERROR - Lebih sensitif pada zona kecil untuk presisi tinggi
    error['NL'] = fuzz.trimf(error.universe, [-200, -160, -70])
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, -8])
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])      # Zona netral lebih sempit
    error['PS'] = fuzz.trimf(error.universe, [8, 40, 100])
    error['PL'] = fuzz.trimf(error.universe, [70, 160, 200])

    # DELTA - Lebih agresif untuk koreksi cepat
    delta['NL'] = fuzz.trimf(delta.universe, [-150, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -25, -4])
    delta['Z']  = fuzz.trimf(delta.universe, [-8, 0, 8])        # Zona netral lebih sempit
    delta['PS'] = fuzz.trimf(delta.universe, [4, 25, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 150])

    # OUTPUT - Lebih agresif pada koreksi kecil dan besar
    output['L']  = fuzz.trimf(output.universe, [-150, -110, -75])
    output['LS'] = fuzz.trimf(output.universe, [-85, -45, -12])
    output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])      # Zona netral sangat sempit
    output['RS'] = fuzz.trimf(output.universe, [12, 45, 85])
    output['R']  = fuzz.trimf(output.universe, [75, 110, 150])

    # Rules yang sama seperti sebelumnya
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
        # Sesuaikan range dengan universe yang baru
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.4):
    # Tingkatkan scaling_factor untuk koreksi lebih agresif
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Pastikan PWM tidak terlalu rendah yang bisa menyebabkan motor berhenti
    pwm_kiri = max(30, min(85, pwm_kiri))  
    pwm_kanan = max(30, min(85, pwm_kanan))
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
    error_filter = ErrorFilter(window_size=1)  # Hilangkan filtering untuk respons maksimal

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

                if frame_count % 10 == 0:  # Monitoring lebih sering untuk analisis
                    print(f"[DEBUG] Error: {error:4d}, Delta: {delta_error:4d}, FLC: {kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
                    if abs(error) > 50:
                        print(f"[WARN]  Error besar terdeteksi: {error}")  # Warning untuk error besar
            else:
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            # Tampilkan tampilan real-time untuk analisis
            frame_with_line = frame.copy()
            cv2.line(frame_with_line, (160, 160), (160, 240), (0, 255, 0), 2)
            if line_detected:
                cv2.circle(frame_with_line, (cx, cy), 5, (0, 0, 255), -1)
                # Tambah indikator error
                cv2.putText(frame_with_line, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Camera View", frame_with_line)
            cv2.imshow("Threshold ROI", roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.03)  # Sedikit lebih cepat untuk responsivitas
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
