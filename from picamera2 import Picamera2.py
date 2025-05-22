from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

def setup_fuzzy_logic():
    """
    Sistem fuzzy logic yang dioptimasi untuk kecepatan rendah
    """
    # Variabel input/output dengan range lebih kecil
    error = ctrl.Antecedent(np.arange(-80, 81, 1), 'error')  # Diperkecil dari ±160
    delta = ctrl.Antecedent(np.arange(-40, 41, 1), 'delta')   # Diperkecil dari ±100
    output = ctrl.Consequent(np.arange(-50, 51, 1), 'output') # Output lebih halus (±50)

    # Membership functions (ada yang pakai trapmf untuk respons lebih smooth)
    error['NL'] = fuzz.trimf(error.universe, [-80, -80, -40])
    error['NS'] = fuzz.trapmf(error.universe, [-50, -30, -10, 0])
    error['Z'] = fuzz.trimf(error.universe, [-15, 0, 15])
    error['PS'] = fuzz.trapmf(error.universe, [0, 10, 30, 50])
    error['PL'] = fuzz.trimf(error.universe, [40, 80, 80])

    delta['NL'] = fuzz.trimf(delta.universe, [-40, -40, -20])
    delta['NS'] = fuzz.trimf(delta.universe, [-25, -10, 0])
    delta['Z'] = fuzz.trimf(delta.universe, [-5, 0, 5])
    delta['PS'] = fuzz.trimf(delta.universe, [0, 10, 25])
    delta['PL'] = fuzz.trimf(delta.universe, [20, 40, 40])

    output['L'] = fuzz.trimf(output.universe, [-50, -50, -25])
    output['LS'] = fuzz.trimf(output.universe, [-30, -15, 0])
    output['Z'] = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [0, 15, 30])
    output['R'] = fuzz.trimf(output.universe, [25, 50, 50])

    # Rules yang lebih halus untuk kecepatan rendah
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),  # Koreksi lebih lembut
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),  # Koreksi lebih lembut
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),
        # Tambahan rule untuk error kecil
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
    ]

    control_system = ctrl.ControlSystem(rules)
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    return fuzzy_controller

def setup_camera():
    """Konfigurasi kamera dengan eksposur tetap untuk hindari motion blur"""
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (320, 240)},
        controls={"ExposureTime": 10000, "AnalogueGain": 1.0}  # Eksposur pendek
    )
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame):
    """Preprocessing gambar dengan filter tambahan untuk kurangi noise"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Kernel lebih besar
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = binary[200:240, :]  # Fokus ke area paling bawah
    return gray, binary, roi

def calculate_line_position(roi):
    """Deteksi garis dengan kontur untuk hasil lebih akurat"""
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) + 200  # Adjust y-coordinate
            return True, cx, cy
    return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = fuzzy_ctrl.output['output']
        print(f"[FLC] Error: {error_val:4d} | Delta: {delta_error:4d} | Output: {kontrol:5.1f}")
        return kontrol
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=40, max_pwm=60):
    """
    PWM dioptimasi untuk kecepatan rendah:
    - base_pwm=40 (default lebih rendah)
    - max_pwm=60 (dibatasi)
    - min_pwm=20 (motor tidak mati)
    """
    pwm_kiri = int(np.clip(base_pwm - kontrol, 20, max_pwm))
    pwm_kanan = int(np.clip(base_pwm + kontrol, 20, max_pwm))
    print(f"[PWM] Kiri: {pwm_kiri:3d}% | Kanan: {pwm_kanan:3d}%")
    return pwm_kiri, pwm_kanan

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
            print(f"[UART] Sent: {cmd.strip()}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def visualize_tracking(frame, line_detected, cx=0, cy=0, error_val=0, kontrol=0):
    """Visualisasi dengan informasi lebih detail"""
    if line_detected:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 1)
    status_text = f"Error: {error_val} | Ctrl: {kontrol:.1f} | Line: {'Found' if line_detected else 'Lost'}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def main():
    # Inisialisasi
    print("Inisialisasi sistem...")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    
    prev_error = 0
    line_lost_timeout = 0
    MAX_LOST_TIME = 2.0
    
    try:
        print("Memulai loop utama (Tekan 'q' untuk keluar)...")
        while True:
            # Ambil gambar
            frame = picam2.capture_array()
            _, _, roi = process_image(frame)
            
            # Deteksi garis
            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                line_lost_timeout = 0
                error_val = cx - 160
                delta_error = error_val - prev_error
                
                # Fuzzy control
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error)
                
                # Hitung PWM (dengan batasan kecepatan)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol, base_pwm=40, max_pwm=60)
                
                # Kirim ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                prev_error = error_val
            else:
                line_lost_timeout += 0.1
                if line_lost_timeout >= MAX_LOST_TIME:
                    send_motor_commands(ser, 0, 0)
                    print("[WARN] Garis hilang, motor dihentikan")
            
            # Visualisasi
            frame = visualize_tracking(frame, line_detected, cx, cy, error_val, kontrol)
            cv2.imshow("Line Follower", frame)
            cv2.imshow("ROI", roi)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Delay untuk stabilisasi kamera
            
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna")
    finally:
        send_motor_commands(ser, 0, 0)
        cv2.destroyAllWindows()
        picam2.stop()
        if ser:
            ser.close()
        print("Program selesai")

if __name__ == "__main__":
    main()
