from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def setup_fuzzy_logic():
    """
    Konfigurasi sistem fuzzy logic untuk kontrol robot
    """
    # Buat variabel fuzzy
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')
    
    # Definisikan membership functions
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -80])
    error['NS'] = fuzz.trimf(error.universe, [-160, -80, 0])
    error['Z']  = fuzz.trimf(error.universe, [-40, 0, 40])
    error['PS'] = fuzz.trimf(error.universe, [0, 80, 160])
    error['PL'] = fuzz.trimf(error.universe, [80, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-100, -50, 0])
    delta['Z']  = fuzz.trimf(delta.universe, [-20, 0, 20])
    delta['PS'] = fuzz.trimf(delta.universe, [0, 50, 100])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 100, 100])

    output['L']  = fuzz.trimf(output.universe, [-100, -100, -50])
    output['LS'] = fuzz.trimf(output.universe, [-100, -50, 0])
    output['Z']  = fuzz.trimf(output.universe, [-20, 0, 20])
    output['RS'] = fuzz.trimf(output.universe, [0, 50, 100])
    output['R']  = fuzz.trimf(output.universe, [50, 100, 100])
    
    # Definisikan rule base
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['L']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['R']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),
    ]
    
    # Buat sistem kontrol
    control_system = ctrl.ControlSystem(rules)
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    
    return fuzzy_controller

def setup_camera():
    """
    Konfigurasi dan inisialisasi kamera
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    """
    Konfigurasi dan inisialisasi komunikasi serial
    """
    try:
        ser = serial.Serial('/dev/serial0', 115200)
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame):
    """
    Memproses frame untuk mendeteksi jalur/garis
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ROI bagian bawah (tempat jalur biasanya terlihat)
    roi = binary[180:240, :]
    
    return gray, binary, roi

def calculate_line_position(roi):
    """
    Menghitung posisi garis dari ROI menggunakan moments
    """
    M = cv2.moments(roi)
    
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + 180  # Tambah offset karena ROI
        return True, cx, cy
    else:
        return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    """
    Menghitung output kontrol berdasarkan fuzzy logic
    """
    try:
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = int(fuzzy_ctrl.output['output'])
        print(f"[LOG] Error: {error_val} | Delta: {delta_error} | Output Fuzzy: {kontrol}")
        return kontrol
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=60):
    """
    Menghitung PWM untuk motor berdasarkan nilai kontrol
    """
    pwm_kiri = base_pwm - kontrol
    pwm_kanan = base_pwm + kontrol
    
    # Batasi PWM ke 0–100
    pwm_kiri = max(0, min(100, pwm_kiri))
    pwm_kanan = max(0, min(100, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    """
    Mengirim perintah ke motor melalui serial
    """
    if ser:
        try:
            ser.write(f"{pwm_kiri},{pwm_kanan}\n".encode())
        except Exception as e:
            print(f"[SERIAL WRITE ERROR] {e}")

def visualize_tracking(frame, line_detected, cx=0, cy=0, error_val=0, kontrol=0):
    """
    Membuat visualisasi tracking pada frame
    """
    if line_detected:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
        cv2.putText(frame, f"Err:{error_val} | Ctrl:{kontrol}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Garis tidak ditemukan", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    """
    Fungsi utama program
    """
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    
    prev_error = 0
    
    try:
        while True:
            # Ambil dan proses gambar
            frame = picam2.capture_array()
            _, _, roi = process_image(frame)
            
            # Deteksi posisi garis
            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                # Hitung error dan kontrol
                error_val = cx - 160
                delta_error = error_val - prev_error
                
                # Komputasi fuzzy
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error)
                
                # Hitung PWM motor
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                
                # Kirim ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                prev_error = error_val
            else:
                # Garis tidak terdeteksi, robot berhenti
                send_motor_commands(ser, 0, 0)
                kontrol = 0
                error_val = 0
            
            # Visualisasi
            frame = visualize_tracking(frame, line_detected, cx, cy, error_val, kontrol)
            cv2.imshow("Deteksi Garis", frame)
            cv2.imshow("ROI", roi)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Dihentikan oleh pengguna")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        picam2.stop()
        if ser:
            ser.close()

if __name__ == "__main__":
    main()