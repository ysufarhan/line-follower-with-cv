from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def setup_pure_fuzzy_pwm():
    """
    Sistem fuzzy PURE untuk langsung menghasilkan PWM differential
    """
    # Input: Error posisi garis
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    
    # Output: PWM Differential (-50 sampai +50)
    # Negatif = belok kiri, Positif = belok kanan
    pwm_diff = ctrl.Consequent(np.arange(-50, 51, 1), 'pwm_diff')
    
    # MEMBERSHIP FUNCTIONS - Sederhana tapi efektif
    # Error membership
    error['big_left']   = fuzz.trapmf(error.universe, [-160, -160, -80, -40])
    error['small_left'] = fuzz.trimf(error.universe, [-60, -20, 0])
    error['center']     = fuzz.trimf(error.universe, [-15, 0, 15])
    error['small_right']= fuzz.trimf(error.universe, [0, 20, 60])
    error['big_right']  = fuzz.trapmf(error.universe, [40, 80, 160, 160])
    
    # PWM Differential membership
    pwm_diff['strong_left']  = fuzz.trapmf(pwm_diff.universe, [-50, -50, -30, -15])
    pwm_diff['weak_left']    = fuzz.trimf(pwm_diff.universe, [-25, -10, 0])
    pwm_diff['straight']     = fuzz.trimf(pwm_diff.universe, [-5, 0, 5])
    pwm_diff['weak_right']   = fuzz.trimf(pwm_diff.universe, [0, 10, 25])
    pwm_diff['strong_right'] = fuzz.trapmf(pwm_diff.universe, [15, 30, 50, 50])
    
    # FUZZY RULES - Simple dan langsung
    rules = [
        ctrl.Rule(error['big_left'], pwm_diff['strong_left']),
        ctrl.Rule(error['small_left'], pwm_diff['weak_left']),
        ctrl.Rule(error['center'], pwm_diff['straight']),
        ctrl.Rule(error['small_right'], pwm_diff['weak_right']),
        ctrl.Rule(error['big_right'], pwm_diff['strong_right'])
    ]
    
    # Buat sistem kontrol
    control_system = ctrl.ControlSystem(rules)
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    
    return fuzzy_controller

def setup_camera():
    """Inisialisasi kamera"""
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    """Inisialisasi serial communication"""
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)
        print("[UART] Serial port opened successfully")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Failed to open serial: {e}")
        return None

def simple_line_detection(frame):
    """
    Deteksi garis yang disederhanakan
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # ROI (Region of Interest) - bagian bawah frame
    height, width = binary.shape
    roi_height = 80
    roi = binary[height-roi_height:height, :]
    
    return gray, binary, roi

def find_line_center(roi):
    """
    Mencari pusat garis dengan metode sederhana
    """
    # Cari contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0
    
    # Ambil contour terbesar (asumsi ini adalah garis)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Hitung area minimum untuk filter noise
    area = cv2.contourArea(largest_contour)
    if area < 100:  # Filter noise kecil
        return False, 0
    
    # Hitung centroid
    M = cv2.moments(largest_contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        return True, cx
    
    return False, 0

def pure_fuzzy_control(fuzzy_ctrl, error_val):
    """
    Kontrol murni menggunakan fuzzy - tanpa if-else
    """
    try:
        # Batasi error dalam range yang valid
        error_val = max(-160, min(160, error_val))
        
        # Input ke fuzzy system
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.compute()
        
        # Output differential PWM
        pwm_differential = fuzzy_ctrl.output['pwm_diff']
        
        return pwm_differential
        
    except Exception as e:
        print(f"[FUZZY ERROR] {e}")
        return 0.0

def calculate_motor_pwm_pure(pwm_differential, base_speed=65):
    """
    Perhitungan PWM motor PURE tanpa if-else kompleks
    Motor kiri dan kanan dihitung langsung dari differential
    """
    # PWM motor dihitung langsung dari base speed + differential
    pwm_left = base_speed - pwm_differential   # Jika diff negatif, kiri lebih cepat
    pwm_right = base_speed + pwm_differential  # Jika diff positif, kanan lebih cepat
    
    # Clamp PWM dalam range yang aman (30-90%)
    pwm_left = max(30, min(90, pwm_left))
    pwm_right = max(30, min(90, pwm_right))
    
    return int(pwm_left), int(pwm_right)

def send_motor_command(ser, pwm_left, pwm_right):
    """Kirim perintah motor via serial"""
    if ser:
        try:
            command = f"{pwm_left},{pwm_right}\n"
            ser.write(command.encode())
            ser.flush()
            print(f"[MOTOR] L:{pwm_left}% R:{pwm_right}%")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def visualize_simple(frame, line_found, cx, error_val, pwm_diff, pwm_left, pwm_right):
    """Visualisasi sederhana"""
    height, width = frame.shape[:2]
    center_x = width // 2
    
    # Gambar garis referensi tengah
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)
    
    # Gambar ROI
    roi_height = 80
    cv2.rectangle(frame, (0, height-roi_height), (width, height), (0, 255, 255), 2)
    
    if line_found:
        # Gambar posisi garis dan error
        line_y = height - roi_height + 40
        cv2.circle(frame, (cx, line_y), 8, (0, 0, 255), -1)
        cv2.line(frame, (center_x, line_y), (cx, line_y), (0, 255, 0), 3)
        
        # Status info
        status = f"Err:{error_val:3d} | Diff:{pwm_diff:5.1f} | L:{pwm_left} R:{pwm_right}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "LINE NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

def main():
    """
    Program utama - Pure FLC tanpa kompleksitas if-else
    """
    print("=== Pure FLC Line Following Robot ===")
    
    # Setup sistem
    fuzzy_ctrl = setup_pure_fuzzy_pwm()
    picam2 = setup_camera()
    ser = setup_serial()
    
    # Variabel kontrol
    frame_count = 0
    
    # Stabilisasi kamera
    print("Camera stabilizing...")
    time.sleep(2)
    
    try:
        print("=== Starting Main Loop ===")
        while True:
            frame_count += 1
            
            # Capture dan process frame
            frame = picam2.capture_array()
            gray, binary, roi = simple_line_detection(frame)
            
            # Deteksi pusat garis
            line_found, cx = find_line_center(roi)
            
            if line_found:
                # Hitung error (center frame = 160 untuk frame 320px)
                error_val = cx - 160
                
                # Pure fuzzy control - menghasilkan PWM differential
                pwm_differential = pure_fuzzy_control(fuzzy_ctrl, error_val)
                
                # Hitung PWM motor langsung tanpa if-else
                pwm_left, pwm_right = calculate_motor_pwm_pure(pwm_differential, base_speed=65)
                
                # Kirim ke motor
                send_motor_command(ser, pwm_left, pwm_right)
                
                print(f"[CONTROL] Error: {error_val:4d} | PWM_Diff: {pwm_differential:6.2f} | Motors: L{pwm_left} R{pwm_right}")
                
            else:
                # Jika garis tidak ditemukan - tetap bergerak pelan
                print("[WARNING] Line not detected - slow forward")
                send_motor_command(ser, 45, 45)  # Maju pelan
                error_val = 0
                pwm_differential = 0
                pwm_left = pwm_right = 45
            
            # Visualisasi
            frame = visualize_simple(frame, line_found, cx if line_found else 160, 
                                   error_val, pwm_differential, pwm_left, pwm_right)
            
            # Tampilkan frame
            cv2.imshow("Pure FLC Robot", frame)
            cv2.imshow("Binary ROI", roi)
            
            # Kontrol program
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Emergency stop
                send_motor_command(ser, 0, 0)
                print("[STOP] Emergency stop activated")
            
            # Frame rate control (20 FPS)
            time.sleep(0.05)
            
            # Status setiap 100 frame
            if frame_count % 100 == 0:
                print(f"[STATUS] Frame: {frame_count}")
                
    except KeyboardInterrupt:
        print("\n=== Program Interrupted ===")
    
    finally:
        # Cleanup
        print("=== Cleanup ===")
        try:
            send_motor_command(ser, 0, 0)  # Stop motors
            time.sleep(0.5)
        except:
            pass
            
        cv2.destroyAllWindows()
        picam2.stop()
        if ser:
            ser.close()
        print("=== Program Finished ===")

if __name__ == "__main__":
    main()
