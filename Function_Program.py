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

def process_image(frame):
    """Simple image processing dengan OTSU thresholding"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ROI - bagian bawah untuk line detection
    roi = binary[160:240, :]
    
    return gray, binary, roi

def calculate_line_position(roi):
    """Hitung posisi garis dari ROI"""
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
    """Compute fuzzy control"""
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

def calculate_motor_pwm_direct(kontrol, base_pwm=55, scaling_factor=0.2):
    """Direct PWM calculation dari FLC output - LOGIKA DIPERBAIKI"""
    # PERBAIKAN LOGIKA MOTOR:
    # Kontrol negatif = garis di kiri = belok kiri = motor kanan lebih cepat
    # Kontrol positif = garis di kanan = belok kanan = motor kiri lebih cepat
    kontrol_scaled = kontrol * scaling_factor
    
    pwm_kiri = base_pwm - kontrol_scaled   # Motor kiri: kurangi saat belok kanan
    pwm_kanan = base_pwm + kontrol_scaled  # Motor kanan: tambah saat belok kiri
    
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
    """Draw simple overlay seperti di foto"""
    # Text overlay sederhana
    text = f"Err:{error:3d} | Ctrl: {kontrol:5.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw center line (biru)
    center_x = frame.shape[1] // 2
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255, 0, 0), 2)
    
    # Draw ROI area (kuning)
    cv2.rectangle(frame, (0, 160), (frame.shape[1], 240), (0, 255, 255), 2)
    
    # Draw detected position (merah)
    if cx > 0:
        cv2.circle(frame, (cx, 200), 6, (0, 0, 255), -1)
    
    return frame

def main():
    print("[SYSTEM] Starting Line Following Robot - Simple Display")
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_smooth()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)
    
    # Variabel kontrol
    prev_error = 0
    frame_count = 0
    
    # Setup OpenCV windows
    cv2.namedWindow('Line Following Robot - Improved', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('ROI Binary', cv2.WINDOW_AUTOSIZE)
    
    print("[DISPLAY] OpenCV windows ready - Press 'q' to quit")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            gray, binary, roi = process_image(frame)
            
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
                
                # Hitung PWM dengan logika yang sudah diperbaiki
                pwm_kiri, pwm_kanan = calculate_motor_pwm_direct(kontrol, base_pwm=55, scaling_factor=0.2)
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Debug info
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Error: {error:3d}, Delta: {delta_error:3d}, "
                          f"FLC: {kontrol:5.1f}, PWM: L={pwm_kiri}, R={pwm_kanan}")
                
            else:
                # Garis tidak terdeteksi
                send_motor_commands(ser, 0, 0)
                error = 0
                kontrol = 0.0
                cx = 0
                
                if frame_count % 20 == 0:
                    print("[DEBUG] Line not detected - stopping")
            
            # Display frames
            display_frame = draw_simple_overlay(frame.copy(), error, kontrol, cx)
            cv2.imshow('Line Following Robot - Improved', display_frame)
            
            # Show binary ROI
            roi_display = cv2.resize(roi, (320, 160))  # Resize untuk display
            cv2.imshow('ROI Binary', roi_display)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[SYSTEM] Quit key pressed")
                break
            
            time.sleep(0.05)  # 20 FPS
            
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
        cv2.destroyAllWindows()
        print("[SYSTEM] Cleanup completed")

if __name__ == '__main__':
    main()
