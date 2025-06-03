#!/usr/bin/env python3
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

# =============================================
# KONFIGURASI SISTEM
# =============================================
class Config:
    # Kamera
    CAM_RESOLUTION = (320, 240)
    ROI_HEIGHT = 80  # Bagian bawah frame (240-80=160)
    
    # Motor
    BASE_PWM = 45           # Kecepatan dasar (30-70)
    PWM_RANGE = (30, 80)     # Batas min/max PWM
    SCALING_FACTOR = 0.3     # Faktor scaling kontrol
    
    # FLC
    ERROR_RANGE = (-160, 160)
    DELTA_RANGE = (-100, 100)
    OUTPUT_RANGE = (-100, 100)
    
    # Serial
    SERIAL_PORT = '/dev/ttyAMA0'
    BAUD_RATE = 115200

# =============================================
# ERROR FILTER (MOVING AVERAGE)
# =============================================
class ErrorFilter:
    def __init__(self, window_size=5):
        self.window = []
        self.window_size = window_size
        
    def filter(self, error):
        self.window.append(error)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return int(np.mean(self.window))

# =============================================
# FUZZY LOGIC CONTROLLER SETUP
# =============================================
def setup_flc():
    # Variabel input/output
    error = ctrl.Antecedent(np.arange(*Config.ERROR_RANGE, 1), 'error')
    delta = ctrl.Antecedent(np.arange(*Config.DELTA_RANGE, 1), 'delta')
    output = ctrl.Consequent(np.arange(*Config.OUTPUT_RANGE, 1), 'output')

    # Membership functions (Gaussian untuk smoothness)
    error['NL'] = fuzz.gaussmf(error.universe, -120, 40)
    error['NS'] = fuzz.gaussmf(error.universe, -60, 20)
    error['Z'] = fuzz.gaussmf(error.universe, 0, 15)
    error['PS'] = fuzz.gaussmf(error.universe, 60, 20)
    error['PL'] = fuzz.gaussmf(error.universe, 120, 40)

    delta['NL'] = fuzz.gaussmf(delta.universe, -80, 30)
    delta['NS'] = fuzz.gaussmf(delta.universe, -30, 15)
    delta['Z'] = fuzz.gaussmf(delta.universe, 0, 10)
    delta['PS'] = fuzz.gaussmf(delta.universe, 30, 15)
    delta['PL'] = fuzz.gaussmf(delta.universe, 80, 30)

    output['L'] = fuzz.trimf(output.universe, [-100, -100, -40])
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -5])
    output['Z'] = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 60])
    output['R'] = fuzz.trimf(output.universe, [40, 100, 100])

    # Rules (optimized untuk respon cepat dan smooth)
    rules = [
        # Rule untuk error besar (respons maksimal)
        ctrl.Rule(error['NL'], output['L']),
        ctrl.Rule(error['PL'], output['R']),
        
        # Rule untuk error sedang
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
        
        # Rule untuk error kecil
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# =============================================
# IMAGE PROCESSING
# =============================================
def process_frame(frame):
    # Normalisasi pencahayaan + CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_norm = clahe.apply(gray)
    
    # Gabungkan Otsu dan Adaptive Threshold
    _, otsu = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_and(otsu, adaptive)
    
    # ROI + Noise Reduction
    roi = combined[Config.CAM_RESOLUTION[1]-Config.ROI_HEIGHT:, :]
    roi = cv2.GaussianBlur(roi, (5,5), 0)
    
    return roi

def get_line_position(roi):
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter kontur kecil
        contours = [c for c in contours if cv2.contourArea(c) > 200]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00']) + (Config.CAM_RESOLUTION[1]-Config.ROI_HEIGHT)
                
                # Smoothing dengan moving average
                if hasattr(get_line_position, 'prev_cx'):
                    cx = 0.7 * cx + 0.3 * get_line_position.prev_cx
                    get_line_position.prev_cx = cx
                else:
                    get_line_position.prev_cx = cx
                
                return True, cx, cy
    return False, 0, 0

# =============================================
# MOTOR CONTROL
# =============================================
def compute_pwm(flc, error, delta_error):
    # Deadzone untuk error kecil
    if abs(error) < 5:
        return 0
    
    # Hitung output FLC
    flc.input['error'] = np.clip(error, *Config.ERROR_RANGE)
    flc.input['delta'] = np.clip(delta_error, *Config.DELTA_RANGE)
    flc.compute()
    kontrol = flc.output['output']
    
    # Dynamic scaling berdasarkan magnitude error
    if abs(error) > 50:
        scaling = Config.SCALING_FACTOR * 1.5
    else:
        scaling = Config.SCALING_FACTOR
    
    # Hitung PWM
    pwm_left = Config.BASE_PWM + kontrol * scaling
    pwm_right = Config.BASE_PWM - kontrol * scaling
    
    # Clamping dan handling deadzone motor
    pwm_min, pwm_max = Config.PWM_RANGE
    pwm_left = np.clip(pwm_left, pwm_min, pwm_max)
    pwm_right = np.clip(pwm_right, pwm_min, pwm_max)
    
    return int(pwm_left), int(pwm_right)

def send_serial(ser, pwm_l, pwm_r):
    if ser is not None:
        try:
            cmd = f"{pwm_l},{pwm_r}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {str(e)}")

# =============================================
# MAIN PROGRAM
# =============================================
def main():
    # Inisialisasi
    print("[INFO] Starting Line Follower Robot")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_RESOLUTION[1])
    
    try:
        ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=1)
        print("[INFO] Serial port initialized")
    except:
        ser = None
        print("[WARNING] Serial port failed, running in simulation mode")
    
    flc = setup_flc()
    error_filter = ErrorFilter()
    prev_error = 0
    lost_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process image
            roi = process_frame(frame)
            line_detected, cx, cy = get_line_position(roi)
            
            if line_detected:
                lost_count = 0
                
                # Hitung error
                error = cx - Config.CAM_RESOLUTION[0] // 2
                error = error_filter.filter(error)
                delta_error = error - prev_error
                prev_error = error
                
                # Kontrol motor
                pwm_l, pwm_r = compute_pwm(flc, error, delta_error)
                send_serial(ser, pwm_l, pwm_r)
                
                # Debug visual
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(frame, f"Error: {error}", (10,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                lost_count += 1
                if lost_count > 10:  # Jika 10 frame tidak detect garis
                    send_serial(ser, 0, 0)  # Stop
                else:
                    # Cari garis dengan memutar perlahan
                    send_serial(ser, 20, -20)
            
            # Hitung FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # Tampilkan debug
            cv2.imshow("ROI", roi)
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Pertahankan FPS konstan
            elapsed = time.time() - start_time
            time.sleep(max(0.05 - elapsed, 0))  # Target 20 FPS
    
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh user")
    finally:
        send_serial(ser, 0, 0)  # Stop motor
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
        print("[INFO] Cleanup completed")

if __name__ == "__main__":
    main()
