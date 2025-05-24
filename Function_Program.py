from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
import os

class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

class Logger:
    def __init__(self):
        self.start_time = time.time()
        
    def log(self, level, message):
        timestamp = time.time() - self.start_time
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] [{level:5s}] {message}")
        
    def info(self, message):
        self.log("INFO", message)
        
    def debug(self, message):
        self.log("DEBUG", message)
        
    def warn(self, message):
        self.log("WARN", message)
        
    def error(self, message):
        self.log("ERROR", message)

def setup_fuzzy_logic_smooth():
    """Setup FLC dengan response yang lebih halus untuk kecepatan rendah"""
    logger.info("Setting up Fuzzy Logic Controller...")
    
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions dengan response yang lebih halus
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -60])
    error['NS'] = fuzz.trimf(error.universe, [-90, -40, -10])
    error['Z']  = fuzz.trimf(error.universe, [-30, 0, 30])
    error['PS'] = fuzz.trimf(error.universe, [10, 40, 90])
    error['PL'] = fuzz.trimf(error.universe, [60, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -20, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 20, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 100])

    # Output dengan range yang lebih kecil untuk gerakan halus
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -40])
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -8])
    output['Z']  = fuzz.trimf(output.universe, [-12, 0, 12])
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 60])
    output['R']  = fuzz.trimf(output.universe, [40, 100, 100])

    # Rules yang lebih konservatif
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    logger.info("Fuzzy Logic Controller setup completed")
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    logger.info("Initializing camera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (320, 240)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Warm up camera
        logger.info("Camera initialized successfully")
        return picam2
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return None

def setup_serial():
    logger.info("Setting up serial communication...")
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        logger.info("Serial port opened successfully")
        return ser
    except Exception as e:
        logger.error(f"Failed to open serial port: {e}")
        return None

def advanced_roi_thresholding(frame):
    """
    Advanced ROI thresholding dengan multiple methods dan validasi
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define multiple ROI zones
    roi_bottom = gray[180:240, :]      # Bottom strip (most important)
    roi_middle = gray[140:200, :]      # Middle strip  
    roi_full = gray[120:240, :]        # Full lower area
    
    # Method 1: Adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Method 2: OTSU thresholding
    _, otsu_thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Method 3: Dynamic thresholding based on lighting
    mean_brightness = np.mean(gray)
    if mean_brightness < 100:  # Dark condition
        _, dynamic_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    elif mean_brightness > 180:  # Bright condition
        _, dynamic_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    else:  # Normal condition
        _, dynamic_thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    # Combine thresholding methods
    combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
    combined = cv2.bitwise_or(combined, dynamic_thresh)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Extract ROI zones
    roi_bottom_clean = combined[180:240, :]
    roi_middle_clean = combined[140:200, :]
    
    return {
        'gray': gray,
        'combined': combined,
        'roi_bottom': roi_bottom_clean,
        'roi_middle': roi_middle_clean,
        'brightness': mean_brightness
    }

def calculate_line_position_robust(processed_img):
    """
    Robust line position calculation dengan multiple validation
    """
    roi_bottom = processed_img['roi_bottom']
    roi_middle = processed_img['roi_middle']
    brightness = processed_img['brightness']
    
    results = []
    
    # Try bottom ROI first (most reliable)
    M_bottom = cv2.moments(roi_bottom)
    if M_bottom['m00'] > 200:  # Sufficient white pixels
        cx_bottom = int(M_bottom['m10'] / M_bottom['m00'])
        cy_bottom = int(M_bottom['m01'] / M_bottom['m00']) + 180
        confidence_bottom = min(M_bottom['m00'] / 1000, 1.0)
        results.append({
            'cx': cx_bottom, 
            'cy': cy_bottom, 
            'confidence': confidence_bottom,
            'source': 'bottom'
        })
    
    # Try middle ROI as backup
    M_middle = cv2.moments(roi_middle)
    if M_middle['m00'] > 150:
        cx_middle = int(M_middle['m10'] / M_middle['m00'])
        cy_middle = int(M_middle['m01'] / M_middle['m00']) + 140
        confidence_middle = min(M_middle['m00'] / 800, 1.0) * 0.8  # Lower confidence
        results.append({
            'cx': cx_middle, 
            'cy': cy_middle, 
            'confidence': confidence_middle,
            'source': 'middle'
        })
    
    # Select best result
    if results:
        best_result = max(results, key=lambda x: x['confidence'])
        return True, best_result['cx'], best_result['cy'], best_result['confidence'], best_result['source']
    
    return False, 0, 0, 0.0, 'none'

def compute_fuzzy_control_smooth(fuzzy_ctrl, error_val, delta_error):
    """Smooth fuzzy control untuk kecepatan rendah"""
    try:
        # Clamp inputs
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        
        kontrol = fuzzy_ctrl.output['output']
        
        # Smooth output processing untuk gerakan halus
        if abs(kontrol) < 5:  # Dead zone lebih besar
            kontrol = 0.0
        
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        logger.error(f"FLC computation error: {e}")
        return 0.0

def calculate_motor_pwm_slow(kontrol, base_pwm=35, scaling_factor=0.15):
    """
    PWM calculation untuk kecepatan rendah dan gerakan halus
    """
    # Scaling yang lebih kecil untuk gerakan halus
    kontrol_scaled = kontrol * scaling_factor
    
    # Calculate PWM dengan base yang lebih rendah
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Minimum PWM untuk memastikan motor bergerak
    min_pwm = 15
    
    # Clamp to safe range dengan maksimum yang lebih rendah
    pwm_kiri = max(min_pwm, min(50, pwm_kiri))
    pwm_kanan = max(min_pwm, min(50, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            logger.error(f"Serial communication error: {e}")

def visualize_tracking(frame, line_detected, cx=0, cy=0, processed_img=None):
    """
    Membuat visualisasi tracking pada frame dengan ROI overlay
    """
    # Draw center line
    cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
    
    # Draw ROI boundaries
    cv2.rectangle(frame, (0, 120), (320, 240), (0, 255, 255), 1)  # Full ROI
    cv2.rectangle(frame, (0, 140), (320, 200), (255, 255, 0), 1)  # Middle ROI
    cv2.rectangle(frame, (0, 180), (320, 240), (0, 255, 0), 2)   # Bottom ROI
    
    if line_detected:
        # Draw detected line center
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # Draw line from center to detected point
        cv2.line(frame, (160, 200), (cx, cy), (0, 0, 255), 2)
    
    return frame

def main():
    global logger
    logger = Logger()
    
    logger.info("="*60)
    logger.info("LINE FOLLOWING ROBOT - Reduced Speed Version")
    logger.info("="*60)
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_smooth()
    picam2 = setup_camera()
    ser = setup_serial()
    
    if not picam2:
        logger.error("Cannot start without camera")
        return
    
    error_filter = ErrorFilter(window_size=3)
    
    # Variabel kontrol
    prev_error = 0
    frame_count = 0
    last_line_time = time.time()
    no_line_count = 0
    
    # Parameter untuk kecepatan rendah
    BASE_PWM = 35           # Kecepatan dasar dikurangi drastis
    SCALING_FACTOR = 0.15   # Faktor scaling dikurangi
    LOG_INTERVAL = 20       # Log lebih jarang
    
    logger.info(f"Configuration (SLOW MODE):")
    logger.info(f"  Base PWM: {BASE_PWM}")
    logger.info(f"  Scaling Factor: {SCALING_FACTOR}")
    logger.info(f"  Log Interval: {LOG_INTERVAL} frames")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            processed_img = advanced_roi_thresholding(frame)
            
            # Deteksi posisi garis dengan robust method
            line_detected, cx, cy, confidence, source = calculate_line_position_robust(processed_img)
            
            current_time = time.time()
            
            if line_detected:
                last_line_time = current_time
                no_line_count = 0
                
                # Hitung error dan delta error
                error = cx - 160  # Setpoint di tengah (160)
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error
                
                # Compute FLC output
                kontrol = compute_fuzzy_control_smooth(fuzzy_ctrl, error, delta_error)
                
                # Hitung PWM dengan reduced speed method
                pwm_kiri, pwm_kanan = calculate_motor_pwm_slow(
                    kontrol, BASE_PWM, SCALING_FACTOR
                )
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Simplified logging - hanya error, delta error, Motor L/R, Fuzzy Out
                if frame_count % LOG_INTERVAL == 0:
                    logger.info(f"Error: {error:4d} | "
                              f"Delta: {delta_error:4d} | "
                              f"Motor L: {pwm_kiri:2d} | "
                              f"Motor R: {pwm_kanan:2d} | "
                              f"Fuzzy Out: {kontrol:6.1f}")
                
            else:
                no_line_count += 1
                time_since_line = current_time - last_line_time
                
                # Strategy saat garis hilang - lebih konservatif
                if time_since_line < 1.0:  # Keep last direction briefly
                    if abs(prev_error) > 30:
                        # Continue turning in last direction dengan kecepatan rendah
                        emergency_kontrol = np.sign(prev_error) * 30
                        pwm_kiri, pwm_kanan = calculate_motor_pwm_slow(
                            emergency_kontrol, BASE_PWM * 0.7, SCALING_FACTOR
                        )
                        send_motor_commands(ser, pwm_kiri, pwm_kanan)
                        
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn(f"Line lost - continue turn | "
                                      f"Motor L: {pwm_kiri:2d} | Motor R: {pwm_kanan:2d}")
                    else:
                        # Go straight slowly
                        straight_pwm = BASE_PWM//2
                        send_motor_commands(ser, straight_pwm, straight_pwm)
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn(f"Line lost - straight | "
                                      f"Motor L: {straight_pwm:2d} | Motor R: {straight_pwm:2d}")
                else:
                    # Stop after longer time
                    send_motor_commands(ser, 0, 0)
                    if frame_count % (LOG_INTERVAL * 2) == 0:
                        logger.warn(f"Line lost {time_since_line:.1f}s - STOP")
            
            # Visualisasi dengan ROI display
            frame_vis = visualize_tracking(frame, line_detected, cx, cy, processed_img)
            cv2.imshow("Line Following - Original", frame_vis)
            cv2.imshow("ROI Thresholding - Combined", processed_img['combined'])
            cv2.imshow("ROI Bottom", processed_img['roi_bottom'])
            cv2.imshow("ROI Middle", processed_img['roi_middle'])
            
            # System health monitoring
            if frame_count % 200 == 0:
                logger.info(f"System Status - Frame: {frame_count}")
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Program terminated by user (q key)")
                break
            
            time.sleep(0.1)  # 10 FPS untuk kecepatan rendah
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        send_motor_commands(ser, 0, 0)  # Stop motors
        logger.info("Motors stopped")
        if ser:
            ser.close()
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

if __name__ == '__main__':
    main()
