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

def setup_fuzzy_logic_aggressive():
    """Setup FLC dengan rules yang lebih agresif untuk mengatasi robot terdiam"""
    logger.info("Setting up Fuzzy Logic Controller...")
    
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership functions dengan response lebih agresif
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -50])
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -5])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [5, 30, 80])
    error['PL'] = fuzz.trimf(error.universe, [50, 160, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 100, 100])

    # Output dengan range lebih lebar untuk respons agresif
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -30])
    output['LS'] = fuzz.trimf(output.universe, [-50, -20, -5])
    output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 50])
    output['R']  = fuzz.trimf(output.universe, [30, 100, 100])

    # Rules yang lebih agresif
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
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
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
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

def compute_fuzzy_control_enhanced(fuzzy_ctrl, error_val, delta_error):
    """Enhanced fuzzy control dengan error handling yang lebih baik"""
    try:
        # Clamp inputs
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        
        kontrol = fuzzy_ctrl.output['output']
        
        # Enhanced output processing
        if abs(kontrol) < 3:  # Very small output
            kontrol = 0.0
        
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        logger.error(f"FLC computation error: {e}")
        return 0.0

def calculate_motor_pwm_enhanced(kontrol, base_pwm=60, scaling_factor=0.3, min_turn_pwm=35):
    """
    Enhanced PWM calculation untuk mengatasi robot terdiam saat berbelok
    """
    # Aggressive scaling untuk mengatasi robot terdiam
    kontrol_scaled = kontrol * scaling_factor
    
    # Calculate base PWM
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Enhanced: Minimum PWM untuk turning
    if abs(kontrol) > 15:  # Significant turning required
        if kontrol > 0:  # Turn right
            pwm_kiri = max(pwm_kiri, base_pwm + 10)  # Ensure left motor has enough power
            pwm_kanan = max(pwm_kanan, min_turn_pwm)  # Minimum power for right motor
        else:  # Turn left
            pwm_kanan = max(pwm_kanan, base_pwm + 10)  # Ensure right motor has enough power
            pwm_kiri = max(pwm_kiri, min_turn_pwm)    # Minimum power for left motor
    
    # Clamp to safe range
    pwm_kiri = max(20, min(85, pwm_kiri))
    pwm_kanan = max(20, min(85, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            logger.error(f"Serial communication error: {e}")

def main():
    global logger
    logger = Logger()
    
    logger.info("="*60)
    logger.info("LINE FOLLOWING ROBOT - Enhanced Version")
    logger.info("="*60)
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_aggressive()
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
    
    # Parameter yang dapat disesuaikan
    BASE_PWM = 60           # Kecepatan dasar (dinaikkan dari 55)
    SCALING_FACTOR = 0.3    # Faktor scaling (dinaikkan dari 0.2)
    MIN_TURN_PWM = 35       # PWM minimum saat berbelok
    LOG_INTERVAL = 10       # Log setiap N frame
    
    logger.info(f"Configuration:")
    logger.info(f"  Base PWM: {BASE_PWM}")
    logger.info(f"  Scaling Factor: {SCALING_FACTOR}")
    logger.info(f"  Minimum Turn PWM: {MIN_TURN_PWM}")
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
                kontrol = compute_fuzzy_control_enhanced(fuzzy_ctrl, error, delta_error)
                
                # Hitung PWM dengan enhanced method
                pwm_kiri, pwm_kanan = calculate_motor_pwm_enhanced(
                    kontrol, BASE_PWM, SCALING_FACTOR, MIN_TURN_PWM
                )
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Detailed logging dengan PWM yang jelas
                if frame_count % LOG_INTERVAL == 0:
                    turn_direction = "STRAIGHT"
                    if abs(kontrol) > 15:
                        turn_direction = "RIGHT" if kontrol > 0 else "LEFT"
                    
                    # Buat indikator visual untuk PWM
                    pwm_diff = pwm_kiri - pwm_kanan
                    pwm_indicator = "→" if pwm_diff > 5 else "←" if pwm_diff < -5 else "↑"
                    
                    logger.debug(f"Frame {frame_count:4d} | "
                               f"Pos: ({cx:3d},{cy:3d}) | "
                               f"Err: {error:4d} | "
                               f"ΔErr: {delta_error:4d} | "
                               f"FLC: {kontrol:6.1f} | "
                               f"PWM_L: {pwm_kiri:2d} | PWM_R: {pwm_kanan:2d} {pwm_indicator} | "
                               f"Turn: {turn_direction:8s} | "
                               f"Conf: {confidence:.2f} | "
                               f"Src: {source} | "
                               f"Bright: {processed_img['brightness']:.0f}")
                    
                    # Tambahan: Log khusus PWM untuk debugging motor
                    logger.info(f"MOTOR_PWM | LEFT: {pwm_kiri:2d} | RIGHT: {pwm_kanan:2d} | "
                              f"DIFF: {pwm_diff:+3d} | ACTION: {turn_direction}")
                
            else:
                no_line_count += 1
                time_since_line = current_time - last_line_time
                
                # Strategy saat garis hilang
                if time_since_line < 0.5:  # Recently lost line - keep last direction
                    if abs(prev_error) > 40:  # Was turning significantly
                        # Continue turning in last direction
                        emergency_kontrol = np.sign(prev_error) * 60
                        pwm_kiri, pwm_kanan = calculate_motor_pwm_enhanced(
                            emergency_kontrol, BASE_PWM * 0.8, SCALING_FACTOR
                        )
                        send_motor_commands(ser, pwm_kiri, pwm_kanan)
                        
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn(f"Line lost - continuing turn | "
                                      f"PWM_L: {pwm_kiri:2d} | PWM_R: {pwm_kanan:2d} | "
                                      f"Last_Error: {prev_error:+3d}")
                    else:
                        # Go straight slowly
                        straight_pwm = BASE_PWM//2
                        send_motor_commands(ser, straight_pwm, straight_pwm)
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn(f"Line lost - going straight | "
                                      f"PWM_L: {straight_pwm:2d} | PWM_R: {straight_pwm:2d}")
                else:
                    # Long time without line - stop
                    send_motor_commands(ser, 0, 0)
                    if frame_count % (LOG_INTERVAL * 2) == 0:
                        logger.warn(f"Line lost for {time_since_line:.1f}s - STOPPING | "
                                  f"PWM_L: 0 | PWM_R: 0")
            
            # System health monitoring
            if frame_count % 100 == 0:
                logger.info(f"System Status - Frame: {frame_count}, "
                          f"Line Detection Rate: {((100-no_line_count)/100)*100:.0f}%")
            
            time.sleep(0.05)  # 20 FPS
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        send_motor_commands(ser, 0, 0)  # Stop motors
        logger.info("MOTOR_PWM | LEFT: 0 | RIGHT: 0 | DIFF: 0 | ACTION: STOPPED")
        if ser:
            ser.close()
        if picam2:
            picam2.stop()
        logger.info("Cleanup completed - Robot stopped")

if __name__ == '__main__':
    main()
