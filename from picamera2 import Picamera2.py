from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
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
        print(f"[{level:5s}] {message}")
        
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
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Method 1: Adaptive thresholding
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
        'adaptive': adaptive_thresh,
        'otsu': otsu_thresh,
        'dynamic': dynamic_thresh,
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
        
        # Enhanced output processing - lebih agresif
        if abs(kontrol) < 5:  # Reduced threshold for more responsive control
            kontrol = 0.0
        
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        logger.error(f"FLC computation error: {e}")
        return 0.0

def calculate_motor_pwm_enhanced(kontrol, base_pwm=70, scaling_factor=0.4, min_turn_pwm=45):
    """
    Enhanced PWM calculation untuk mengatasi robot terdiam saat berbelok
    PWM values dinaikkan untuk mengatasi masalah robot terdiam
    """
    # More aggressive scaling
    kontrol_scaled = kontrol * scaling_factor
    
    # Calculate base PWM
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Enhanced: Minimum PWM untuk turning dengan nilai lebih tinggi
    if abs(kontrol) > 10:  # More sensitive turning threshold
        if kontrol > 0:  # Turn right
            pwm_kiri = max(pwm_kiri, base_pwm + 15)  # Boost left motor
            pwm_kanan = max(pwm_kanan, min_turn_pwm)  # Minimum power for right motor
        else:  # Turn left
            pwm_kanan = max(pwm_kanan, base_pwm + 15)  # Boost right motor
            pwm_kiri = max(pwm_kiri, min_turn_pwm)     # Minimum power for left motor
    
    # Clamp to safe range with higher minimum
    pwm_kiri = max(30, min(90, pwm_kiri))   # Raised minimum from 20 to 30
    pwm_kanan = max(30, min(90, pwm_kanan)) # Raised minimum from 20 to 30
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            logger.error(f"Serial communication error: {e}")

def display_thresholding_results(processed_img, frame_count):
    """
    Menampilkan hasil thresholding untuk debugging
    """
    if frame_count % 5 == 0:  # Show every 5th frame to reduce processing load
        # Create display windows
        combined_display = np.hstack([
            processed_img['adaptive'], 
            processed_img['otsu'], 
            processed_img['dynamic']
        ])
        
        # Add labels
        cv2.putText(combined_display, 'Adaptive', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
        cv2.putText(combined_display, 'OTSU', (330, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
        cv2.putText(combined_display, 'Dynamic', (650, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
        
        # Show thresholding results
        cv2.imshow('Thresholding Methods', combined_display)
        cv2.imshow('Combined Result', processed_img['combined'])
        cv2.imshow('ROI Bottom', processed_img['roi_bottom'])
        
        cv2.waitKey(1)  # Non-blocking

def main():
    global logger
    logger = Logger()
    
    logger.info("="*60)
    logger.info("LINE FOLLOWING ROBOT - Enhanced Version with Display")
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
    
    # Parameter yang ditingkatkan untuk mengatasi robot terdiam
    BASE_PWM = 70           # Dinaikkan dari 60
    SCALING_FACTOR = 0.4    # Dinaikkan dari 0.3
    MIN_TURN_PWM = 45       # Dinaikkan dari 35
    LOG_INTERVAL = 10       # Log setiap N frame
    
    logger.info(f"Enhanced Configuration:")
    logger.info(f"  Base PWM: {BASE_PWM}")
    logger.info(f"  Scaling Factor: {SCALING_FACTOR}")
    logger.info(f"  Minimum Turn PWM: {MIN_TURN_PWM}")
    logger.info(f"  Log Interval: {LOG_INTERVAL} frames")
    logger.info("  Thresholding display: ENABLED")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            processed_img = advanced_roi_thresholding(frame)
            
            # Display thresholding results
            display_thresholding_results(processed_img, frame_count)
            
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
                
                # Detailed logging
                if frame_count % LOG_INTERVAL == 0:
                    turn_direction = "STRAIGHT"
                    if abs(kontrol) > 10:  # Reduced threshold
                        turn_direction = "RIGHT" if kontrol > 0 else "LEFT"
                    
                    logger.debug(f"Frame {frame_count:4d} | "
                               f"Pos: ({cx:3d},{cy:3d}) | "
                               f"Err: {error:4d} | "
                               f"ΔErr: {delta_error:4d} | "
                               f"FLC: {kontrol:6.1f} | "
                               f"PWM: L={pwm_kiri:2d} R={pwm_kanan:2d} | "
                               f"Turn: {turn_direction:8s} | "
                               f"Conf: {confidence:.2f} | "
                               f"Src: {source} | "
                               f"Bright: {processed_img['brightness']:.0f}")
                
            else:
                no_line_count += 1
                time_since_line = current_time - last_line_time
                
                # Enhanced strategy saat garis hilang
                if time_since_line < 0.3:  # Reduced time threshold
                    if abs(prev_error) > 30:  # Lower threshold for turning
                        # Continue turning in last direction with higher PWM
                        emergency_kontrol = np.sign(prev_error) * 80  # Increased from 60
                        pwm_kiri, pwm_kanan = calculate_motor_pwm_enhanced(
                            emergency_kontrol, BASE_PWM * 0.9, SCALING_FACTOR  # Increased multiplier
                        )
                        send_motor_commands(ser, pwm_kiri, pwm_kanan)
                        
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn(f"Line lost - aggressive turn continuation (Error: {prev_error})")
                    else:
                        # Go straight with higher PWM
                        straight_pwm = max(BASE_PWM // 1.5, 40)  # Minimum 40 PWM
                        send_motor_commands(ser, straight_pwm, straight_pwm)
                        if frame_count % LOG_INTERVAL == 0:
                            logger.warn("Line lost - going straight with higher PWM")
                else:
                    # Long time without line - stop
                    send_motor_commands(ser, 0, 0)
                    if frame_count % (LOG_INTERVAL * 2) == 0:
                        logger.warn(f"Line lost for {time_since_line:.1f}s - stopping")
            
            # System health monitoring
            if frame_count % 100 == 0:
                detection_rate = ((100-min(no_line_count, 100))/100)*100
                logger.info(f"System Status - Frame: {frame_count}, "
                          f"Line Detection Rate: {detection_rate:.0f}%")
            
            time.sleep(0.04)  # 25 FPS (slightly faster)
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        send_motor_commands(ser, 0, 0)  # Stop motors
        cv2.destroyAllWindows()  # Close display windows
        if ser:
            ser.close()
        if picam2:
            picam2.stop()
        logger.info("Cleanup completed - Robot stopped")

if __name__ == '__main__':
    main()
