from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import threading
from queue import Queue

class AdaptiveErrorFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.error_history = []
        self.velocity_history = []
        self.alpha = 0.8  # Increased for more responsiveness
        self.adaptive_threshold = 15
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Calculate velocity (rate of change)
        if len(self.error_history) >= 2:
            velocity = self.error_history[-1] - self.error_history[-2]
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 3:
                self.velocity_history.pop(0)
        
        # Adaptive filtering based on error magnitude
        if abs(error) > self.adaptive_threshold:
            # High error: Use more responsive filtering
            weight = 0.9
        else:
            # Low error: Use more smoothing
            weight = 0.6
            
        # Weighted moving average
        weights = np.linspace(weight, 1.0, len(self.error_history))
        weighted_avg = np.average(self.error_history, weights=weights)
        
        return int(weighted_avg)

class PerformanceOptimizer:
    def __init__(self):
        self.frame_skip_counter = 0
        self.processing_times = []
        self.max_processing_time = 0.03  # 30ms max processing time
        
    def should_skip_frame(self, processing_time):
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
            
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        # Skip frames if processing is too slow
        if avg_time > self.max_processing_time:
            self.frame_skip_counter += 1
            if self.frame_skip_counter % 2 == 0:
                return True
        else:
            self.frame_skip_counter = 0
            
        return False

def setup_optimized_fuzzy_logic():
    # Refined universe ranges for better precision
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Optimized membership functions for smoother response
    error['NL'] = fuzz.trapmf(error.universe, [-160, -120, -80, -40])
    error['NS'] = fuzz.trapmf(error.universe, [-60, -30, -15, -5])
    error['Z']  = fuzz.trapmf(error.universe, [-8, -3, 3, 8])      # Tighter zero zone
    error['PS'] = fuzz.trapmf(error.universe, [5, 15, 30, 60])
    error['PL'] = fuzz.trapmf(error.universe, [40, 80, 120, 160])

    delta['NL'] = fuzz.trapmf(delta.universe, [-100, -60, -30, -10])
    delta['NS'] = fuzz.trapmf(delta.universe, [-20, -10, -5, -2])
    delta['Z']  = fuzz.trapmf(delta.universe, [-5, -1, 1, 5])
    delta['PS'] = fuzz.trapmf(delta.universe, [2, 5, 10, 20])
    delta['PL'] = fuzz.trapmf(delta.universe, [10, 30, 60, 100])

    output['L']  = fuzz.trapmf(output.universe, [-100, -80, -60, -30])
    output['LS'] = fuzz.trapmf(output.universe, [-50, -25, -15, -5])
    output['Z']  = fuzz.trapmf(output.universe, [-8, -2, 2, 8])
    output['RS'] = fuzz.trapmf(output.universe, [5, 15, 25, 50])
    output['R']  = fuzz.trapmf(output.universe, [30, 60, 80, 100])

    # Enhanced rule set for smoother control
    rules = [
        # Aggressive correction for large errors
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

        # Moderate correction for small errors
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),

        # Minimal correction when centered
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        # Symmetric rules for positive errors
        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_optimized_camera():
    picam2 = Picamera2()
    # Optimized camera settings for better performance
    config = picam2.create_still_configuration(
        main={"size": (320, 240)},
        controls={
            "FrameRate": 30,
            "ExposureTime": 8000,  # Fixed exposure for consistent lighting
            "AnalogueGain": 2.0,
            "AeEnable": False,     # Disable auto exposure for consistent performance
            "AwbEnable": False,    # Disable auto white balance
        }
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Camera warm-up
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.01)  # Reduced timeout
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def optimized_image_processing(frame):
    """Optimized image processing pipeline"""
    # Convert to grayscale more efficiently
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optimized ROI processing
    roi_y_start = 160
    roi = gray[roi_y_start:240, :]
    
    # More efficient thresholding
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optimized morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return gray, binary_roi, roi_y_start

def enhanced_line_detection(binary_roi, roi_y_start):
    """Enhanced line detection with multiple methods"""
    # Method 1: Contour-based detection (more robust)
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the line)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 50:  # Minimum area threshold
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00']) + roi_y_start
                
                # Calculate line angle for better prediction
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    angle = ellipse[2]
                else:
                    angle = 0
                    
                return True, cx, cy, angle, area
    
    # Method 2: Fallback to moments (original method)
    M = cv2.moments(binary_roi)
    if M['m00'] > 80:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + roi_y_start
        return True, cx, cy, 0, M['m00']
    
    return False, 0, 0, 0, 0

def adaptive_motor_control(kontrol, base_pwm=55, error_magnitude=0):
    """Adaptive motor control based on error magnitude"""
    # Dynamic scaling based on error
    if abs(error_magnitude) > 50:
        scaling_factor = 0.4  # More aggressive for large errors
        base_pwm = 45  # Slower base speed for sharp turns
    elif abs(error_magnitude) > 20:
        scaling_factor = 0.3
        base_pwm = 50
    else:
        scaling_factor = 0.2  # Gentle correction for small errors
        base_pwm = 55  # Higher speed when following straight
    
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Adaptive PWM limits
    min_pwm = 25 if abs(error_magnitude) > 50 else 35
    max_pwm = 80 if abs(error_magnitude) > 50 else 70
    
    pwm_kiri = max(min_pwm, min(max_pwm, pwm_kiri))
    pwm_kanan = max(min_pwm, min(max_pwm, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands_optimized(ser, pwm_kiri, pwm_kanan):
    """Optimized serial communication"""
    if ser and ser.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode('ascii'))
            # Don't flush every time to improve performance
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def main():
    print("[INFO] Memulai sistem line following yang dioptimasi...")
    
    # Initialize components
    fuzzy_ctrl = setup_optimized_fuzzy_logic()
    picam2 = setup_optimized_camera()
    ser = setup_serial()
    error_filter = AdaptiveErrorFilter(window_size=4)
    optimizer = PerformanceOptimizer()

    # Control variables
    prev_error = 0
    frame_count = 0
    lost_line_counter = 0
    max_lost_frames = 10
    
    # Performance monitoring
    fps_counter = 0
    fps_start_time = time.time()
    
    # PID-like enhancement for fuzzy control
    integral_error = 0
    max_integral = 1000

    try:
        print("[INFO] Sistem siap! Memulai line following...")
        
        while True:
            start_time = time.time()
            
            # Capture frame
            frame = picam2.capture_array()
            
            # Process image
            gray, binary_roi, roi_y_start = optimized_image_processing(frame)
            
            # Detect line
            line_detected, cx, cy, angle, area = enhanced_line_detection(binary_roi, roi_y_start)
            
            if line_detected:
                lost_line_counter = 0
                
                # Calculate error
                raw_error = cx - 160
                filtered_error = error_filter.filter_error(raw_error)
                delta_error = filtered_error - prev_error
                
                # Integral term for steady-state error reduction
                integral_error += filtered_error
                integral_error = max(-max_integral, min(max_integral, integral_error))
                
                # Enhanced error with integral component
                enhanced_error = filtered_error + (integral_error * 0.01)
                
                prev_error = filtered_error

                # Fuzzy logic control
                try:
                    fuzzy_ctrl.input['error'] = np.clip(enhanced_error, -160, 160)
                    fuzzy_ctrl.input['delta'] = np.clip(delta_error, -100, 100)
                    fuzzy_ctrl.compute()
                    kontrol = fuzzy_ctrl.output['output']
                except:
                    kontrol = 0.0

                # Adaptive motor control
                pwm_kiri, pwm_kanan = adaptive_motor_control(kontrol, error_magnitude=abs(filtered_error))
                send_motor_commands_optimized(ser, pwm_kiri, pwm_kanan)

                # Debug output (reduced frequency)
                if frame_count % 20 == 0:
                    print(f"[CTRL] E:{filtered_error:3d} D:{delta_error:3d} FLC:{kontrol:5.1f} PWM:L{pwm_kiri}R{pwm_kanan} Area:{area:4.0f}")
                    
            else:
                lost_line_counter += 1
                integral_error *= 0.9  # Decay integral when line is lost
                
                if lost_line_counter < max_lost_frames:
                    # Continue with last known control for a few frames
                    pwm_kiri, pwm_kanan = adaptive_motor_control(prev_error * 0.5)
                    send_motor_commands_optimized(ser, pwm_kiri, pwm_kanan)
                else:
                    # Stop if line is lost for too long
                    send_motor_commands_optimized(ser, 0, 0)
                    if frame_count % 30 == 0:
                        print("[WARN] Garis hilang - robot berhenti")

            # Performance monitoring
            processing_time = time.time() - start_time
            
            # Adaptive frame skipping
            if not optimizer.should_skip_frame(processing_time):
                # Display (optional, comment out for max performance)
                if frame_count % 3 == 0:  # Reduced display frequency
                    display_frame = frame.copy()
                    cv2.line(display_frame, (160, roi_y_start), (160, 240), (0, 255, 0), 2)
                    if line_detected:
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, f"E:{filtered_error}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow("Optimized Camera View", display_frame)
                    cv2.imshow("Binary ROI", binary_roi)

            # FPS calculation
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed
                print(f"[PERF] FPS: {fps:.1f}, Proc: {processing_time*1000:.1f}ms")
                fps_start_time = time.time()

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            
            # Adaptive sleep for thermal management
            if processing_time < 0.02:
                time.sleep(0.005)  # Brief sleep to prevent overheating
                
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    finally:
        # Cleanup
        send_motor_commands_optimized(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Sistem shutdown complete")

if __name__ == "__main__":
    main()
