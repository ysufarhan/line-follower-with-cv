from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ErrorFilter:
    def __init__(self, window_size=5):  # Increased window size for smoother filtering
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.6  # Adjusted smoothing parameter

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Weighted moving average (recent errors have more weight)
        weights = np.linspace(0.5, 1.5, len(self.error_history))
        weighted_errors = np.array(self.error_history) * weights
        avg_error = np.sum(weighted_errors) / np.sum(weights)
        
        # Exponential smoothing
        if len(self.error_history) > 1:
            prev_avg = np.mean(self.error_history[:-1])
            smoothed_error = self.alpha * avg_error + (1 - self.alpha) * prev_avg
        else:
            smoothed_error = avg_error
            
        return int(smoothed_error)

def setup_fuzzy_logic():
    # Adjusted universe ranges for better centering
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-120, 121, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # More balanced membership functions
    error['NL'] = fuzz.trimf(error.universe, [-160, -120, -60])
    error['NS'] = fuzz.trimf(error.universe, [-80, -40, -10])
    error['Z'] = fuzz.trimf(error.universe, [-20, 0, 20])  # Wider zero zone
    error['PS'] = fuzz.trimf(error.universe, [10, 40, 80])
    error['PL'] = fuzz.trimf(error.universe, [60, 120, 160])

    delta['NL'] = fuzz.trimf(delta.universe, [-120, -80, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -25, -5])
    delta['Z'] = fuzz.trimf(delta.universe, [-15, 0, 15])  # Wider zero zone
    delta['PS'] = fuzz.trimf(delta.universe, [5, 25, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 80, 120])

    output['L'] = fuzz.trimf(output.universe, [-100, -70, -40])
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -5])
    output['Z'] = fuzz.trimf(output.universe, [-10, 0, 10])  # Very tight zero zone
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 50])
    output['R'] = fuzz.trimf(output.universe, [40, 70, 100])

    # Revised rules for better centering
    rules = [
        # When error is zero, keep output zero regardless of delta
        ctrl.Rule(error['Z'], output['Z']),
        
        # Small errors with any delta should produce small corrections
        ctrl.Rule(error['NS'] & (delta['NL'] | delta['NS']), output['LS']),
        ctrl.Rule(error['NS'] & (delta['Z'] | delta['PS'] | delta['PL']), output['Z']),
        ctrl.Rule(error['PS'] & (delta['NL'] | delta['NS']), output['Z']),
        ctrl.Rule(error['PS'] & (delta['Z'] | delta['PS'] | delta['PL']), output['RS']),
        
        # Large errors need stronger corrections
        ctrl.Rule(error['NL'] & (delta['NL'] | delta['NS']), output['L']),
        ctrl.Rule(error['NL'] & (delta['Z'] | delta['PS'] | delta['PL']), output['LS']),
        ctrl.Rule(error['PL'] & (delta['NL'] | delta['NS']), output['RS']),
        ctrl.Rule(error['PL'] & (delta['Z'] | delta['PS'] | delta['PL']), output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (320, 240)},
        transform=cv2.ROTATE_180  # Adjust if camera is mounted upside down
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to settle
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        ser.reset_input_buffer()
        time.sleep(0.1)
        print("[UART] Serial port initialized")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Failed to open serial port: {e}")
        return None

def process_image(frame):
    # Convert to HSV and use saturation channel for better line detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(saturation, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # ROI - focus on bottom 1/3 of image
    roi = binary[160:240, :]
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary, roi

def calculate_line_position(roi):
    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get moments
        M = cv2.moments(largest_contour)
        if M['m00'] > 100:  # Minimum area threshold
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) + 160  # Adjust for ROI offset
            return True, cx, cy
    
    return False, 160, 240  # Default to center if no line found

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # Apply deadzone for small errors
        if abs(error_val) < 5 and abs(delta_error) < 5:
            return 0.0
            
        fuzzy_ctrl.input['error'] = np.clip(error_val, -160, 160)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -120, 120)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50):
    # Symmetrical PWM adjustment with deadzone
    if abs(kontrol) < 5:  # Deadzone for very small corrections
        return base_pwm, base_pwm
    
    # Non-linear mapping for smoother response
    adjustment = np.sign(kontrol) * (abs(kontrol)**0.8) * 0.3
    
    pwm_kiri = base_pwm + adjustment
    pwm_kanan = base_pwm - adjustment
    
    # Ensure motors get exactly the same PWM when going straight
    if abs(kontrol) < 10:
        pwm_kiri = pwm_kanan = base_pwm
    
    # Clamp PWM values
    pwm_kiri = max(40, min(70, pwm_kiri))  # Reduced max PWM for smoother operation
    pwm_kanan = max(40, min(70, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            # Send command with checksum
            cmd = f"{pwm_kiri:03d},{pwm_kanan:03d}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=5)

    prev_error = 0
    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            
            frame = picam2.capture_array()
            binary, roi = process_image(frame)

            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                error = cx - 160
                filtered_error = error_filter.filter_error(error)
                
                # Calculate delta error with time consideration
                delta_error = (filtered_error - prev_error) / (dt + 0.0001)
                prev_error = filtered_error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, filtered_error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    print(f"[CTRL] Error: {filtered_error:4d}, Delta: {delta_error:6.1f}, FLC: {kontrol:6.1f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                # If line is lost, stop gently
                send_motor_commands(ser, 0, 0)
                if frame_count % 20 == 0:
                    print("[WARN] Line lost")

            # Visualization
            debug_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.line(debug_frame, (160, 0), (160, 240), (0, 255, 0), 1)
            if line_detected:
                cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(debug_frame, f"E:{filtered_error}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Debug View", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.03)  # Consistent loop timing
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopping by user request")
    finally:
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete")

if __name__ == "__main__":
    main()
