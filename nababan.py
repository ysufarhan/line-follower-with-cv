from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    "SERIAL_PORT": '/dev/ttyS0', 
    "BAUD_RATE": 115200,
    "BASE_PWM": 45,
    "SCALING_FACTOR": 0.08,
    "MIN_PWM_OUTPUT": 25,
    "MAX_PWM_OUTPUT": 65,
    "FLC_DEAD_ZONE_ERROR": 15,
    "INITIAL_PUSH_PWM": 50,
    "INITIAL_PUSH_DURATION": 1.0,
    "ERROR_FILTER_WINDOW_SIZE": 3,
    "ERROR_FILTER_ALPHA": 0.7,
    "LINE_RECOVERY_SPEED": 40,
    "CAMERA_RESOLUTION": (640, 480),
    "ROI_START_FACTOR": 0.3, # % of height
    "ROI_END_FACTOR": 0.95,  # % of height
    "ROI_MARGIN_FACTOR": 0.05, # % of width (both sides)
    "CONTOUR_AREA_THRESHOLD": 300, # Min area for line contour
    "LOOP_DELAY": 0.02 # in seconds, for main loop framerate
}

# --- ERROR FILTER CLASS ---
class ErrorFilter:
    def __init__(self, window_size=CONFIG["ERROR_FILTER_WINDOW_SIZE"], alpha=CONFIG["ERROR_FILTER_ALPHA"]):
        self.window_size = window_size
        self.error_history = []
        self.alpha = alpha
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size: self.error_history.pop(0)
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors)//2]
        if len(self.error_history) > 1:
            prev_median_history = self.error_history[:-1]
            prev_median = sorted(prev_median_history)[len(prev_median_history)//2] if len(prev_median_history) > 0 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_median
        else: smoothed_error = median_error
        return int(smoothed_error)

# --- LINE RECOVERY CLASS ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.recovery_speed = CONFIG["LINE_RECOVERY_SPEED"]
    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        if self.lost_count < 10:
            send_motor_commands(ser_instance, 0, 0)
            return "STOP"
        elif self.lost_count < 30:
            send_motor_commands(ser_instance, -30, -30)
            return "REVERSE"
        else:
            if self.last_valid_error > 0:
                send_motor_commands(ser_instance, self.recovery_speed, -self.recovery_speed)
                return "SEARCH_RIGHT"
            else:
                send_motor_commands(ser_instance, -self.recovery_speed, self.recovery_speed)
                return "SEARCH_LEFT"
    def line_found(self, current_error):
        self.lost_count = 0
        self.last_valid_error = current_error

# --- FUZZY LOGIC CONTROL SETUP ---
def setup_fuzzy_logic():
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    error['NL'] = fuzz.trimf(error.universe, [-350, -200, -60])
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])
    error['PL'] = fuzz.trimf(error.universe, [60, 200, 350])

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-5, 0, 5])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 150])

    output['L']  = fuzz.trimf(output.universe, [-100, -70, -40])
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -8])
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 50])
    output['R']  = fuzz.trimf(output.universe, [40, 70, 100])

    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), ctrl.Rule(error['NL'] & delta['NS'], output['L']), ctrl.Rule(error['NL'] & delta['Z'], output['LS']), ctrl.Rule(error['NL'] & delta['PS'], output['Z']), ctrl.Rule(error['NL'] & delta['PL'], output['Z']),
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']), ctrl.Rule(error['NS'] & delta['NS'], output['LS']), ctrl.Rule(error['NS'] & delta['Z'], output['Z']), ctrl.Rule(error['NS'] & delta['PS'], output['Z']), ctrl.Rule(error['NS'] & delta['PL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NL'], output['Z']), ctrl.Rule(error['Z'] & delta['NS'], output['Z']), ctrl.Rule(error['Z'] & delta['Z'], output['Z']), ctrl.Rule(error['Z'] & delta['PS'], output['Z']), ctrl.Rule(error['Z'] & delta['PL'], output['Z']),
        ctrl.Rule(error['PS'] & delta['NL'], output['Z']), ctrl.Rule(error['PS'] & delta['NS'], output['Z']), ctrl.Rule(error['PS'] & delta['Z'], output['Z']), ctrl.Rule(error['PS'] & delta['PS'], output['RS']), ctrl.Rule(error['PS'] & delta['PL'], output['RS']),
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']), ctrl.Rule(error['PL'] & delta['NS'], output['Z']), ctrl.Rule(error['PL'] & delta['Z'], output['RS']), ctrl.Rule(error['PL'] & delta['PS'], output['R']), ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]
    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- CAMERA SETUP ---
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": CONFIG["CAMERA_RESOLUTION"]}) 
    picam2.configure(config)
    picam2.start()
    return picam2

# --- SERIAL COMMUNICATION SETUP ---
def setup_serial():
    try:
        ser = serial.Serial(CONFIG["SERIAL_PORT"], CONFIG["BAUD_RATE"], timeout=1) 
        print(f"[UART] Port serial {CONFIG['SERIAL_PORT']} opened.")
        return ser
    except serial.SerialException as e:
        print(f"[UART ERROR] Failed to open {CONFIG['SERIAL_PORT']}: {e}")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] Failed to open serial port: {e}")
        return None

# --- IMAGE PROCESSING ---
def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    roi_y_start = int(h * CONFIG["ROI_START_FACTOR"])
    roi_y_end = int(h * CONFIG["ROI_END_FACTOR"])
    roi_x_start = int(w * CONFIG["ROI_MARGIN_FACTOR"])
    roi_x_end = int(w * (1 - CONFIG["ROI_MARGIN_FACTOR"]))
    
    roi_y_start = max(0, min(roi_y_start, h - 1))
    roi_y_end = max(0, min(roi_y_end, h))
    roi_x_start = max(0, min(roi_x_start, w - 1))
    roi_x_end = max(0, min(roi_x_end, w))

    roi = binary[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    return gray, binary, roi, roi_y_start, roi_x_start, roi_y_end, roi_x_end

# --- CALCULATE LINE POSITION ---
def calculate_line_position(roi_image, roi_start_y, roi_start_x, frame_width):
    kernel = np.ones((5,5), np.uint8) 
    roi_clean = cv2.morphologyEx(roi_image, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > CONFIG["CONTOUR_AREA_THRESHOLD"]:
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + roi_start_x
                cy = int(M['m01'] / M['m00']) + roi_start_y
                return True, cx, cy, largest_contour
    return False, 0, 0, None

# --- COMPUTE FUZZY CONTROL ---
def compute_fuzzy_control(fuzzy_ctrl_system, error_value, delta_error_value):
    try:
        fuzzy_ctrl_system.input['error'] = np.clip(error_value, -350, 350)
        fuzzy_ctrl_system.input['delta'] = np.clip(delta_error_value, -150, 150)
        fuzzy_ctrl_system.compute()
        return np.clip(fuzzy_ctrl_system.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}. Input Error: {error_value}, Delta Error: {delta_error_value}")
        return 0.0

# --- CALCULATE MOTOR PWM ---
def calculate_motor_pwm(control_output):
    if abs(control_output) < CONFIG["FLC_DEAD_ZONE_ERROR"]:
        control_scaled = 0
    else:
        control_scaled = control_output * CONFIG["SCALING_FACTOR"]

    pwm_left = CONFIG["BASE_PWM"] + control_scaled
    pwm_right = CONFIG["BASE_PWM"] - control_scaled

    pwm_left = max(CONFIG["MIN_PWM_OUTPUT"], min(CONFIG["MAX_PWM_OUTPUT"], pwm_left))
    pwm_right = max(CONFIG["MIN_PWM_OUTPUT"], min(CONFIG["MAX_PWM_OUTPUT"], pwm_right))

    return int(pwm_left), int(pwm_right)

# --- SEND MOTOR COMMANDS ---
def send_motor_commands(ser_instance, pwm_left, pwm_right):
    if ser_instance and ser_instance.is_open:
        try:
            cmd = f"{pwm_left},{pwm_right}\n"
            ser_instance.write(cmd.encode())
            ser_instance.flush()
        except serial.SerialException as e:
            print(f"[SERIAL SEND ERROR] Failed to send: {e}")
        except Exception as e:
            print(f"[SERIAL SEND ERROR] Unknown error: {e}")

# --- MAIN PROGRAM ---
def main():
    fuzzy_controller = setup_fuzzy_logic()
    camera = setup_camera()
    serial_port = setup_serial()
    error_filter = ErrorFilter()
    line_recovery_handler = LineRecovery()

    prev_filtered_error = 0
    frame_counter = 0
    
    print("[INFO] Starting Line Follower...")
    print("[INFO] Warming up camera...")
    time.sleep(2)

    # START-UP KICK-START
    if serial_port and serial_port.is_open:
        print(f"[STARTUP] Initial push ({CONFIG['INITIAL_PUSH_PWM']} PWM for {CONFIG['INITIAL_PUSH_DURATION']}s)...")
        start_time = time.time()
        while time.time() - start_time < CONFIG["INITIAL_PUSH_DURATION"]:
            send_motor_commands(serial_port, CONFIG["INITIAL_PUSH_PWM"], CONFIG["INITIAL_PUSH_PWM"])
            time.sleep(0.05)
        print("[STARTUP] Initial push finished.")
    else:
        print("[STARTUP WARNING] Serial port not active, cannot perform initial push.")

    try:
        while True:
            frame = camera.capture_array()
            h, w = frame.shape[:2]
            center_x = w // 2

            gray_frame, binary_frame, roi_frame, roi_y_start, roi_x_start, roi_y_end, roi_x_end = process_image(frame)
            line_found, line_cx, line_cy, line_contour = calculate_line_position(roi_frame, roi_y_start, roi_x_start, w)

            if line_found:
                line_recovery_handler.line_found(line_cx - center_x)
                
                current_error = line_cx - center_x
                filtered_error = error_filter.filter_error(current_error)
                
                delta_error = filtered_error - prev_filtered_error
                prev_filtered_error = filtered_error

                control_output = compute_fuzzy_control(fuzzy_controller, filtered_error, delta_error)
                pwm_left, pwm_right = calculate_motor_pwm(control_output)
                
                send_motor_commands(serial_port, pwm_left, pwm_right)

                if frame_counter % 10 == 0:
                    status_text = "STRAIGHT" if abs(filtered_error) < CONFIG["FLC_DEAD_ZONE_ERROR"] else "TURNING"
                    print(f"[{status_text}] Err:{filtered_error:3d}, ΔErr:{delta_error:3d}, FLC:{control_output:5.1f}, PWM: L{pwm_left} R{pwm_right}")
            else:
                recovery_action = line_recovery_handler.handle_line_lost(serial_port)
                prev_filtered_error = 0
                if frame_counter % 15 == 0:
                    print(f"[LOST] Line lost, action: {recovery_action}, lost_count: {line_recovery_handler.lost_count}")
            
            # --- VISUALIZATION ---
            display_frame = frame.copy()
            cv2.line(display_frame, (center_x, 0), (center_x, h), (0, 255, 0), 2)
            roi_color = (255, 255, 0)
            cv2.rectangle(display_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), roi_color, 2)
            
            if line_found:
                cv2.circle(display_frame, (line_cx, line_cy), 8, (0, 0, 255), -1)
                if line_contour is not None:
                    adj_contour = line_contour.copy()
                    adj_contour[:, :, 0] += roi_x_start
                    adj_contour[:, :, 1] += roi_y_start
                    cv2.drawContours(display_frame, [adj_contour], -1, (255, 0, 255), 2)
                
                raw_error_display = line_cx - center_x
                cv2.putText(display_frame, f"Error: {raw_error_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Filtered Err: {filtered_error}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Status: TRACKING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Status: LINE LOST ({line_recovery_handler.lost_count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Line Follower Live", cv2.resize(display_frame, (640, 480)))
            cv2.imshow("Binary Processed ROI", cv2.resize(binary_frame, (320, 240)))
            cv2.imshow("Cleaned ROI for Contour", cv2.resize(roi_frame, (320, 100)))

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_counter += 1
            time.sleep(CONFIG["LOOP_DELAY"])
            
    except KeyboardInterrupt: print("\n[INFO] Program stopped by user.")
    except Exception as e: print(f"[CRITICAL ERROR] Unexpected error: {e}")
    finally:
        print("[INFO] Stopping motors and releasing resources.")
        send_motor_commands(serial_port, 0, 0) 
        if serial_port and serial_port.is_open: serial_port.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program finished.")

if __name__ == "__main__":
    main()
