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

class AdaptiveLineDetector:
    def __init__(self, frame_width=320, frame_height=240):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        
        # Multiple detection zones untuk adaptabilitas
        self.zones = {
            'near': (int(frame_height * 0.75), frame_height),      # 75-100% dari bawah
            'mid': (int(frame_height * 0.5), int(frame_height * 0.75)),  # 50-75%
            'far': (int(frame_height * 0.25), int(frame_height * 0.5))   # 25-50%
        }
        
        # Tracking state untuk konsistensi
        self.last_valid_position = self.center_x
        self.confidence_threshold = 500  # Minimum area untuk detection yang valid
        
    def detect_line_multi_zone(self, binary_image):
        """Detect line menggunakan multiple zones dengan prioritas"""
        detections = {}
        
        for zone_name, (y1, y2) in self.zones.items():
            roi = binary_image[y1:y2, :]
            line_found, cx, confidence = self._detect_in_zone(roi, y1)
            
            detections[zone_name] = {
                'found': line_found,
                'center_x': cx,
                'confidence': confidence,
                'weight': self._get_zone_weight(zone_name)
            }
        
        return self._compute_weighted_position(detections)
    
    def _detect_in_zone(self, roi, y_offset):
        """Detect line dalam satu zone"""
        # Morphological operations untuk cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
        
        # Cari contours untuk deteksi yang lebih robust
        contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pilih contour terbesar yang reasonably shaped
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter noise
                    # Check aspect ratio untuk filter objek yang terlalu bulat/kotak
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5:  # Reasonable line shape
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # Pilih contour dengan area terbesar
                best_contour, confidence = max(valid_contours, key=lambda x: x[1])
                
                # Hitung centroid
                M = cv2.moments(best_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    return True, cx, confidence
        
        # Fallback ke moment-based detection
        M = cv2.moments(roi_clean)
        if M['m00'] > self.confidence_threshold:
            cx = int(M['m10'] / M['m00'])
            return True, cx, M['m00']
        
        return False, 0, 0
    
    def _get_zone_weight(self, zone_name):
        """Weight untuk setiap zone - zone terdekat lebih prioritas"""
        weights = {'near': 0.6, 'mid': 0.3, 'far': 0.1}
        return weights.get(zone_name, 0.1)
    
    def _compute_weighted_position(self, detections):
        """Hitung posisi final dengan weighted average"""
        total_weight = 0
        weighted_sum = 0
        valid_detections = 0
        
        for zone, data in detections.items():
            if data['found'] and data['confidence'] > self.confidence_threshold:
                weight = data['weight'] * (data['confidence'] / 1000)  # Normalize confidence
                weighted_sum += data['center_x'] * weight
                total_weight += weight
                valid_detections += 1
        
        if total_weight > 0:
            final_x = int(weighted_sum / total_weight)
            self.last_valid_position = final_x
            return True, final_x, valid_detections
        else:
            # Gunakan last known position dengan confidence rendah
            return False, self.last_valid_position, 0

def setup_fuzzy_logic_adaptive():
    """Setup FLC yang lebih adaptive untuk berbagai kondisi"""
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Error membership functions
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -50])
    error['NM'] = fuzz.trimf(error.universe, [-80, -40, -10])
    error['NS'] = fuzz.trimf(error.universe, [-25, -10, -2])
    error['Z']  = fuzz.trimf(error.universe, [-8, 0, 8])
    error['PS'] = fuzz.trimf(error.universe, [2, 10, 25])
    error['PM'] = fuzz.trimf(error.universe, [10, 40, 80])
    error['PL'] = fuzz.trimf(error.universe, [50, 160, 160])

    # Delta error
    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 100, 100])

    # Confidence level
    confidence['LOW'] = fuzz.trimf(confidence.universe, [0, 0, 30])
    confidence['MED'] = fuzz.trimf(confidence.universe, [20, 50, 80])
    confidence['HIGH'] = fuzz.trimf(confidence.universe, [70, 100, 100])

    # Output actions
    output['FL'] = fuzz.trimf(output.universe, [-100, -100, -60])  # Fast Left
    output['L']  = fuzz.trimf(output.universe, [-80, -50, -20])    # Left
    output['SL'] = fuzz.trimf(output.universe, [-30, -15, -3])     # Slow Left
    output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])         # Zero
    output['SR'] = fuzz.trimf(output.universe, [3, 15, 30])        # Slow Right
    output['R']  = fuzz.trimf(output.universe, [20, 50, 80])       # Right
    output['FR'] = fuzz.trimf(output.universe, [60, 100, 100])     # Fast Right

    # Expanded rule base dengan confidence consideration
    rules = [
        # High confidence rules - aggressive correction
        ctrl.Rule(error['NL'] & confidence['HIGH'], output['FL']),
        ctrl.Rule(error['NM'] & confidence['HIGH'], output['L']),
        ctrl.Rule(error['NS'] & confidence['HIGH'], output['SL']),
        ctrl.Rule(error['Z'] & confidence['HIGH'], output['Z']),
        ctrl.Rule(error['PS'] & confidence['HIGH'], output['SR']),
        ctrl.Rule(error['PM'] & confidence['HIGH'], output['R']),
        ctrl.Rule(error['PL'] & confidence['HIGH'], output['FR']),
        
        # Medium confidence - moderate correction
        ctrl.Rule(error['NL'] & confidence['MED'], output['L']),
        ctrl.Rule(error['NM'] & confidence['MED'], output['SL']),
        ctrl.Rule(error['NS'] & confidence['MED'], output['SL']),
        ctrl.Rule(error['Z'] & confidence['MED'], output['Z']),
        ctrl.Rule(error['PS'] & confidence['MED'], output['SR']),
        ctrl.Rule(error['PM'] & confidence['MED'], output['SR']),
        ctrl.Rule(error['PL'] & confidence['MED'], output['R']),
        
        # Low confidence - conservative correction
        ctrl.Rule(error['NL'] & confidence['LOW'], output['SL']),
        ctrl.Rule(error['NM'] & confidence['LOW'], output['SL']),
        ctrl.Rule(error['NS'] & confidence['LOW'], output['Z']),
        ctrl.Rule(error['Z'] & confidence['LOW'], output['Z']),
        ctrl.Rule(error['PS'] & confidence['LOW'], output['Z']),
        ctrl.Rule(error['PM'] & confidence['LOW'], output['SR']),
        ctrl.Rule(error['PL'] & confidence['LOW'], output['SR']),
        
        # Delta-based rules untuk stabilitas
        ctrl.Rule(delta['NL'] & error['Z'], output['SR']),  # Counteract oscillation
        ctrl.Rule(delta['PL'] & error['Z'], output['SL']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    """Setup camera dengan konfigurasi yang lebih optimal"""
    picam2 = Picamera2()
    # Gunakan resolusi yang sedikit lebih tinggi untuk akurasi lebih baik
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Beri waktu untuk camera warm-up
    return picam2

def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image_enhanced(frame):
    """Enhanced image processing dengan multiple thresholding"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur untuk noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kombinasi OTSU dan adaptive threshold
    _, otsu_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Kombinasi kedua metode untuk hasil yang lebih robust
    combined_binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    
    return gray, combined_binary

def compute_fuzzy_control_adaptive(fuzzy_ctrl, error_val, delta_error, confidence_level):
    """Enhanced fuzzy control dengan confidence consideration"""
    try:
        # Clamp input values
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        confidence_level = max(0, min(100, confidence_level))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.input['confidence'] = confidence_level
        fuzzy_ctrl.compute()
        
        kontrol = fuzzy_ctrl.output['output']
        return np.clip(kontrol, -100, 100)
        
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm_adaptive(kontrol, base_pwm=50, confidence_level=100):
    """Adaptive PWM calculation berdasarkan confidence level"""
    # Scaling factor berdasarkan confidence
    if confidence_level > 80:
        scaling_factor = 0.25  # High confidence - aggressive
    elif confidence_level > 50:
        scaling_factor = 0.2   # Medium confidence - moderate
    else:
        scaling_factor = 0.15  # Low confidence - conservative
    
    kontrol_scaled = kontrol * scaling_factor
    
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Adaptive PWM limits berdasarkan confidence
    if confidence_level > 80:
        min_pwm, max_pwm = 30, 75
    else:
        min_pwm, max_pwm = 35, 65  # Lebih konservatif untuk confidence rendah
    
    pwm_kiri = max(min_pwm, min(max_pwm, pwm_kiri))
    pwm_kanan = max(min_pwm, min(max_pwm, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def draw_visualization(frame, binary, cx, error, kontrol, pwm_left, pwm_right, confidence, zones):
    """Draw visualization overlay pada main camera frame"""
    height, width = frame.shape[:2]
    
    # Draw detection zones dengan warna berbeda
    zone_colors = {'near': (0, 255, 0), 'mid': (255, 255, 0), 'far': (255, 0, 255)}
    for zone_name, (y1, y2) in zones.items():
        color = zone_colors.get(zone_name, (128, 128, 128))
        cv2.rectangle(frame, (0, y1), (width, y2), color, 1)
        cv2.putText(frame, zone_name.upper(), (5, y1 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw center line (setpoint)
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    
    # Draw detected line position
    if cx > 0:
        cv2.circle(frame, (cx, int(height * 0.8)), 8, (0, 255, 255), -1)  # Yellow circle
        cv2.line(frame, (cx, 0), (cx, height), (0, 255, 255), 1)  # Yellow line
    
    # Draw error line
    cv2.line(frame, (center_x, int(height * 0.8)), (cx, int(height * 0.8)), (255, 0, 0), 3)
    
    # Text overlay dengan background
    overlay_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status text
    status_texts = [
        f"Err: {error:3d} | Ctrl: {kontrol:5.1f}",
        f"PWM: L={pwm_left:2d} R={pwm_right:2d}",
        f"Conf: {confidence:3.0f}% | Pos: {cx:3d}",
        f"Center: {center_x:3d} | FPS: ~20"
    ]
    
    for i, text in enumerate(status_texts):
        y_pos = 20 + i * 20
        if i == 0:  # Error text - larger and colored
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def draw_no_line_visualization(frame, no_line_counter, zones):
    """Draw visualization ketika garis tidak terdeteksi"""
    height, width = frame.shape[:2]
    
    # Draw zones dengan warna red (warning)
    for zone_name, (y1, y2) in zones.items():
        cv2.rectangle(frame, (0, y1), (width, y2), (0, 0, 255), 2)
    
    # Warning overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 128), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Warning text
    cv2.putText(frame, "LINE NOT DETECTED!", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Counter: {no_line_counter} frames", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw center reference
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    
    return frame

def create_roi_binary_display(binary, zones, detected_cx):
    """Create display untuk ROI binary seperti di gambar"""
    height, width = binary.shape
    
    # Create colored version dari binary image
    roi_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Highlight different zones dengan warna
    zone_colors = {'near': (0, 255, 0), 'mid': (255, 255, 0), 'far': (255, 0, 255)}
    for zone_name, (y1, y2) in zones.items():
        color = zone_colors.get(zone_name, (128, 128, 128))
        # Draw zone boundaries
        cv2.line(roi_colored, (0, y1), (width, y1), color, 2)
        cv2.line(roi_colored, (0, y2), (width, y2), color, 2)
        
        # Add zone labels
        cv2.putText(roi_colored, zone_name, (5, y1 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw center line
    center_x = width // 2
    cv2.line(roi_colored, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    
    # Draw detected position jika ada
    if detected_cx > 0:
        cv2.line(roi_colored, (detected_cx, 0), (detected_cx, height), (0, 255, 255), 2)
        # Add position marker
        cv2.circle(roi_colored, (detected_cx, height//2), 8, (0, 255, 255), -1)
    
    # Resize untuk display yang lebih baik
    display_height = 300
    display_width = int(width * display_height / height)
    roi_display = cv2.resize(roi_colored, (display_width, display_height))
    
    # Add info overlay
    info_overlay = roi_display.copy()
    cv2.rectangle(info_overlay, (0, display_height-60), (display_width, display_height), (0, 0, 0), -1)
    cv2.addWeighted(info_overlay, 0.7, roi_display, 0.3, 0, roi_display)
    
    # Add position info
    cv2.putText(roi_display, f"Center: {center_x}", (10, display_height-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(roi_display, f"Detected: {detected_cx if detected_cx > 0 else 'N/A'}", 
               (10, display_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return roi_display

def main():
    print("[SYSTEM] Starting Adaptive Line Following Robot")
    
    # Setup komponen
    fuzzy_ctrl = setup_fuzzy_logic_adaptive()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)
    line_detector = AdaptiveLineDetector()
    
    # Variabel kontrol
    prev_error = 0
    frame_count = 0
    no_line_counter = 0
    
    # Setup untuk visualisasi
    cv2.namedWindow('Line Following Robot - Improved', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('ROI Binary', cv2.WINDOW_AUTOSIZE)
    
    print("[CONFIG] Adaptive multi-zone line detection enabled")
    print("[CONFIG] Confidence-based control scaling enabled")
    print("[DISPLAY] OpenCV windows initialized - Press 'q' to quit")
    
    try:
        while True:
            frame_count += 1
            
            # Capture dan process image
            frame = picam2.capture_array()
            gray, binary = process_image_enhanced(frame)
            
            # Deteksi garis menggunakan adaptive detector
            line_detected, cx, detection_count = line_detector.detect_line_multi_zone(binary)
            
            # Prepare visualization frame
            display_frame = frame.copy()
            
            if line_detected:
                no_line_counter = 0
                
                # Hitung error dan delta error
                error = cx - 160  # Setpoint di tengah
                filtered_error = error_filter.filter_error(error)
                delta_error = filtered_error - prev_error
                prev_error = filtered_error
                
                # Calculate confidence berdasarkan detection quality
                confidence_level = min(100, detection_count * 30 + 40)  # 40-100 range
                
                # Compute FLC output dengan confidence
                kontrol = compute_fuzzy_control_adaptive(fuzzy_ctrl, filtered_error, delta_error, confidence_level)
                
                # Hitung PWM dengan adaptive scaling
                pwm_kiri, pwm_kanan = calculate_motor_pwm_adaptive(kontrol, base_pwm=50, confidence_level=confidence_level)
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Visualisasi pada frame
                display_frame = draw_visualization(display_frame, binary, cx, filtered_error, kontrol, 
                                                pwm_kiri, pwm_kanan, confidence_level, line_detector.zones)
                
                # Debug info
                if frame_count % 15 == 0:
                    print(f"[DEBUG] Raw Error: {error:3d}, Filtered: {filtered_error:3d}, "
                          f"Delta: {delta_error:3d}, Conf: {confidence_level:3.0f}%, "
                          f"FLC: {kontrol:5.1f}, PWM: L={pwm_kiri}, R={pwm_kanan}")
                
            else:
                no_line_counter += 1
                
                # Strategy saat tidak ada garis
                if no_line_counter < 10:
                    # Continue dengan last command sebentar
                    pass
                elif no_line_counter < 30:
                    # Slow search pattern
                    send_motor_commands(ser, 40, 45)  # Slight right turn
                else:
                    # Stop dan reset
                    send_motor_commands(ser, 0, 0)
                
                # Visualisasi untuk no line detected
                display_frame = draw_no_line_visualization(display_frame, no_line_counter, line_detector.zones)
                
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Line not detected for {no_line_counter} frames")
            
            # Display frames
            cv2.imshow('Line Following Robot - Improved', display_frame)
            
            # Create ROI binary display
            roi_display = create_roi_binary_display(binary, line_detector.zones, cx if line_detected else -1)
            cv2.imshow('ROI Binary', roi_display)
            
            # Check for quit key
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
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Cleanup completed")

if __name__ == '__main__':
    main()
