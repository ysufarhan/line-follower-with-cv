from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

class ErrorFilter:
    """
    Filter untuk menstabilkan error yang berfluktuasi dengan adaptive filtering
    """
    def __init__(self, window_size=3, confidence_threshold=0.7):
        self.window_size = window_size
        self.error_history = []
        self.confidence_history = []
        self.confidence_threshold = confidence_threshold
        self.last_valid_error = 0
    
    def filter_error(self, error, line_confidence=1.0):
        """Filter error dengan confidence-based averaging"""
        self.error_history.append(error)
        self.confidence_history.append(line_confidence)
        
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            self.confidence_history.pop(0)
        
        # Weighted average berdasarkan confidence
        total_weight = sum(self.confidence_history)
        if total_weight > 0:
            weighted_error = sum(e * c for e, c in zip(self.error_history, self.confidence_history)) / total_weight
            
            # Jika confidence rendah, gunakan campuran dengan error sebelumnya
            if line_confidence < self.confidence_threshold:
                filtered_error = 0.3 * weighted_error + 0.7 * self.last_valid_error
            else:
                filtered_error = weighted_error
                self.last_valid_error = filtered_error
        else:
            filtered_error = self.last_valid_error
        
        return int(filtered_error)

class LineDetector:
    """
    Detector garis yang lebih robust dengan filtering noise
    """
    def __init__(self):
        self.prev_line_width = 0
        self.expected_line_width = 20  # Estimasi lebar garis dalam pixel
        self.width_tolerance = 0.5     # Toleransi lebar garis
        
    def advanced_thresholding(self, gray):
        """
        Thresholding yang lebih robust untuk mengabaikan noise
        """
        # 1. Adaptive threshold dengan parameter yang disesuaikan
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 8)
        
        # 2. Otsu threshold sebagai backup
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Kombinasi kedua metode dengan weighted blending
        # Adaptive lebih baik untuk detail, Otsu lebih konsisten
        combined = cv2.addWeighted(adaptive, 0.6, otsu, 0.4, 0)
        
        # 4. Morphological operations untuk membersihkan noise
        kernel_small = np.ones((3,3), np.uint8)
        kernel_large = np.ones((5,5), np.uint8)
        
        # Remove small noise
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        # Fill small gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
        
        return cleaned
    
    def filter_contours_by_line_properties(self, binary_roi):
        """
        Filter kontour berdasarkan properti garis yang diharapkan
        """
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0, 0, 0
        
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area minimum
            if area < 50:  # Terlalu kecil, kemungkinan noise
                continue
                
            # Hitung bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter berdasarkan aspect ratio (garis harus lebih panjang dari lebar)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 5:  # Terlalu horizontal, bukan garis vertikal
                continue
                
            # Filter berdasarkan lebar yang diharapkan
            if self.expected_line_width > 0:
                width_ratio = w / self.expected_line_width
                if width_ratio < (1 - self.width_tolerance) or width_ratio > (1 + self.width_tolerance):
                    continue
            
            # Hitung solidity (area/convex_hull_area) untuk filter bentuk aneh
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.7:  # Bentuk terlalu tidak beraturan
                continue
                
            valid_contours.append((contour, area, w))
        
        if not valid_contours:
            return None, 0, 0, 0
        
        # Pilih kontour terbesar yang memenuhi kriteria
        best_contour = max(valid_contours, key=lambda x: x[1])
        contour, area, width = best_contour
        
        # Update expected line width berdasarkan deteksi
        self.prev_line_width = width
        if self.expected_line_width == 0:
            self.expected_line_width = width
        else:
            # Smooth update
            self.expected_line_width = 0.8 * self.expected_line_width + 0.2 * width
        
        # Hitung centroid
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Hitung confidence berdasarkan kualitas deteksi
            confidence = min(1.0, area / 500.0)  # Normalize area to confidence
            
            return contour, cx, cy, confidence
        
        return None, 0, 0, 0

def setup_fuzzy_logic():
    """
    Konfigurasi sistem fuzzy logic untuk respons yang lebih cepat
    """
    # Buat variabel fuzzy
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')
    
    # MEMBERSHIP FUNCTIONS DIPERBAIKI - respons lebih cepat untuk belokan
    # Error - dead zone diperkecil untuk respons lebih cepat
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -60])   # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-90, -30, -5])     # Negative Small - lebih sensitif
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])       # Zero - diperkecil untuk respons cepat
    error['PS'] = fuzz.trimf(error.universe, [5, 30, 90])       # Positive Small - lebih sensitif
    error['PL'] = fuzz.trimf(error.universe, [60, 160, 160])     # Positive Large

    # Delta error - lebih responsif terhadap perubahan
    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -15, -2])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])       # Diperkecil
    delta['PS'] = fuzz.trimf(delta.universe, [2, 15, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 100, 100])

    # Output - respons lebih agresif untuk belokan cepat
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -40])  # Left
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -5])    # Left Small
    output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])        # Zero - sangat kecil
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 60])      # Right Small
    output['R']  = fuzz.trimf(output.universe, [40, 100, 100])   # Right
    
    # RULE BASE DIPERBAIKI - lebih agresif untuk belokan cepat
    rules = [
        # Error NL - respons cepat untuk belokan kiri
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),   
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),   # Lebih agresif
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),   
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']), 
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),   
        
        # Error NS - deteksi awal belokan
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),  
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),  # Lebih responsif
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),   # Antisipasi belokan
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),  
        
        # Error Z - lurus tapi siap belok
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),   # Antisipasi
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),     # LURUS
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),   # Antisipasi
        
        # Error PS - deteksi awal belokan kanan
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),  
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),   # Antisipasi belokan
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),  # Lebih responsif
        ctrl.Rule(error['PS'] & delta['PL'], output['R']),   
        
        # Error PL - respons cepat untuk belokan kanan
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),   
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']),  
        ctrl.Rule(error['PL'] & delta['Z'], output['LS']),   
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),   # Lebih agresif
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),   
    ]
    
    # Buat sistem kontrol
    control_system = ctrl.ControlSystem(rules)
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    
    return fuzzy_controller

def setup_camera():
    """
    Konfigurasi dan inisialisasi kamera
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    """
    Konfigurasi dan inisialisasi komunikasi serial
    """
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image_advanced(frame, line_detector):
    """
    Pemrosesan gambar yang lebih advanced dengan noise filtering
    """
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Advanced thresholding
    binary = line_detector.advanced_thresholding(blurred)
    
    # ROI yang lebih besar untuk deteksi belokan lebih awal
    roi_height = 100  # Diperbesar dari 80 ke 100
    roi = binary[240-roi_height:240, :]
    
    return gray, binary, roi

def calculate_line_position_advanced(roi, line_detector):
    """
    Deteksi posisi garis yang lebih robust dengan filtering
    """
    contour, cx, cy, confidence = line_detector.filter_contours_by_line_properties(roi)
    
    if contour is not None:
        # Adjust cy untuk offset ROI
        cy_adjusted = cy + (240 - roi.shape[0])
        return True, cx, cy_adjusted, confidence, contour
    else:
        return False, 0, 0, 0, None

def compute_fuzzy_control_fast(fuzzy_ctrl, error_val, delta_error, dead_zone=15):  # Dead zone diperkecil
    """
    Komputasi fuzzy yang lebih responsif untuk belokan cepat
    """
    try:
        # DEAD ZONE diperkecil untuk respons lebih cepat
        if abs(error_val) <= dead_zone and abs(delta_error) <= 8:
            print(f"[FLC] DEAD ZONE - Error: {error_val:4d} | Delta: {delta_error:4d} | Output: 0.00")
            return 0.0
        
        # Batasi input dalam range yang valid
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = fuzzy_ctrl.output['output']
        
        # Smoothing output minimal untuk respons cepat
        if abs(kontrol) < 5:  # Threshold lebih kecil
            kontrol = 0.0
            
        # Batasi output tapi tidak terlalu ketat
        kontrol = np.clip(kontrol, -90, 90)
        
        print(f"[FLC] Error: {error_val:4d} | Delta: {delta_error:4d} | Output: {kontrol:6.2f}")
        return kontrol
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm_responsive(kontrol, base_pwm=60):  # Base PWM dinaikkan
    """
    Perhitungan PWM yang lebih responsif untuk belokan cepat
    """
    # SCALING yang lebih agresif untuk respons cepat
    if abs(kontrol) <= 5:
        # Dead zone sangat kecil
        kontrol_scaled = 0
    elif abs(kontrol) <= 20:
        # Koreksi kecil - lebih agresif dari sebelumnya
        kontrol_scaled = kontrol * 0.15  # Dinaikkan dari 0.08
    elif abs(kontrol) <= 50:
        # Koreksi sedang - responsif untuk belokan
        kontrol_scaled = kontrol * 0.25  # Dinaikkan dari 0.12
    else:
        # Koreksi besar - maksimal untuk belokan tajam
        kontrol_scaled = kontrol * 0.35  # Dinaikkan dari 0.15
    
    # PWM calculation
    if abs(kontrol_scaled) < 0.5:
        pwm_kiri = base_pwm
        pwm_kanan = base_pwm
        arah = "LURUS"
        status = "DEAD_ZONE"
    else:
        pwm_kiri = base_pwm + kontrol_scaled   
        pwm_kanan = base_pwm - kontrol_scaled  
        
        if kontrol_scaled < -0.5:
            arah = "KIRI"
            status = f"CTRL_{abs(kontrol_scaled):.1f}"
        elif kontrol_scaled > 0.5:
            arah = "KANAN" 
            status = f"CTRL_{abs(kontrol_scaled):.1f}"
        else:
            arah = "LURUS"
            status = "MINIMAL"
    
    # Range PWM yang lebih luas untuk manuver cepat
    pwm_kiri = max(25, min(85, pwm_kiri))   # Range lebih luas: 25-85%
    pwm_kanan = max(25, min(85, pwm_kanan)) 
    
    print(f"[PWM] Kontrol: {kontrol:6.2f} | Scaled: {kontrol_scaled:5.2f} | {status:10s} | Kiri: {pwm_kiri:5.1f}% | Kanan: {pwm_kanan:5.1f}%")
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    """
    Mengirim perintah ke motor melalui serial
    """
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
            print(f"[UART] Sent: {cmd.strip()}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def visualize_tracking_advanced(frame, line_detected, cx=0, cy=0, error_val=0, kontrol=0, confidence=0, contour=None):
    """
    Visualisasi yang lebih informatif
    """
    # Gambar garis tengah referensi
    cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
    
    # Gambar ROI yang lebih besar
    cv2.rectangle(frame, (0, 140), (320, 240), (0, 255, 255), 2)
    
    if line_detected and contour is not None:
        # Gambar contour yang terdeteksi
        cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
        
        # Gambar titik tengah dan garis error
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(frame, (160, cy), (cx, cy), (0, 255, 0), 2)
        
        # Status text dengan background dan confidence
        status_text = f"Err:{error_val:3d} | Ctrl:{kontrol:5.1f} | Conf:{confidence:.2f}"
        cv2.rectangle(frame, (5, 5), (300, 35), (0, 0, 0), -1)
        
        # Warna berdasarkan confidence
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        cv2.rectangle(frame, (5, 5), (300, 35), (0, 0, 0), -1)
        cv2.putText(frame, "GARIS TIDAK DITEMUKAN", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    """
    Fungsi utama program - OPTIMIZED untuk respons cepat dan filtering noise
    """
    # Setup komponen
    print("=== Inisialisasi Sistem Line Following Robot - Advanced ===")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=2, confidence_threshold=0.6)  # Window lebih kecil untuk respons cepat
    line_detector = LineDetector()
    
    # Variabel kontrol
    prev_error = 0
    prev_kontrol = 0
    line_lost_timeout = 0
    MAX_LOST_TIME = 1.0  # Diperpendek untuk respons cepat
    
    # Statistik
    frame_count = 0
    start_program = time.time()
    detection_stats = {"good": 0, "poor": 0, "lost": 0}
    
    # Tunggu kamera stabil
    print("Menunggu kamera stabil...")
    time.sleep(2)
    
    try:
        print("=== Memulai Loop Utama - Fast Response Mode ===")
        while True:
            start_time = time.time()
            frame_count += 1
            
            # Ambil dan proses gambar dengan advanced filtering
            frame = picam2.capture_array()
            _, binary, roi = process_image_advanced(frame, line_detector)
            
            # Deteksi posisi garis dengan filtering
            line_detected, cx, cy, confidence, contour = calculate_line_position_advanced(roi, line_detector)
            
            if line_detected and confidence > 0.3:  # Threshold confidence minimal
                line_lost_timeout = 0
                
                # Update statistik
                if confidence > 0.7:
                    detection_stats["good"] += 1
                else:
                    detection_stats["poor"] += 1
                
                # Hitung error dengan filtering berbasis confidence
                raw_error = cx - 160
                error_val = error_filter.filter_error(raw_error, confidence)
                delta_error = error_val - prev_error
                
                # Debug info
                if frame_count % 15 == 0:
                    print(f"[DEBUG] Raw: {raw_error:4d} | Filtered: {error_val:4d} | Delta: {delta_error:4d} | Conf: {confidence:.2f}")
                
                # Komputasi fuzzy yang responsif
                kontrol = compute_fuzzy_control_fast(fuzzy_ctrl, error_val, delta_error, dead_zone=15)
                
                # Smoothing minimal untuk respons cepat
                if abs(kontrol - prev_kontrol) > 30:  # Threshold lebih besar
                    kontrol_smooth = prev_kontrol + np.sign(kontrol - prev_kontrol) * 15
                    print(f"[SMOOTH] {prev_kontrol:.1f} → {kontrol_smooth:.1f} (dari {kontrol:.1f})")
                    kontrol = kontrol_smooth
                
                # PWM responsif
                pwm_kiri, pwm_kanan = calculate_motor_pwm_responsive(kontrol, base_pwm=60)
                
                # Kirim ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                prev_error = error_val
                prev_kontrol = kontrol
                
            else:
                line_lost_timeout += 0.05
                detection_stats["lost"] += 1
                
                if line_lost_timeout >= MAX_LOST_TIME:
                    send_motor_commands(ser, 0, 0)
                    print("[WARN] Garis hilang, motor dihentikan")
                else:
                    print(f"[WARN] Garis hilang/confidence rendah: {confidence:.2f}")
                    
                kontrol = 0
                error_val = 0
                confidence = 0
            
            # Visualisasi advanced
            frame = visualize_tracking_advanced(frame, line_detected, cx, cy, error_val, kontrol, confidence, contour)
            
            # Tampilkan statistik deteksi
            if frame_count % 20 == 0:
                total = sum(detection_stats.values())
                if total > 0:
                    good_pct = detection_stats["good"] / total * 100
                    cv2.putText(frame, f"Good: {good_pct:.1f}%", (5, 215),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Tampilkan frame
            cv2.imshow("Advanced Line Following Robot", frame)
            cv2.imshow("ROI Binary", roi)
            cv2.imshow("Full Binary", binary)
            
            # Kontrol
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("[INFO] Reset sistem")
                error_filter = ErrorFilter(window_size=2, confidence_threshold=0.6)
                line_detector = LineDetector()
                prev_error = 0
                prev_kontrol = 0
                detection_stats = {"good": 0, "poor": 0, "lost": 0}
            elif key == ord('t'):
                # Toggle threshold mode
                print("[INFO] Menggunakan threshold berbeda")
            
            # Frame rate control - dinaikkan ke 15 FPS untuk respons cepat
            elapsed = time.time() - start_time
            target_fps = 1.0/15.0  # 15 FPS
            if elapsed < target_fps:
                time.sleep(target_fps - elapsed)
            
            # Status berkala
            if frame_count % 150 == 0:
                uptime = time.time() - start_program
                avg_fps = frame_count / uptime
                total_detections = sum(detection_stats.values())
                if total_detections > 0:
                    success_rate = detection_stats["good"] / total_detections * 100
                    print(f"[STATUS] Frame: {frame_count} | FPS: {avg_fps:.1f} | Success: {success_rate:.1f}%")
            
    except KeyboardInterrupt:
        print("\n=== Program dihentikan ===")
    
    finally:
        # Cleanup
        print("=== Cleanup ===")
        try:
            send_motor_commands(ser, 0, 0)
            time.sleep(0.5)
        except:
            pass
            
        cv2.destroyAllWindows()
        picam2.stop()
        if ser:
            ser.close()
        
        # Final stats
        total_time = time.time() - start_program
        if frame_count > 0:
            avg_fps = frame_count / total_time
            total_detections = sum(detection_stats.values())
            success_rate = detection_stats["good"] / total_detections * 100 if total_detections > 0 else 0
            print(f"[FINAL] Runtime: {total_time:.1f}s | Avg FPS: {avg_fps:.1f} | Success Rate: {success_rate:.1f}%")
        
        print("=== Program Selesai ===")

if __name__ == "__main__":
    main()
