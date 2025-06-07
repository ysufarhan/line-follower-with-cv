from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ErrorFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.6  # Sedikit lebih smooth untuk kamera rendah

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Moving average dengan weight lebih besar untuk nilai terbaru
        if len(self.error_history) >= 3:
            weights = [0.2, 0.3, 0.5]  # Prioritas nilai terbaru
            weighted_sum = sum(w * e for w, e in zip(weights, self.error_history[-3:]))
            smoothed_error = weighted_sum
        else:
            smoothed_error = sum(self.error_history) / len(self.error_history)
            
        return int(smoothed_error)

def setup_fuzzy_logic():
    # Universe disesuaikan untuk kamera rendah - range error lebih besar
    error = ctrl.Antecedent(np.arange(-300, 301, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-200, 201, 1), 'delta')
    output = ctrl.Consequent(np.arange(-200, 201, 1), 'output')

    # ERROR - Disesuaikan untuk perspektif kamera rendah
    error['NL'] = fuzz.trimf(error.universe, [-300, -200, -80])
    error['NS'] = fuzz.trimf(error.universe, [-120, -50, -10])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])      
    error['PS'] = fuzz.trimf(error.universe, [10, 50, 120])
    error['PL'] = fuzz.trimf(error.universe, [80, 200, 300])

    # DELTA - Responsif untuk perubahan cepat
    delta['NL'] = fuzz.trimf(delta.universe, [-200, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -25, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])      
    delta['PS'] = fuzz.trimf(delta.universe, [5, 25, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 200])

    # OUTPUT - Range lebih besar untuk koreksi yang lebih agresif
    output['L']  = fuzz.trimf(output.universe, [-200, -120, -80])
    output['LS'] = fuzz.trimf(output.universe, [-100, -50, -15])
    output['Z']  = fuzz.trimf(output.universe, [-20, 0, 20])    
    output['RS'] = fuzz.trimf(output.universe, [15, 50, 100])
    output['R']  = fuzz.trimf(output.universe, [80, 120, 200])

    # Rules yang sama tapi lebih responsif
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NL'] & delta['PL'], output['RS']),

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    # Konfigurasi optimal untuk RPi4 - resolusi lebih rendah untuk performa
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(3)  # RPi4 butuh waktu lebih lama untuk stabilize
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
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur - kernel lebih kecil untuk RPi4 (performa)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu threshold - otomatis cari threshold optimal
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ROI disesuaikan untuk kamera rendah - untuk resolusi 320x240
    roi_start = int(frame.shape[0] * 0.6)  # Mulai dari 60% tinggi gambar (pixel 144)
    roi = binary[roi_start:, :]
    
    return gray, binary, roi, roi_start

def calculate_line_position(roi):
    # Morphological operations - kernel lebih kecil untuk RPi4
    kernel = np.ones((3,3), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    # Cari contours untuk deteksi line yang lebih akurat
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Ambil contour terbesar (kemungkinan line)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Threshold area disesuaikan untuk resolusi 320x240
        if area > 200:  # Threshold area minimum untuk resolusi rendah
            # Hitung moments untuk mendapatkan centroid
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return True, cx, cy, area
    
    return False, 0, 0, 0

def perspective_correction(error, roi_height=96):
    # Koreksi perspektif untuk kamera rendah - disesuaikan untuk resolusi 320x240
    # ROI height = 240 - 144 = 96 pixels
    perspective_factor = 1.0 + (roi_height * 0.003)  # Faktor koreksi untuk resolusi rendah
    corrected_error = error * perspective_factor
    return int(corrected_error)

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -300, 300)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -200, 200)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -200, 200)
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=55, scaling_factor=0.4):
    # Scaling factor lebih besar untuk responsivitas yang lebih baik
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Range PWM yang lebih lebar untuk manuver yang lebih agresif
    pwm_kiri = max(30, min(80, pwm_kiri))  
    pwm_kanan = max(30, min(80, pwm_kanan))
    return int(pwm_kiri), int(pwm_kanan)

def adaptive_speed_control(error, base_pwm=55):
    # Kurangi kecepatan saat error besar (saat berbelok)
    abs_error = abs(error)
    if abs_error > 100:
        return base_pwm - 15  # Kurangi speed saat berbelok tajam
    elif abs_error > 50:
        return base_pwm - 8   # Kurangi speed sedikit saat berbelok
    else:
        return base_pwm       # Speed normal saat lurus

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def main():
    print("[INFO] Inisialisasi sistem...")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=5)

    prev_error = 0
    frame_count = 0
    lost_line_count = 0
    last_valid_error = 0

    print("[INFO] Sistem siap! Tekan 'q' untuk keluar.")
    
    try:
        while True:
            frame = picam2.capture_array()
            gray, binary, roi, roi_start = process_image(frame)

            line_detected, cx, cy, area = calculate_line_position(roi)
            
            if line_detected:
                lost_line_count = 0
                
                # Hitung error relatif terhadap center frame
                center_x = frame.shape[1] // 2
                raw_error = cx - center_x
                
                # Terapkan koreksi perspektif
                error = perspective_correction(raw_error)
                
                # Filter error untuk smooth movement
                error = error_filter.filter_error(error)
                
                # Hitung delta error
                delta_error = error - prev_error
                prev_error = error
                last_valid_error = error

                # Hitung kontrol fuzzy
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                
                # Adaptive speed berdasarkan error
                adaptive_base_pwm = adaptive_speed_control(error)
                
                # Hitung PWM motor
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol, adaptive_base_pwm)
                
                # Kirim command ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                # Debug info
                if frame_count % 10 == 0:
                    print(f"[DEBUG] Err:{error:4d} ΔErr:{delta_error:4d} FLC:{kontrol:6.1f} PWM:L{pwm_kiri} R{pwm_kanan} Area:{area:4.0f}")
                    
            else:
                lost_line_count += 1
                
                if lost_line_count < 10:  # Coba recovery dulu
                    # Gunakan error terakhir yang valid untuk recovery
                    recovery_control = last_valid_error * 0.3
                    pwm_kiri, pwm_kanan = calculate_motor_pwm(recovery_control, 40)
                    send_motor_commands(ser, pwm_kiri, pwm_kanan)
                    print(f"[RECOVERY] Menggunakan error terakhir: {last_valid_error}")
                else:
                    # Stop motor jika line hilang terlalu lama
                    send_motor_commands(ser, 0, 0)
                    if frame_count % 30 == 0:
                        print("[WARNING] Garis hilang - Robot berhenti")

            # Visualisasi untuk debugging - dikurangi frekuensi untuk RPi4
            if frame_count % 10 == 0:  # Update display setiap 10 frame untuk performa
                frame_display = frame.copy()
                
                # Gambar ROI area
                cv2.rectangle(frame_display, (0, roi_start), (frame.shape[1], frame.shape[0]), (0, 255, 0), 1)
                
                # Gambar center line
                center_x = frame.shape[1] // 2
                cv2.line(frame_display, (center_x, 0), (center_x, frame.shape[0]), (0, 255, 0), 1)
                
                if line_detected:
                    # Gambar detected line point
                    actual_cy = cy + roi_start
                    cv2.circle(frame_display, (cx, actual_cy), 4, (0, 0, 255), -1)
                    
                    # Tampilkan error
                    cv2.putText(frame_display, f"E:{error}", (5, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_display, f"PWM:L{pwm_kiri} R{pwm_kanan}", (5, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Tampilkan feed - resize untuk performa yang lebih baik
                cv2.imshow("Line Following - RPi4", frame_display)
                cv2.imshow("Binary ROI", roi)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.05)  # ~20 FPS untuk RPi4 (lebih lambat dari RPi5)
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh user")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # Cleanup
        print("[INFO] Membersihkan sistem...")
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
