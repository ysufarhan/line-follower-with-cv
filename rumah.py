from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class StableLineFollower:
    def __init__(self):
        self.running = True
        self.current_error = 0
        self.prev_error = 0
        self.line_detected = False
        
        # Filter untuk stabilitas
        self.error_history = []
        self.error_filter_size = 5
        self.lost_line_counter = 0
        self.max_lost_frames = 10
        
        # Setup komponen
        self.fuzzy_ctrl = self.setup_fuzzy_logic()
        self.picam2 = self.setup_camera()
        self.ser = self.setup_serial()
        
    def setup_fuzzy_logic(self):
        """Setup fuzzy logic yang sangat stabil untuk jalan lurus"""
        # Universe yang lebih konservatif
        error = ctrl.Antecedent(np.arange(-120, 121, 1), 'error')
        delta = ctrl.Antecedent(np.arange(-50, 51, 1), 'delta')
        output = ctrl.Consequent(np.arange(-60, 61, 1), 'output')

        # Membership functions dengan zona tengah yang SANGAT LEBAR
        # ERROR - Zona tengah diperlebar drastis untuk stabilitas
        error['NL'] = fuzz.trimf(error.universe, [-120, -80, -40])
        error['NS'] = fuzz.trimf(error.universe, [-60, -25, -5])
        error['Z']  = fuzz.trimf(error.universe, [-30, 0, 30])      # ZONA TENGAH SANGAT LEBAR
        error['PS'] = fuzz.trimf(error.universe, [5, 25, 60])
        error['PL'] = fuzz.trimf(error.universe, [40, 80, 120])

        # DELTA - Sangat konservatif
        delta['NL'] = fuzz.trimf(delta.universe, [-50, -30, -10])
        delta['NS'] = fuzz.trimf(delta.universe, [-20, -8, -2])
        delta['Z']  = fuzz.trimf(delta.universe, [-5, 0, 5])
        delta['PS'] = fuzz.trimf(delta.universe, [2, 8, 20])
        delta['PL'] = fuzz.trimf(delta.universe, [10, 30, 50])

        # OUTPUT - Range kecil untuk gerakan halus
        output['L']  = fuzz.trimf(output.universe, [-60, -40, -20])
        output['LS'] = fuzz.trimf(output.universe, [-30, -15, -3])
        output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])      # Output netral lebar
        output['RS'] = fuzz.trimf(output.universe, [3, 15, 30])
        output['R']  = fuzz.trimf(output.universe, [20, 40, 60])

        # Rules yang sangat konservatif - prioritas LURUS
        rules = [
            # Error besar kiri (NL)
            ctrl.Rule(error['NL'] & delta['NL'], output['L']),
            ctrl.Rule(error['NL'] & delta['NS'], output['L']),
            ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
            ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
            ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

            # Error sedang kiri (NS)
            ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
            ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
            ctrl.Rule(error['NS'] & delta['Z'], output['Z']),       # Lebih cenderung lurus
            ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
            ctrl.Rule(error['NS'] & delta['PL'], output['Z']),

            # Error netral (Z) - SELALU LURUS
            ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
            ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
            ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
            ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
            ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

            # Error sedang kanan (PS)
            ctrl.Rule(error['PS'] & delta['NL'], output['Z']),
            ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
            ctrl.Rule(error['PS'] & delta['Z'], output['Z']),       # Lebih cenderung lurus
            ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
            ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

            # Error besar kanan (PL)
            ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
            ctrl.Rule(error['PL'] & delta['NS'], output['RS']),
            ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
            ctrl.Rule(error['PL'] & delta['PS'], output['R']),
            ctrl.Rule(error['PL'] & delta['PL'], output['R']),
        ]

        control_system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(control_system)

    def setup_camera(self):
        """Setup kamera dengan konfigurasi stabil"""
        picam2 = Picamera2()
        # Resolusi kecil untuk processing cepat dan stabil
        config = picam2.create_still_configuration(main={"size": (160, 120)})
        picam2.configure(config)
        picam2.start()
        time.sleep(3)  # Warm up lebih lama
        return picam2

    def setup_serial(self):
        """Setup komunikasi serial"""
        try:
            ser = serial.Serial('/dev/serial0', 115200, timeout=0.1)
            print("[UART] Port serial berhasil dibuka")
            return ser
        except Exception as e:
            print(f"[UART ERROR] Gagal membuka serial port: {e}")
            return None

    def filter_error(self, error):
        """Filter error dengan moving average untuk stabilitas"""
        self.error_history.append(error)
        if len(self.error_history) > self.error_filter_size:
            self.error_history.pop(0)
        
        # Weighted average - nilai terbaru lebih berpengaruh
        weights = np.arange(1, len(self.error_history) + 1)
        weighted_avg = np.average(self.error_history, weights=weights)
        return int(weighted_avg)

    def process_image_stable(self, frame):
        """Image processing yang sangat stabil"""
        height, width = frame.shape[:2]
        
        # ROI yang lebih fokus - hanya bagian bawah
        roi_height = height // 3
        roi_start = height - roi_height
        roi = frame[roi_start:, :]
        
        # Konversi ke grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # Pre-processing untuk hasil threshold yang stabil
        # 1. Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced = clahe.apply(blurred)
        
        # 3. Threshold adaptif yang lebih stabil
        # Kombinasi OTSU dan threshold manual
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Gabungkan kedua threshold untuk hasil yang lebih stabil
        binary = cv2.bitwise_and(thresh1, thresh2)
        
        # 4. Morphological operations untuk cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 5. Dilasi ringan untuk memperkuat garis
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        return binary, roi_start

    def calculate_line_position_stable(self, binary_roi):
        """Perhitungan posisi garis yang sangat stabil"""
        height, width = binary_roi.shape
        
        # Method 1: Weighted centroid (lebih stabil dari moments biasa)
        y_coords, x_coords = np.where(binary_roi > 0)
        
        if len(x_coords) > 20:  # Minimum pixel count
            # Berikan weight lebih besar pada pixel di bagian bawah ROI
            weights = (height - y_coords) / height
            weighted_x = np.average(x_coords, weights=weights)
            return True, int(weighted_x)
        
        # Method 2: Fallback - cari kolom dengan intensitas maksimal
        column_sums = np.sum(binary_roi, axis=0)
        if np.max(column_sums) > 1000:  # Threshold minimum
            # Smooth column sums untuk mengurangi noise
            smoothed = cv2.GaussianBlur(column_sums.reshape(1, -1), (9, 1), 0).flatten()
            cx = np.argmax(smoothed)
            return True, cx
            
        return False, 0

    def compute_fuzzy_control(self, error_val, delta_error):
        """Komputasi fuzzy control dengan error handling"""
        try:
            # Clip input sesuai universe
            error_clipped = np.clip(error_val, -120, 120)
            delta_clipped = np.clip(delta_error, -50, 50)
            
            self.fuzzy_ctrl.input['error'] = error_clipped
            self.fuzzy_ctrl.input['delta'] = delta_clipped
            self.fuzzy_ctrl.compute()
            
            output = self.fuzzy_ctrl.output['output']
            return np.clip(output, -60, 60)
        except Exception as e:
            print(f"[FLC ERROR] {e}")
            return 0.0

    def calculate_motor_pwm_stable(self, kontrol, base_pwm=55):
        """
        Perhitungan PWM yang sangat stabil dengan dead zone
        Fokus pada jalan lurus yang stabil
        """
        # Dead zone yang lebih besar untuk mencegah gerakan tidak perlu
        if abs(kontrol) < 3:  # Dead zone diperbesar
            # PWM sama untuk jalan lurus
            return base_pwm, base_pwm
        
        # Scaling factor yang sangat konservatif
        scaling_factor = 0.4
        kontrol_scaled = kontrol * scaling_factor
        
        if kontrol < 0:  # Perlu belok kiri
            # Motor kanan sedikit lebih cepat
            pwm_kanan = base_pwm + abs(kontrol_scaled)
            pwm_kiri = base_pwm - abs(kontrol_scaled) * 0.5  # Pengurangan lebih sedikit
        else:  # Perlu belok kanan
            # Motor kiri sedikit lebih cepat
            pwm_kiri = base_pwm + abs(kontrol_scaled)
            pwm_kanan = base_pwm - abs(kontrol_scaled) * 0.5  # Pengurangan lebih sedikit
        
        # Batasi PWM dalam range yang sangat aman
        pwm_kiri = max(35, min(75, pwm_kiri))
        pwm_kanan = max(35, min(75, pwm_kanan))
        
        return int(pwm_kiri), int(pwm_kanan)

    def send_motor_commands(self, pwm_kiri, pwm_kanan):
        """Kirim perintah motor via serial"""
        if self.ser:
            try:
                cmd = f"{pwm_kiri},{pwm_kanan}\n"
                self.ser.write(cmd.encode())
                self.ser.flush()
            except Exception as e:
                print(f"[SERIAL ERROR] {e}")

    def run(self):
        """Main loop yang sangat stabil"""
        print("[INFO] Memulai stable line follower...")
        
        frame_count = 0
        stable_error = 0
        
        try:
            while self.running:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Process image dengan metode yang stabil
                binary_roi, roi_offset = self.process_image_stable(frame)
                
                # Deteksi posisi garis
                line_detected, cx = self.calculate_line_position_stable(binary_roi)
                
                if line_detected:
                    # Reset lost line counter
                    self.lost_line_counter = 0
                    
                    # Hitung error relatif terhadap center
                    center_x = binary_roi.shape[1] // 2
                    raw_error = cx - center_x
                    
                    # Filter error untuk stabilitas
                    filtered_error = self.filter_error(raw_error)
                    
                    # Hitung delta error
                    delta_error = filtered_error - self.prev_error
                    
                    # Fuzzy control
                    kontrol = self.compute_fuzzy_control(filtered_error, delta_error)
                    
                    # Hitung PWM motor dengan stabilitas tinggi
                    pwm_kiri, pwm_kanan = self.calculate_motor_pwm_stable(kontrol)
                    
                    # Kirim perintah motor
                    self.send_motor_commands(pwm_kiri, pwm_kanan)
                    
                    # Update state
                    self.current_error = filtered_error
                    self.prev_error = filtered_error
                    self.line_detected = True
                    stable_error = filtered_error
                    
                    # Debug info
                    if frame_count % 20 == 0:
                        status = "LURUS" if abs(filtered_error) < 10 else ("KIRI" if filtered_error < 0 else "KANAN")
                        print(f"[DEBUG] Raw:{raw_error:3d} Filt:{filtered_error:3d} FLC:{kontrol:5.1f} PWM L:{pwm_kiri} R:{pwm_kanan} | {status}")
                
                else:
                    # Garis tidak terdeteksi
                    self.lost_line_counter += 1
                    
                    if self.lost_line_counter < self.max_lost_frames:
                        # Gunakan error terakhir yang stabil untuk sementara
                        kontrol = self.compute_fuzzy_control(stable_error, 0)
                        pwm_kiri, pwm_kanan = self.calculate_motor_pwm_stable(kontrol, base_pwm=45)
                        self.send_motor_commands(pwm_kiri, pwm_kanan)
                        
                        if frame_count % 10 == 0:
                            print(f"[WARN] Garis hilang {self.lost_line_counter}/{self.max_lost_frames} - menggunakan error terakhir")
                    else:
                        # Stop jika garis hilang terlalu lama
                        self.send_motor_commands(0, 0)
                        if frame_count % 30 == 0:
                            print("[STOP] Garis tidak terdeteksi - BERHENTI")
                
                # Visualisasi debug (comment untuk performa maksimal)
                if frame_count % 8 == 0:
                    self.show_debug_display(frame, binary_roi, cx if line_detected else None)
                
                frame_count += 1
                
                # Delay untuk stabilitas
                time.sleep(0.05)  # 20 FPS - lebih stabil
                
        except KeyboardInterrupt:
            print("\n[INFO] Dihentikan oleh pengguna")
        finally:
            self.cleanup()

    def show_debug_display(self, frame, binary_roi, cx):
        """Tampilan debug"""
        try:
            # Resize untuk tampilan
            display_frame = cv2.resize(frame, (320, 240))
            binary_display = cv2.resize(binary_roi, (320, 80))
            
            h, w = display_frame.shape[:2]
            
            # Garis tengah referensi
            cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
            
            # ROI boundary  
            roi_y = int(h * 0.67)
            cv2.line(display_frame, (0, roi_y), (w, roi_y), (255, 0, 0), 1)
            
            # Posisi garis terdeteksi
            if cx is not None:
                # Scale cx sesuai resize
                cx_scaled = int(cx * 320 / 160)
                cv2.circle(display_frame, (cx_scaled, roi_y + 30), 4, (0, 0, 255), -1)
                
            # Status info
            status = f"Line: {'OK' if self.line_detected else 'NO'} | Error: {self.current_error} | Lost: {self.lost_line_counter}"
            cv2.putText(display_frame, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Tampilkan
            cv2.imshow("Stable Line Follower", display_frame)
            cv2.imshow("Stable Binary", binary_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        except:
            pass

    def cleanup(self):
        """Cleanup resources"""
        print("[INFO] Membersihkan resources...")
        self.running = False
        
        # Stop motors
        self.send_motor_commands(0, 0)
        time.sleep(0.5)  # Pastikan motor stop
        
        # Close serial
        if self.ser:
            self.ser.close()
            
        # Stop camera
        if self.picam2:
            self.picam2.stop()
            
        # Close windows
        cv2.destroyAllWindows()
        
        print("[INFO] Program selesai")

def main():
    """Main function"""
    follower = StableLineFollower()
    follower.run()

if __name__ == "__main__":
    main()
