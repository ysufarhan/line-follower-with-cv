from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from threading import Thread
import queue

class OptimizedLineFollower:
    def __init__(self):
        self.image_queue = queue.Queue(maxsize=2)
        self.running = True
        self.current_error = 0
        self.prev_error = 0
        self.line_detected = False
        
        # Setup komponen
        self.fuzzy_ctrl = self.setup_fuzzy_logic()
        self.picam2 = self.setup_camera()
        self.ser = self.setup_serial()
        
        # Thread untuk capture image
        self.capture_thread = Thread(target=self.capture_images, daemon=True)
        
    def setup_fuzzy_logic(self):
        """Setup sistem fuzzy logic yang dioptimasi untuk smooth control"""
        # Universe yang lebih fokus untuk presisi
        error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
        delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
        output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

        # Membership functions yang dioptimasi untuk smooth turning
        # ERROR - Zona tengah lebih lebar untuk stabilitas garis lurus
        error['NL'] = fuzz.trimf(error.universe, [-160, -120, -60])
        error['NS'] = fuzz.trimf(error.universe, [-80, -30, -5])
        error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])      # Zona netral lebih lebar
        error['PS'] = fuzz.trimf(error.universe, [5, 30, 80])
        error['PL'] = fuzz.trimf(error.universe, [60, 120, 160])

        # DELTA - Untuk prediksi pergerakan
        delta['NL'] = fuzz.trimf(delta.universe, [-100, -60, -20])
        delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -3])
        delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
        delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 40])
        delta['PL'] = fuzz.trimf(delta.universe, [20, 60, 100])

        # OUTPUT - Smooth control untuk turning yang optimal
        output['LL'] = fuzz.trimf(output.universe, [-100, -80, -50])  # Belok kiri tajam
        output['L']  = fuzz.trimf(output.universe, [-60, -35, -15])   # Belok kiri sedang
        output['LS'] = fuzz.trimf(output.universe, [-25, -10, -2])    # Belok kiri halus
        output['Z']  = fuzz.trimf(output.universe, [-5, 0, 5])        # Lurus
        output['RS'] = fuzz.trimf(output.universe, [2, 10, 25])       # Belok kanan halus
        output['R']  = fuzz.trimf(output.universe, [15, 35, 60])      # Belok kanan sedang
        output['RR'] = fuzz.trimf(output.universe, [50, 80, 100])     # Belok kanan tajam

        # Rules yang dioptimasi untuk smooth control
        rules = [
            # Error besar kiri (NL)
            ctrl.Rule(error['NL'] & delta['NL'], output['LL']),
            ctrl.Rule(error['NL'] & delta['NS'], output['L']),
            ctrl.Rule(error['NL'] & delta['Z'], output['L']),
            ctrl.Rule(error['NL'] & delta['PS'], output['LS']),
            ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

            # Error sedang kiri (NS)
            ctrl.Rule(error['NS'] & delta['NL'], output['L']),
            ctrl.Rule(error['NS'] & delta['NS'], output['L']),
            ctrl.Rule(error['NS'] & delta['Z'], output['LS']),
            ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
            ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

            # Error netral (Z) - Prioritas lurus
            ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
            ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
            ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
            ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
            ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

            # Error sedang kanan (PS)
            ctrl.Rule(error['PS'] & delta['NL'], output['RS']),
            ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
            ctrl.Rule(error['PS'] & delta['Z'], output['RS']),
            ctrl.Rule(error['PS'] & delta['PS'], output['R']),
            ctrl.Rule(error['PS'] & delta['PL'], output['R']),

            # Error besar kanan (PL)
            ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
            ctrl.Rule(error['PL'] & delta['NS'], output['RS']),
            ctrl.Rule(error['PL'] & delta['Z'], output['R']),
            ctrl.Rule(error['PL'] & delta['PS'], output['R']),
            ctrl.Rule(error['PL'] & delta['PL'], output['RR']),
        ]

        control_system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(control_system)

    def setup_camera(self):
        """Setup kamera dengan resolusi optimal untuk Pi 4"""
        picam2 = Picamera2()
        # Resolusi yang lebih rendah untuk processing cepat
        config = picam2.create_still_configuration(main={"size": (240, 180)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Warm up kamera
        return picam2

    def setup_serial(self):
        """Setup komunikasi serial"""
        try:
            ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
            print("[UART] Port serial berhasil dibuka")
            return ser
        except Exception as e:
            print(f"[UART ERROR] Gagal membuka serial port: {e}")
            return None

    def capture_images(self):
        """Thread terpisah untuk capture image - mengurangi blocking"""
        while self.running:
            try:
                frame = self.picam2.capture_array()
                if not self.image_queue.full():
                    self.image_queue.put(frame)
                time.sleep(0.02)  # ~50 FPS capture rate
            except:
                pass

    def process_image_optimized(self, frame):
        """Image processing yang dioptimasi untuk kecepatan"""
        # Crop langsung ke ROI untuk mengurangi processing
        roi_y_start = int(frame.shape[0] * 0.7)  # 70% dari bawah
        roi = frame[roi_y_start:, :]
        
        # Konversi ke grayscale hanya ROI
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
            
        # Threshold adaptif yang cepat
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphology ringan untuk cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary, roi_y_start

    def calculate_line_position_optimized(self, binary_roi):
        """Perhitungan posisi garis yang dioptimasi"""
        # Menggunakan moments untuk mencari centroid
        M = cv2.moments(binary_roi)
        
        if M['m00'] > 50:  # Threshold area minimum
            cx = int(M['m10'] / M['m00'])
            return True, cx
        
        # Fallback: cari kolom dengan pixel putih terbanyak
        column_sums = np.sum(binary_roi, axis=0)
        if np.max(column_sums) > 20:
            cx = np.argmax(column_sums)
            return True, cx
            
        return False, 0

    def compute_fuzzy_control(self, error_val, delta_error):
        """Komputasi fuzzy control yang aman"""
        try:
            # Clip input sesuai universe
            error_clipped = np.clip(error_val, -160, 160)
            delta_clipped = np.clip(delta_error, -100, 100)
            
            self.fuzzy_ctrl.input['error'] = error_clipped
            self.fuzzy_ctrl.input['delta'] = delta_clipped
            self.fuzzy_ctrl.compute()
            
            return np.clip(self.fuzzy_ctrl.output['output'], -100, 100)
        except Exception as e:
            print(f"[FLC ERROR] {e}")
            return 0.0

    def calculate_motor_pwm_smooth(self, kontrol, base_pwm=60):
        """
        Perhitungan PWM yang dioptimasi untuk smooth turning
        - Garis lurus: PWM kiri = kanan
        - Belok kiri: PWM kanan dominan, PWM kiri disesuaikan
        - Belok kanan: PWM kiri dominan, PWM kanan disesuaikan
        """
        # Scaling factor yang lebih halus
        scaling_factor = 0.6
        kontrol_scaled = kontrol * scaling_factor
        
        if abs(kontrol) < 5:  # Zona lurus
            # PWM sama untuk kedua motor
            pwm_kiri = base_pwm
            pwm_kanan = base_pwm
        elif kontrol < 0:  # Perlu belok kiri (error negatif)
            # Motor kanan dominan (lebih cepat), motor kiri diperlambat
            pwm_kanan = base_pwm + abs(kontrol_scaled)
            pwm_kiri = base_pwm - abs(kontrol_scaled) * 0.7
        else:  # Perlu belok kanan (error positif)
            # Motor kiri dominan (lebih cepat), motor kanan diperlambat
            pwm_kiri = base_pwm + abs(kontrol_scaled)
            pwm_kanan = base_pwm - abs(kontrol_scaled) * 0.7
        
        # Batasi PWM dalam range yang aman
        pwm_kiri = max(25, min(90, pwm_kiri))
        pwm_kanan = max(25, min(90, pwm_kanan))
        
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
        """Main loop yang dioptimasi"""
        print("[INFO] Memulai line follower...")
        
        # Mulai thread capture
        self.capture_thread.start()
        
        frame_count = 0
        fps_time = time.time()
        
        try:
            while self.running:
                # Ambil frame terbaru dari queue
                if not self.image_queue.empty():
                    frame = self.image_queue.get()
                    
                    # Process image
                    binary_roi, roi_offset = self.process_image_optimized(frame)
                    
                    # Deteksi posisi garis
                    line_detected, cx = self.calculate_line_position_optimized(binary_roi)
                    
                    if line_detected:
                        # Hitung error relatif terhadap center
                        center_x = frame.shape[1] // 2
                        error = cx - center_x
                        delta_error = error - self.prev_error
                        
                        # Fuzzy control
                        kontrol = self.compute_fuzzy_control(error, delta_error)
                        
                        # Hitung PWM motor
                        pwm_kiri, pwm_kanan = self.calculate_motor_pwm_smooth(kontrol)
                        
                        # Kirim perintah motor
                        self.send_motor_commands(pwm_kiri, pwm_kanan)
                        
                        # Update state
                        self.current_error = error
                        self.prev_error = error
                        self.line_detected = True
                        
                        # Debug info
                        if frame_count % 15 == 0:
                            direction = "LURUS" if abs(error) < 20 else ("KIRI" if error < 0 else "KANAN")
                            print(f"[DEBUG] E:{error:4d} | FLC:{kontrol:6.2f} | PWM L:{pwm_kiri} R:{pwm_kanan} | {direction}")
                    
                    else:
                        # Garis tidak terdeteksi
                        self.send_motor_commands(0, 0)
                        self.line_detected = False
                        if frame_count % 30 == 0:
                            print("[WARN] Garis tidak terdeteksi - STOP")
                    
                    # Visualisasi (opsional, comment untuk performa maksimal)
                    if frame_count % 5 == 0:  # Update display setiap 5 frame
                        self.show_debug_display(frame, binary_roi, cx if line_detected else None)
                    
                    frame_count += 1
                    
                    # FPS monitoring
                    if time.time() - fps_time > 5:
                        fps = frame_count / (time.time() - fps_time)
                        print(f"[INFO] FPS: {fps:.1f}")
                        frame_count = 0
                        fps_time = time.time()
                
                # Small delay untuk mencegah CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[INFO] Dihentikan oleh pengguna")
        finally:
            self.cleanup()

    def show_debug_display(self, frame, binary_roi, cx):
        """Tampilan debug (opsional)"""
        try:
            # Tampilkan frame dengan overlay
            display_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Garis tengah referensi
            cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            
            # ROI boundary
            roi_y = int(h * 0.7)
            cv2.line(display_frame, (0, roi_y), (w, roi_y), (255, 0, 0), 1)
            
            # Posisi garis terdeteksi
            if cx is not None:
                cv2.circle(display_frame, (cx, roi_y + binary_roi.shape[0]//2), 3, (0, 0, 255), -1)
                
            # Status info
            status = f"Line: {'OK' if self.line_detected else 'NO'} | Error: {self.current_error}"
            cv2.putText(display_frame, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Tampilkan
            cv2.imshow("Line Follower", cv2.resize(display_frame, (320, 240)))
            cv2.imshow("Binary ROI", cv2.resize(binary_roi, (320, 60)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        except:
            pass  # Ignore display errors

    def cleanup(self):
        """Cleanup resources"""
        print("[INFO] Membersihkan resources...")
        self.running = False
        
        # Stop motors
        self.send_motor_commands(0, 0)
        
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
    follower = OptimizedLineFollower()
    follower.run()

if __name__ == "__main__":
    main()
