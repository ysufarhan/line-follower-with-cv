from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# import matplotlib.pyplot as plt # Bisa di-uncomment jika ingin visualisasi FLC membership functions

class ErrorFilter:
    def __init__(self, window_size=3): # PERUBAHAN: window_size diperkecil untuk respons lebih cepat
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.7 # PERUBAHAN: alpha ditingkatkan untuk respon lebih cepat
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        # Median filter untuk mengurangi noise
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors)//2]
        
        # Exponential smoothing
        if len(self.error_history) > 1:
            # Gunakan median_error saat ini dan median_error sebelumnya untuk smoothing
            # Ini sedikit berbeda dari implementasi sebelumnya yang menggunakan sorted_errors[-2]
            # yang mungkin bukan median dari history sebelumnya.
            prev_median = sorted(self.error_history[:-1])[len(self.error_history[:-1])//2] if len(self.error_history) > 1 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_median
        else:
            smoothed_error = median_error
        
        return int(smoothed_error)

class LineRecovery:
    def __init__(self):
        self.lost_count = 0
        self.last_valid_error = 0
        self.search_direction = 1 # 1 untuk kanan, -1 untuk kiri
        self.recovery_speed = 40 # Bisa disesuaikan jika robot terlalu cepat/lambat mencari
        
    def handle_line_lost(self, ser):
        self.lost_count += 1
        
        if self.lost_count < 10: # Coba diam dulu (bisa disesuaikan)
            self.send_motor_commands(ser, 0, 0)
            return "stop"
        elif self.lost_count < 30: # Mundur sedikit (bisa disesuaikan)
            self.send_motor_commands(ser, -30, -30) # Mundur dengan PWM -30
            return "reverse"
        else: # Cari garis dengan berputar
            # Berputar ke arah terakhir garis terdeteksi
            if self.last_valid_error > 0: # Garis terakhir di kanan (putar kanan)
                self.send_motor_commands(ser, self.recovery_speed, -self.recovery_speed)
                return "search_right"
            else: # Garis terakhir di kiri (putar kiri)
                self.send_motor_commands(ser, -self.recovery_speed, self.recovery_speed)
                return "search_left"
    
    def line_found(self, error):
        self.lost_count = 0
        self.last_valid_error = error
        
    def send_motor_commands(self, ser, pwm_kiri, pwm_kanan):
        if ser:
            try:
                # CATATAN: Ini adalah fungsi duplikat dari yang ada di luar class.
                # Sebaiknya motor_commands dihandle oleh satu fungsi utama di main loop
                # atau panggil fungsi global send_motor_commands.
                # Untuk saat ini, biarkan dulu agar fungsi recovery bisa mandiri.
                cmd = f"{pwm_kiri},{pwm_kanan}\n"
                ser.write(cmd.encode())
                ser.flush()
            except Exception as e:
                print(f"[SERIAL ERROR in Recovery] {e}")

def setup_fuzzy_logic():
    # PERUBAHAN: Range error diperlebar signifikan
    # Asumsi lebar frame 640. Error bisa mencapai sekitar -320 hingga 320.
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error')
    # PERUBAHAN: Range delta juga diperlebar
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # PERUBAHAN: Membership functions disesuaikan dengan rentang baru
    # Perhatikan overlap dan penempatan puncak.
    error['NL'] = fuzz.trimf(error.universe, [-350, -250, -80]) # Lebih lebar
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, -10]) # Lebih lebar
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])   # Zona zero sedikit dipersempit untuk lebih responsif
    error['PS'] = fuzz.trimf(error.universe, [10, 40, 100]) # Lebih lebar
    error['PL'] = fuzz.trimf(error.universe, [80, 250, 350]) # Lebih lebar

    delta['NL'] = fuzz.trimf(delta.universe, [-150, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-50, -20, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 20, 50])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 150])

    # Output dengan zona zero yang dominan (ini sudah cukup baik)
    output['L']  = fuzz.trimf(output.universe, [-100, -70, -40])
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -8])
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 50])
    output['R']  = fuzz.trimf(output.universe, [40, 70, 100])

    # PERUBAHAN: Rules sedikit dimodifikasi untuk lebih responsif di tikungan
    # Fokus pada NS/PS dan perubahan delta untuk memberikan koreksi lebih awal.
    rules = [
        # Error Negative Large (Garis jauh di kanan)
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), # Koreksi keras ke kiri
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']), # Koreksi ke kiri sedang
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),  # Tetap koreksi ke kiri
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),  # Jika error besar tapi delta positif, garis kembali ke tengah
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),  # Jika error besar tapi delta sangat positif, garis kembali ke tengah

        # Error Negative Small (Garis sedikit di kanan)
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']), # PERUBAHAN: Dari Z ke LS (lebih responsif)
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']), # PERUBAHAN: Dari Z ke LS (lebih responsif)
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),   # Tetap Z
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   # Tetap Z
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),   # Tetap Z

        # Error Zero (Garis di tengah) - SEMUA KE ZERO untuk jalan lurus stabil
        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

        # Error Positive Small (Garis sedikit di kiri)
        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),   # Tetap Z
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   # Tetap Z
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),   # Tetap Z
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']), # PERUBAHAN: Dari Z ke RS (lebih responsif)
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']), # PERUBAHAN: Dari Z ke RS (lebih responsif)

        # Error Positive Large (Garis jauh di kiri)
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),   # Jika error besar tapi delta negatif, garis kembali ke tengah
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),   # Jika error besar tapi delta sangat negatif, garis kembali ke tengah
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),  # Tetap koreksi ke kanan
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']), # Koreksi ke kanan sedang
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),  # Koreksi keras ke kanan
    ]

    control_system = ctrl.ControlSystem(rules)
    
    # if you want to visualize the membership functions:
    # error.view()
    # delta.view()
    # output.view()
    # plt.show()
    
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    # Resolusi default 640x480 seringkali cukup baik.
    # Jika perlu detail lebih, bisa naikkan, tapi akan mempengaruhi framerate.
    config = picam2.create_still_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    picam2.start()
    
    # CATATAN: Untuk kondisi pencahayaan yang sangat bervariasi (pagi/siang/sore)
    # Anda bisa coba mengontrol exposure atau gain secara manual jika auto tidak optimal.
    # picam2.set_controls({"ExposureTime": 8000}) # Contoh: 8000 mikrodetik (8ms)
    # picam2.set_controls({"AnalogueGain": 1.0}) # Contoh: 1.0 (minimum gain)
    # picam2.set_controls({"Brightness": 0.0}) # Contoh: 0.0 (default)
    # picam2.set_controls({"AeEnable": False}) # Nonaktifkan auto-exposure jika ingin manual
    
    return picam2

def setup_serial():
    try:
        # Menggunakan Serial0/ttyS0 untuk Pi 4, ttyAMA0 untuk Pi 3/Zero
        # '/dev/ttyS0' adalah port serial yang umum untuk Raspberry Pi 4
        # Jika Anda mengaktifkan UART melalui raspi-config dan menghubungkannya
        # ke GPIO 14/15, ttyAMA0 juga bisa digunakan. Konfirmasi port yang benar!
        ser = serial.Serial('/dev/ttyS0', 115200, timeout=1) 
        print("[UART] Port serial berhasil dibuka")
        return ser
    except serial.SerialException as e: # Tangkap exception serial secara spesifik
        print(f"[UART ERROR] Gagal membuka serial port: {e}. Pastikan kabel terhubung dan port benar.")
        print("Coba cek: 'ls /dev/tty*' dan 'sudo raspi-config' -> Interface Options -> Serial Port.")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adaptive histogram equalization untuk lighting yang tidak merata
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Multiple blur untuk mengurangi noise
    # Ukuran kernel bisa disesuaikan, (7,7) dan 5 sudah cukup umum
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 5)
    
    # OTSU thresholding - otomatis menentukan threshold optimal
    # Sangat baik untuk berbagai kondisi pencahayaan
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    height, width = binary.shape
    # PERUBAHAN: ROI lebih lebar dan lebih jauh (mulai lebih tinggi)
    # Ini krusial untuk mendeteksi tikungan lebih awal.
    roi_start = int(height * 0.3)  # Mulai dari 30% tinggi frame (sesuaikan jika perlu)
    roi_end = int(height * 0.95)   # Akhir di 95% tinggi frame (hampir bawah)
    roi_left = int(width * 0.05)   # 5% margin kiri
    roi_right = int(width * 0.95)  # 5% margin kanan
    
    roi = binary[roi_start:roi_end, roi_left:roi_right]
    
    # Return semua parameter ROI untuk visualisasi dan perhitungan
    return gray, binary, roi, roi_start, roi_left, roi_end, roi_right

def calculate_line_position(roi, roi_start, roi_left, frame_width):
    # Morphological operations yang lebih agresif
    kernel = np.ones((5,5), np.uint8) # Kernel (5,5) sudah cukup baik
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    # Cari kontur untuk deteksi garis yang lebih akurat
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Cari kontur terbesar (kemungkinan garis)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # PERUBAHAN: Threshold area minimum disesuaikan.
        # Dengan garis 17mm, area 200 mungkin terlalu rendah jika ada noise.
        # Coba 300 atau 400. Jika garis tidak terdeteksi, turunkan.
        # Jika noise terdeteksi sebagai garis, naikkan.
        if cv2.contourArea(largest_contour) > 300: # Eksperimen dengan nilai ini!
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                # cx dihitung relatif terhadap frame asli
                cx = int(M['m10'] / M['m00']) + roi_left # Kompensasi margin kiri ROI
                cy = int(M['m01'] / M['m00']) + roi_start # Kompensasi start Y ROI
                return True, cx, cy, largest_contour
    
    return False, 0, 0, None

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # PERUBAHAN: clip sesuai dengan rentang Antecedent baru
        fuzzy_ctrl.input['error'] = np.clip(error_val, -350, 350)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -100, 100)
    except Exception as e:
        # Ini akan muncul jika nilai input masih di luar range meski sudah di-clip,
        # atau ada masalah lain di fuzzy logic engine.
        print(f"[FLC ERROR] {e}. Error_val: {error_val}, Delta_error: {delta_error}")
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=45, scaling_factor=0.08):
    # Dead zone diperlebar untuk stabilitas di garis lurus (sudah bagus)
    if abs(kontrol) < 15: # Zona mati lebih besar (bisa 10-20)
        kontrol_scaled = 0 # PWM kiri = kanan (jalan lurus)
    else:
        # Kontrol yang lebih halus
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Range PWM yang lebih kecil dan aman (sudah bagus)
    # Sesuaikan base_pwm (kecepatan dasar) jika robot terlalu cepat/lambat
    # Sesuaikan range (25, 65) jika motor Anda bisa beroperasi lebih luas/sempit
    pwm_kiri = max(25, min(65, pwm_kiri))
    pwm_kanan = max(25, min(65, pwm_kanan))

    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL SEND ERROR] {e}")

def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter() # Menggunakan default window_size dan alpha yang sudah diubah
    line_recovery = LineRecovery()

    prev_error = 0
    frame_count = 0
    
    # Stabilization period
    print("[INFO] Warming up camera...")
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x = width // 2
            
            # PERUBAHAN: Tangkap semua nilai ROI yang di-return
            gray, binary, roi, roi_start, roi_left, roi_end, roi_right = process_image(frame)
            # PERUBAHAN: Lewatkan semua nilai ROI yang relevan ke calculate_line_position
            line_detected, cx, cy, contour = calculate_line_position(roi, roi_start, roi_left, width)

            if line_detected:
                line_recovery.line_found(cx - center_x)
                
                error = cx - center_x
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    status = "STRAIGHT" if abs(error) < 15 else "TURNING"
                    print(f"[{status}] Err:{error:3d}, ΔErr:{delta_error:3d}, FLC:{kontrol:5.1f}, PWM: L{pwm_kiri} R{pwm_kanan}")
                    
            else:
                recovery_action = line_recovery.handle_line_lost(ser)
                # CATATAN: Motor commands dari recovery class akan langsung dikirim di sini.
                # Ini memastikan robot bergerak meskipun garis hilang.
                if frame_count % 15 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, count: {line_recovery.lost_count}")
                
                # Saat garis hilang, reset error dan delta_error untuk menghindari lonjakan besar
                prev_error = 0 # Atau last_valid_error dari line_recovery
                
            # Visualisasi yang lebih informatif
            frame_with_line = frame.copy()
            
            # Center line
            cv2.line(frame_with_line, (center_x, 0), (center_x, height), (0, 255, 0), 2)
            
            # ROI area (PERUBAHAN: Menggunakan semua nilai ROI yang baru)
            roi_color = (255, 255, 0)
            cv2.rectangle(frame_with_line, 
                          (roi_left, roi_start), 
                          (roi_right, roi_end), 
                          roi_color, 2)
            
            if line_detected:
                # Line position
                cv2.circle(frame_with_line, (cx, cy), 8, (0, 0, 255), -1)
                
                # Draw contour
                if contour is not None:
                    # Adjust contour coordinates (PERUBAHAN: Menggunakan roi_left yang baru)
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += roi_left
                    adjusted_contour[:, :, 1] += roi_start
                    cv2.drawContours(frame_with_line, [adjusted_contour], -1, (255, 0, 255), 2)
                
                # Error display
                error_display = cx - center_x # Ini adalah error non-filter untuk display
                cv2.putText(frame_with_line, f"Error: {error_display}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_line, f"Status: TRACKING", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_with_line, f"Status: LINE LOST ({line_recovery.lost_count})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display images
            cv2.imshow("Line Follower", cv2.resize(frame_with_line, (640, 480)))
            cv2.imshow("Binary", cv2.resize(binary, (320, 240)))
            cv2.imshow("ROI", cv2.resize(roi, (320, 100))) # Ubah ukuran ROI display jika ROI Anda lebih tinggi
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.02) # PERUBAHAN: Sedikit lebih cepat (50 FPS)
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti
        if ser:
            ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
