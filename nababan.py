from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# import matplotlib.pyplot as plt # Uncomment if you want to visualize FLC membership functions (requires 'sudo pip install matplotlib')

# --- Konfigurasi Global ---
# Sesuaikan port serial jika diperlukan. Umumnya '/dev/ttyS0' untuk RPi 4, atau '/dev/ttyAMA0'
# Pastikan sudah diaktifkan di raspi-config (Interface Options -> P6 Serial Port -> No for login shell, Yes for hardware serial)
SERIAL_PORT = '/dev/ttyS0' 
BAUD_RATE = 115200

# Kecepatan dasar robot saat lurus dan scaling factor untuk belokan
BASE_PWM = 45 # Kecepatan maju dasar (0-100)
SCALING_FACTOR = 0.08 # Mengatur seberapa kuat FLC mempengaruhi belokan (eksperimen: 0.06 - 0.12)

# Batasan PWM untuk motor agar aman dan efektif (0-100)
MIN_PWM_OUTPUT = 25
MAX_PWM_OUTPUT = 65

# Ukuran Dead Zone FLC: dalam piksel. Error di bawah nilai ini dianggap nol.
# Membantu stabilitas di garis lurus.
FLC_DEAD_ZONE_ERROR = 15 

# --- Kelas untuk Filter Error ---
class ErrorFilter:
    def __init__(self, window_size=3, alpha=0.7):
        # window_size: Jumlah data error yang disimpan untuk median filter. Lebih kecil = lebih responsif.
        # alpha: Faktor smoothing eksponensial. Lebih tinggi = lebih responsif (lebih memprioritaskan data terbaru).
        self.window_size = window_size
        self.error_history = []
        self.alpha = alpha
        
    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        # Median filter untuk mengurangi noise (mengambil nilai tengah dari data history)
        sorted_errors = sorted(self.error_history)
        median_error = sorted_errors[len(sorted_errors)//2]
        
        # Exponential smoothing
        if len(self.error_history) > 1:
            # Menggunakan median error sebelumnya untuk smoothing
            prev_median_history = self.error_history[:-1] # History tanpa data terbaru
            prev_median = sorted(prev_median_history)[len(prev_median_history)//2] if len(prev_median_history) > 0 else median_error
            smoothed_error = self.alpha * median_error + (1 - self.alpha) * prev_median
        else:
            smoothed_error = median_error
        
        return int(smoothed_error)

# --- Kelas untuk Pemulihan Garis ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang
        self.last_valid_error = 0 # Error terakhir saat garis masih terdeteksi
        self.recovery_speed = 40 # Kecepatan PWM saat mencari garis

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # Tahap 1: Diam sebentar
        if self.lost_count < 10: # Contoh: diam selama 10 frame
            send_motor_commands(ser_instance, 0, 0)
            return "STOP"
        # Tahap 2: Mundur sedikit
        elif self.lost_count < 30: # Contoh: mundur selama 20 frame (dari frame ke-10 hingga ke-29)
            send_motor_commands(ser_instance, -30, -30) # Mundur dengan PWM -30
            return "REVERSE"
        # Tahap 3: Berputar mencari garis
        else:
            # Berputar ke arah terakhir garis terdeteksi (putar di tempat)
            if self.last_valid_error > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
                send_motor_commands(ser_instance, self.recovery_speed, -self.recovery_speed)
                return "SEARCH_RIGHT"
            else: # Garis terakhir di kiri, putar ke kiri (motor kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -self.recovery_speed, self.recovery_speed)
                return "SEARCH_LEFT"
    
    def line_found(self, current_error):
        self.lost_count = 0
        self.last_valid_error = current_error # Simpan error terakhir yang valid

# --- Setup Fuzzy Logic Control (FLC) ---
def setup_fuzzy_logic():
    # Antecedents (Input variables)
    # error: Posisi garis relatif terhadap pusat frame. Range -350 (jauh kiri) hingga 350 (jauh kanan).
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error')
    # delta: Perubahan error antar frame. Range -150 (berubah cepat ke kiri) hingga 150 (berubah cepat ke kanan).
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')

    # Consequent (Output variable)
    # output: Nilai kontrol yang akan digunakan untuk menghitung PWM. Range -100 (belok keras kiri) hingga 100 (belok keras kanan).
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

    # Membership Functions (Tuning untuk "mulus saat berbelok dan tetap mempertahankan titik tengah")
    # Error:
    error['NL'] = fuzz.trimf(error.universe, [-350, -200, -60]) # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])   # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])     # Zero (Zona Lurus yang stabil)
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])    # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 200, 350])    # Positive Large

    # Delta Error:
    delta['NL'] = fuzz.trimf(delta.universe, [-150, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -3])
    delta['Z']  = fuzz.trimf(delta.universe, [-5, 0, 5])       # Zona perubahan sangat kecil
    delta['PS'] = fuzz.trimf(delta.universe, [3, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 150])

    # Output:
    output['L']  = fuzz.trimf(output.universe, [-100, -70, -40]) # Full Left
    output['LS'] = fuzz.trimf(output.universe, [-50, -25, -8])  # Light-Medium Left
    output['Z']  = fuzz.trimf(output.universe, [-10, 0, 10])    # Straight (Zona lurus)
    output['RS'] = fuzz.trimf(output.universe, [8, 25, 50])    # Light-Medium Right
    output['R']  = fuzz.trimf(output.universe, [40, 70, 100])   # Full Right

    # Rule Base (Disempurnakan untuk belokan mulus dan stabilitas)
    rules = [
        # Error Negative Large (Garis di kiri jauh dari pusat)
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),  # Sangat jauh & menjauh -> Belok keras kiri
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),  # Sangat jauh & sedikit menjauh -> Belok keras kiri
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),  # Sangat jauh & stabil -> Belok kiri sedang
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),  # Sangat jauh & mendekat -> Luruskan (stabilkan)
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),  # Sangat jauh & cepat mendekat -> Luruskan (stabilkan)

        # Error Negative Small (Garis di kiri sedikit dari pusat)
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']), # Sedikit kiri & menjauh cepat -> Belok kiri sedang
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']), # Sedikit kiri & menjauh -> Belok kiri sedang
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),   # Sedikit kiri & stabil -> Luruskan (prioritas Z untuk stabilitas)
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   # Sedikit kiri & mendekat -> Luruskan
        ctrl.Rule(error['NS'] & delta['PL'], output['Z']),   # Sedikit kiri & cepat mendekat -> Luruskan

        # Error Zero (Garis di tengah)
        # Prioritas utama: Tetap lurus
        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),

        # Error Positive Small (Garis di kanan sedikit dari pusat)
        ctrl.Rule(error['PS'] & delta['NL'], output['Z']),   # Sedikit kanan & cepat mendekat -> Luruskan
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   # Sedikit kanan & mendekat -> Luruskan
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),   # Sedikit kanan & stabil -> Luruskan (prioritas Z untuk stabilitas)
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']), # Sedikit kanan & menjauh -> Belok kanan sedang
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']), # Sedikit kanan & menjauh cepat -> Belok kanan sedang

        # Error Positive Large (Garis di kanan jauh dari pusat)
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),   # Sangat jauh & cepat mendekat -> Luruskan (stabilkan)
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),   # Sangat jauh & mendekat -> Luruskan (stabilkan)
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),  # Sangat jauh & stabil -> Belok kanan sedang
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),  # Sangat jauh & menjauh -> Belok keras kanan
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),  # Sangat jauh & menjauh cepat -> Belok keras kanan
    ]

    control_system = ctrl.ControlSystem(rules)
    
    # Uncomment baris di bawah jika ingin memvisualisasikan fungsi keanggotaan.
    # error.view()
    # delta.view()
    # output.view()
    # plt.show()
    
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera Raspberry Pi ---
def setup_camera():
    picam2 = Picamera2()
    # Resolusi default 640x480 adalah titik awal yang baik untuk performa dan detail.
    # Jika perlu lebih banyak detail (misal garis sangat jauh), bisa naikkan resolusi,
    # tetapi akan meningkatkan beban CPU dan mengurangi framerate.
    config = picam2.create_still_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    picam2.start()
    
    # CATATAN: Untuk kondisi pencahayaan yang sangat bervariasi (pagi/siang/sore),
    # Anda bisa coba mengontrol exposure, gain, atau brightness secara manual
    # jika auto-exposure bawaan Picamera2 tidak memberikan hasil optimal.
    # Misalnya untuk mengurangi overexposure di siang hari atau saat lampu terlalu terang.
    # picam2.set_controls({"ExposureTime": 8000}) # Contoh: 8000 mikrodetik (8ms)
    # picam2.set_controls({"AnalogueGain": 1.0}) # Contoh: 1.0 (minimum gain)
    # picam2.set_controls({"Brightness": 0.0}) # Contoh: 0.0 (default)
    # picam2.set_controls({"AeEnable": False}) # Nonaktifkan auto-exposure jika ingin manual

    return picam2

# --- Setup Komunikasi Serial dengan ESP32 ---
def setup_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
        print(f"[UART] Port serial {SERIAL_PORT} berhasil dibuka.")
        return ser
    except serial.SerialException as e:
        print(f"[UART ERROR] Gagal membuka serial port {SERIAL_PORT}: {e}")
        print("Pastikan ESP32 terhubung, port serial benar, dan tidak ada program lain yang menggunakannya.")
        print("Coba cek: 'ls /dev/tty*' dan 'sudo raspi-config' -> Interface Options -> P6 Serial Port.")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] Gagal membuka serial port: {e}")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame):
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adaptive histogram equalization untuk meningkatkan kontras di area lokal
    # Berguna untuk kondisi pencahayaan yang tidak merata atau kurang optimal.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Multiple blur untuk mengurangi noise
    # Gaussian Blur: menghaluskan gambar, mengurangi detail tinggi
    # Median Blur: efektif menghilangkan noise salt-and-pepper tanpa merusak tepi
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 5)
    
    # OTSU Thresholding: Secara otomatis menentukan nilai ambang biner yang optimal
    # Sangat baik untuk beradaptasi dengan berbagai kondisi pencahayaan (pagi/siang/sore).
    # THRESH_BINARY_INV: Invers biner (objek gelap menjadi putih, latar terang menjadi hitam)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    height, width = binary.shape
    
    # Region of Interest (ROI)
    # Disetel untuk kamera yang tidak terlalu tinggi di atas meja (garis 17mm)
    # Memastikan robot "melihat" cukup jauh ke depan dan juga ke samping untuk tikungan.
    roi_start = int(height * 0.3)  # Mulai dari 30% dari atas frame (sesuaikan jika perlu)
    roi_end = int(height * 0.95)   # Berakhir di 95% dari tinggi frame (hampir bawah)
    roi_left = int(width * 0.05)   # Margin 5% dari kiri
    roi_right = int(width * 0.95)  # Margin 5% dari kanan
    
    # Pastikan ROI tidak keluar dari batas gambar
    roi_start = max(0, min(roi_start, height - 1))
    roi_end = max(0, min(roi_end, height))
    roi_left = max(0, min(roi_left, width - 1))
    roi_right = max(0, min(roi_right, width))

    roi = binary[roi_start:roi_end, roi_left:roi_right]
    
    # Mengembalikan semua data yang diperlukan untuk perhitungan dan visualisasi
    return gray, binary, roi, roi_start, roi_left, roi_end, roi_right

# --- Menghitung Posisi Garis ---
def calculate_line_position(roi_image, roi_start_y, roi_start_x, frame_width):
    # Operasi morfologi untuk membersihkan ROI dan menghilangkan noise
    # Morfologi Close: Menutup celah kecil pada objek (garis)
    # Morfologi Open: Menghilangkan objek kecil (noise)
    kernel = np.ones((5,5), np.uint8) 
    roi_clean = cv2.morphologyEx(roi_image, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    # Mencari kontur objek putih (garis) di ROI
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Mencari kontur dengan area terbesar, diasumsikan sebagai garis utama
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Threshold area minimum untuk memastikan kontur yang terdeteksi benar-benar garis
        # Dengan garis 17mm, nilai ini perlu disesuaikan eksperimental.
        # Jika robot tidak mendeteksi garis, turunkan nilai ini.
        # Jika noise terdeteksi sebagai garis, naikkan nilai ini.
        if cv2.contourArea(largest_contour) > 300: # Coba 300; bisa 200-500
            M = cv2.moments(largest_contour)
            if M['m00'] > 0: # Pastikan momen massa tidak nol untuk menghindari division by zero
                # cx: pusat X kontur (ditambah offset ROI untuk mendapatkan posisi di frame asli)
                cx = int(M['m10'] / M['m00']) + roi_start_x
                # cy: pusat Y kontur (ditambah offset ROI untuk mendapatkan posisi di frame asli)
                cy = int(M['m01'] / M['m00']) + roi_start_y
                return True, cx, cy, largest_contour # Garis terdeteksi, posisi, dan kontur
    
    return False, 0, 0, None # Garis tidak terdeteksi

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl_system, error_value, delta_error_value):
    try:
        # Memastikan nilai input berada dalam rentang Antecedent FLC untuk menghindari error
        fuzzy_ctrl_system.input['error'] = np.clip(error_value, -350, 350)
        fuzzy_ctrl_system.input['delta'] = np.clip(delta_error_value, -150, 150)
        fuzzy_ctrl_system.compute()
        # Mengembalikan output yang sudah di-clip ke rentang -100 hingga 100
        return np.clip(fuzzy_ctrl_system.output['output'], -100, 100)
    except Exception as e:
        print(f"[FLC ERROR] {e}. Input Error: {error_value}, Delta Error: {delta_error_value}")
        return 0.0 # Kembali ke 0 jika ada error

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(control_output, base_pwm, scaling_factor):
    # Menerapkan dead zone untuk stabilitas di garis lurus
    # Jika output kontrol sangat kecil, anggap robot harus lurus.
    if abs(control_output) < FLC_DEAD_ZONE_ERROR: 
        control_scaled = 0 # Tidak ada koreksi, PWM kiri = kanan
    else:
        control_scaled = control_output * scaling_factor

    pwm_kiri = base_pwm + control_scaled
    pwm_kanan = base_pwm - control_scaled

    # Membatasi nilai PWM agar dalam rentang yang aman dan efektif untuk motor
    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))

    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser_instance, pwm_kiri, pwm_kanan):
    if ser_instance and ser_instance.is_open: # Pastikan port serial terbuka
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser_instance.write(cmd.encode())
            ser_instance.flush() # Pastikan semua data terkirim
        except serial.SerialException as e:
            print(f"[SERIAL SEND ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL SEND ERROR] Error tidak dikenal saat mengirim: {e}")
    elif not ser_instance:
        # print("[SERIAL WARNING] Serial port tidak terinisialisasi.")
        pass # Biarkan ini kosong jika tidak ingin spam di konsol

# --- Fungsi Utama Program ---
def main():
    fuzzy_controller = setup_fuzzy_logic()
    camera = setup_camera()
    serial_port = setup_serial() # Simpan objek serial port
    error_filter = ErrorFilter()
    line_recovery_handler = LineRecovery()

    prev_filtered_error = 0 # Error dari frame sebelumnya (setelah difilter)
    frame_counter = 0
    
    print("[INFO] Memulai robot Line Follower...")
    print("[INFO] Menunggu kamera stabil...")
    time.sleep(2) # Beri waktu kamera untuk melakukan auto-exposure

    try:
        while True:
            # Ambil frame dari kamera
            frame = camera.capture_array()
            height, width = frame.shape[:2]
            center_x = width // 2 # Titik tengah horizontal frame

            # Proses gambar (grayscale, blur, thresholding, ROI)
            gray_frame, binary_frame, roi_frame, roi_y_start, roi_x_start, roi_y_end, roi_x_end = process_image(frame)
            
            # Hitung posisi garis di dalam ROI
            line_found, line_cx, line_cy, line_contour = calculate_line_position(roi_frame, roi_y_start, roi_x_start, width)

            current_pwm_kiri = 0
            current_pwm_kanan = 0

            if line_found:
                line_recovery_handler.line_found(line_cx - center_x) # Reset recovery state
                
                # Hitung error (penyimpangan dari pusat)
                current_error = line_cx - center_x
                
                # Filter error untuk menghaluskan dan mengurangi noise
                filtered_error = error_filter.filter_error(current_error)
                
                # Hitung delta error (perubahan error antar frame)
                delta_error = filtered_error - prev_filtered_error
                prev_filtered_error = filtered_error # Update error sebelumnya untuk frame berikutnya

                # Hitung output kontrol dari FLC
                control_output = compute_fuzzy_control(fuzzy_controller, filtered_error, delta_error)
                
                # Hitung PWM motor berdasarkan output FLC
                current_pwm_kiri, current_pwm_kanan = calculate_motor_pwm(control_output, BASE_PWM, SCALING_FACTOR)
                
                # Kirim perintah PWM ke ESP32
                send_motor_commands(serial_port, current_pwm_kiri, current_pwm_kanan)

                # Output debug ke konsol setiap 10 frame
                if frame_counter % 10 == 0:
                    status_text = "STRAIGHT" if abs(filtered_error) < FLC_DEAD_ZONE_ERROR else "TURNING"
                    print(f"[{status_text}] Err:{filtered_error:3d}, ΔErr:{delta_error:3d}, FLC:{control_output:5.1f}, PWM: L{current_pwm_kiri} R{current_pwm_kanan}")
                    
            else: # Garis tidak terdeteksi
                # Tangani situasi garis hilang
                recovery_action = line_recovery_handler.handle_line_lost(serial_port)
                # Saat garis hilang, reset prev_filtered_error untuk menghindari lonjakan besar saat garis ditemukan lagi
                prev_filtered_error = 0 
                if frame_counter % 15 == 0:
                    print(f"[LOST] Garis hilang, aksi: {recovery_action}, lost_count: {line_recovery_handler.lost_count}")
            
            # --- Visualisasi (untuk debugging dan monitoring) ---
            frame_display = frame.copy()
            
            # Gambar garis tengah frame (hijau)
            cv2.line(frame_display, (center_x, 0), (center_x, height), (0, 255, 0), 2)
            
            # Gambar kotak ROI (cyan)
            roi_color = (255, 255, 0) # Cyan
            cv2.rectangle(frame_display, 
                          (roi_x_start, roi_y_start), 
                          (roi_x_end, roi_y_end), 
                          roi_color, 2)
            
            if line_found:
                # Gambar lingkaran di pusat garis yang terdeteksi (merah)
                cv2.circle(frame_display, (line_cx, line_cy), 8, (0, 0, 255), -1)
                
                # Gambar kontur garis (magenta)
                if line_contour is not None:
                    # Sesuaikan koordinat kontur dari ROI ke frame asli
                    adjusted_contour = line_contour.copy()
                    adjusted_contour[:, :, 0] += roi_x_start
                    adjusted_contour[:, :, 1] += roi_y_start
                    cv2.drawContours(frame_display, [adjusted_contour], -1, (255, 0, 255), 2)
                
                # Tampilkan teks informasi error dan status
                cv2.putText(frame_display, f"Error: {current_error}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Filtered Err: {filtered_error}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame_display, f"Status: TRACKING", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_display, f"Status: LINE LOST ({line_recovery_handler.lost_count})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Tampilkan jendela OpenCV
            # Resolusi display bisa disesuaikan agar tidak terlalu besar di layar kecil
            cv2.imshow("Line Follower Live", cv2.resize(frame_display, (640, 480)))
            cv2.imshow("Binary Processed ROI", cv2.resize(binary_frame, (320, 240)))
            cv2.imshow("Cleaned ROI for Contour", cv2.resize(roi_frame, (320, 100))) # Tampilkan hanya ROI yang relevan

            # Keluar dari loop jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_counter += 1
            time.sleep(0.02) # Kontrol framerate (sekitar 50 FPS). Sesuaikan jika robot terlalu cepat/lambat atau CPU tinggi.
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna (KeyboardInterrupt).")
    except Exception as e:
        print(f"[CRITICAL ERROR] Terjadi error tak terduga: {e}")
    finally:
        print("[INFO] Menghentikan motor dan membersihkan sumber daya.")
        send_motor_commands(serial_port, 0, 0) # Pastikan motor berhenti total
        if serial_port and serial_port.is_open:
            serial_port.close() # Tutup port serial
        camera.stop() # Hentikan kamera
        cv2.destroyAllWindows() # Tutup semua jendela OpenCV
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
