from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
manual_threshold_value = 100 # Nilai default awal, sesuaikan setelah tuning pertama

# --- Callback Function untuk Trackbar (Slider) ---
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        if not self.error_history: # Menghindari pembagian dengan nol jika history kosong
            return 0
        return int(sum(self.error_history) / len(self.error_history))

# --- Kelas untuk Pemulihan Garis (Diperbarui dengan PWM yang Disesuaikan) ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang
        self.last_valid_error_direction = 0 # Menyimpan arah error terakhir (+:kanan, -:kiri)
        # Tidak perlu menyimpan PWM terakhir, cukup arah error.

        # --- PENYESUAIAN PWM UNTUK RECOVERY ---
        self.search_speed = 40          # Kecepatan PWM saat berputar mencari garis (sesuai min PWM baru Anda)
        self.hold_direction_factor = 0.5 # Faktor untuk seberapa agresif belok saat hold direction
        self.hold_speed_base_pwm = 50   # Kecepatan dasar saat hold direction (maju) - SESUAI PERMINTAAN ANDA

        # Batas PWM baru untuk recovery
        self.min_pwm_recovery = 40      # SESUAI PERMINTAAN ANDA
        self.max_pwm_recovery = 70      # SESUAI PERMINTAAN ANDA
        # --- AKHIR PENYESUAIAN ---

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # --- Strategi Prediksi/Hold Last Direction ---
        # Tahap 1: Lanjutkan dengan arah terakhir yang diketahui (maju sambil belok)
        if self.lost_count < 15: # Coba maju terus selama 15 frame sambil belok
            if self.last_valid_error_direction > 0: # Terakhir belok kanan/garis di kanan
                pwm_kiri_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
            elif self.last_valid_error_direction < 0: # Terakhir belok kiri/garis di kiri
                pwm_kiri_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
            else: # Terakhir lurus, coba lurus terus
                pwm_kiri_rec = self.hold_speed_base_pwm
                pwm_kanan_rec = self.hold_speed_base_pwm
            
            # Pastikan PWM tidak keluar batas saat recovery menggunakan nilai baru
            pwm_kiri_rec = max(self.min_pwm_recovery, min(self.max_pwm_recovery, int(pwm_kiri_rec)))
            pwm_kanan_rec = max(self.min_pwm_recovery, min(self.max_pwm_recovery, int(pwm_kanan_rec)))

            send_motor_commands(ser_instance, pwm_kiri_rec, pwm_kanan_rec)
            return f"HOLD_DIR ({pwm_kiri_rec},{pwm_kanan_rec})"
        
        # Tahap 2: Jika masih hilang, beralih ke strategi putar di tempat (fallback)
        else:
            if self.last_valid_error_direction > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
                send_motor_commands(ser_instance, self.search_speed, -self.search_speed)
                return "SEARCH_RIGHT_SPIN"
            else: # Garis terakhir di kiri atau lurus, putar ke kiri (motor kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed)
                return "SEARCH_LEFT_SPIN" # Default putar kiri jika hilang dari lurus
        
    def line_found(self, current_error):
        self.lost_count = 0
        # Simpan arah error terakhir yang signifikan
        if abs(current_error) > 5: # Anggap error signifikan jika lebih dari 5 piksel
            self.last_valid_error_direction = current_error 
        else: # Jika error sangat kecil, anggap arahnya lurus
            self.last_valid_error_direction = 0 
        # Tidak perlu menyimpan last_valid_pwm_kiri/kanan di sini, cukup arah error.

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# PASTIKAN FUNGSI-FUNGSI BERIKUT TELAH ANDA DEFINISIKAN DENGAN BENAR
# SESUAI DENGAN KEBUTUHAN DAN SETUP ROBOT ANDA.
# Ini adalah placeholder berdasarkan komentar di kode awal Anda.
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def setup_fuzzy_logic():
    """
    Menginisialisasi dan mengkonfigurasi kontroler logika fuzzy.
    Termasuk pendefinisian Antecedents (error, delta_error) dan Consequent (kontrol),
    serta aturan fuzzy.
    """
    # Contoh Implementasi (harap sesuaikan dengan milik Anda)
    error_val = ctrl.Antecedent(np.arange(-160, 161, 1), 'error') # Sesuaikan rentang error
    delta_error_val = ctrl.Antecedent(np.arange(-50, 51, 1), 'delta_error') # Sesuaikan rentang delta_error
    kontrol_val = ctrl.Consequent(np.arange(-100, 101, 1), 'kontrol') # Sesuaikan rentang output kontrol

    # Membership functions untuk error (contoh sederhana)
    error_val['kiri_jauh'] = fuzz.trimf(error_val.universe, [-160, -160, -80])
    error_val['kiri'] = fuzz.trimf(error_val.universe, [-120, -60, 0])
    error_val['tengah'] = fuzz.trimf(error_val.universe, [-30, 0, 30]) # Diperkecil untuk sensitivitas
    error_val['kanan'] = fuzz.trimf(error_val.universe, [0, 60, 120])
    error_val['kanan_jauh'] = fuzz.trimf(error_val.universe, [80, 160, 160])

    # Membership functions untuk delta_error (contoh sederhana)
    delta_error_val['negatif'] = fuzz.trimf(delta_error_val.universe, [-50, -50, 0])
    delta_error_val['nol'] = fuzz.trimf(delta_error_val.universe, [-10, 0, 10])
    delta_error_val['positif'] = fuzz.trimf(delta_error_val.universe, [0, 50, 50])

    # Membership functions untuk kontrol (contoh sederhana)
    kontrol_val['belok_kiri_kuat'] = fuzz.trimf(kontrol_val.universe, [-100, -100, -50])
    kontrol_val['belok_kiri'] = fuzz.trimf(kontrol_val.universe, [-80, -40, 0])
    kontrol_val['lurus'] = fuzz.trimf(kontrol_val.universe, [-20, 0, 20]) # Diperkecil
    kontrol_val['belok_kanan'] = fuzz.trimf(kontrol_val.universe, [0, 40, 80])
    kontrol_val['belok_kanan_kuat'] = fuzz.trimf(kontrol_val.universe, [50, 100, 100])
    
    # Aturan Fuzzy (contoh, perlu banyak aturan untuk performa baik)
    rule1 = ctrl.Rule(error_val['kiri_jauh'], kontrol_val['belok_kanan_kuat'])
    rule2 = ctrl.Rule(error_val['kanan_jauh'], kontrol_val['belok_kiri_kuat'])
    rule3 = ctrl.Rule(error_val['kiri'], kontrol_val['belok_kanan'])
    rule4 = ctrl.Rule(error_val['kanan'], kontrol_val['belok_kiri'])
    rule5 = ctrl.Rule(error_val['tengah'] & delta_error_val['nol'], kontrol_val['lurus'])
    # Tambahkan lebih banyak aturan di sini, terutama yang melibatkan delta_error
    # Contoh:
    # rule6 = ctrl.Rule(error_val['tengah'] & delta_error_val['positif'], kontrol_val['belok_kiri']) # Jika error tengah tapi cenderung ke kanan
    # rule7 = ctrl.Rule(error_val['tengah'] & delta_error_val['negatif'], kontrol_val['belok_kanan']) # Jika error tengah tapi cenderung ke kiri

    kontrol_fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5]) # Tambahkan rule6, rule7 jika dibuat
    fuzzy_sim = ctrl.ControlSystemSimulation(kontrol_fuzzy_ctrl)
    print("[INFO] Fuzzy logic system initialized.")
    return fuzzy_sim

def setup_camera():
    """Menginisialisasi dan mengkonfigurasi Picamera2."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"}) # Resolusi lebih kecil
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Waktu untuk kamera stabil
    print("[INFO] Camera initialized and started.")
    return picam2

def setup_serial(port='/dev/ttyUSB0', baudrate=9600):
    """Menginisialisasi koneksi serial ke mikrokontroler."""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2) # Waktu untuk koneksi serial stabil
        print(f"[INFO] Serial connection established on {port} at {baudrate} baud.")
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port {port}: {e}")
        print("[WARNING] Running without serial communication. Motor commands will be simulated.")
        return None # Mengembalikan None jika koneksi gagal

def process_image(frame, display_mode=False):
    """
    Memproses frame dari kamera: konversi ke grayscale, ROI, thresholding.
    Menggunakan manual_threshold_value dari global scope.
    """
    global manual_threshold_value # Akses nilai threshold manual global

    height, width = frame.shape[:2]

    # --- ROI Definition ---
    # ROI disesuaikan agar lebih fokus ke bagian bawah gambar
    # Anda mungkin perlu menyesuaikan nilai ini berdasarkan posisi kamera Anda
    roi_top_ratio = 0.65  # Mulai ROI dari 65% bagian atas frame
    roi_bottom_ratio = 0.95 # Akhiri ROI di 95% bagian bawah frame (menyisakan sedikit ruang di bawah)
    
    roi_start_y = int(height * roi_top_ratio)
    roi_end_y = int(height * roi_bottom_ratio)

    # Pastikan roi_start_y < roi_end_y
    if roi_start_y >= roi_end_y:
        print(f"[WARNING] ROI y-coordinates invalid: start_y={roi_start_y}, end_y={roi_end_y}. Using full height.")
        roi_start_y = 0
        roi_end_y = height

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Gaussian Blur untuk mengurangi noise sebelum thresholding
    blurred_gray = cv2.GaussianBlur(gray_full, (5, 5), 0)

    # Menggunakan nilai threshold dari trackbar
    _, binary_full = cv2.threshold(blurred_gray, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Ambil ROI dari gambar biner
    roi_binary = binary_full[roi_start_y:roi_end_y, :]

    if roi_binary.size == 0:
        print("[WARNING] ROI is empty. Check ROI parameters and frame dimensions.")
        return gray_full, binary_full, None, roi_start_y, roi_end_y # Kembalikan None jika ROI tidak valid

    return gray_full, binary_full, roi_binary, roi_start_y, roi_end_y


def calculate_line_position(roi_binary_image, roi_offset_y):
    """
    Menghitung posisi garis (centroid) pada ROI biner.
    Mengembalikan status deteksi, koordinat x, dan y (relatif terhadap frame asli).
    """
    if roi_binary_image is None or roi_binary_image.size == 0:
        return False, 0, 0

    M = cv2.moments(roi_binary_image)
    if M["m00"] != 0:
        # cx dihitung relatif terhadap lebar ROI (sama dengan lebar frame)
        cx = int(M["m10"] / M["m00"])
        # cy dihitung relatif terhadap tinggi ROI, lalu ditambahkan offset ROI
        cy_roi = int(M["m01"] / M["m00"])
        cy_frame = cy_roi + roi_offset_y # Koordinat y di frame asli
        return True, cx, cy_frame
    else:
        return False, 0, 0 # Garis tidak terdeteksi


def compute_fuzzy_control(fuzzy_simulation, error, delta_error):
    """Menghitung output kontrol menggunakan sistem fuzzy."""
    # Masukkan nilai error dan delta_error ke sistem fuzzy
    fuzzy_simulation.input['error'] = error
    fuzzy_simulation.input['delta_error'] = delta_error
    
    try:
        fuzzy_simulation.compute()
        kontrol = fuzzy_simulation.output['kontrol']
        return kontrol
    except Exception as e:
        # print(f"[FUZZY WARNING] Could not compute fuzzy output: {e}. Defaulting control.")
        # Default behavior jika fuzzy gagal (misalnya, karena nilai input di luar rentang yang didefinisikan ketat)
        # Anda bisa mengembalikan 0 atau nilai aman lainnya.
        # Atau, jika error adalah karena input di luar universe, clamp inputnya.
        # Contoh clamping sederhana (mungkin perlu disesuaikan):
        error_antecedent = fuzzy_simulation.ctrl.antecedents[0] # Asumsi 'error' adalah antecedent pertama
        delta_error_antecedent = fuzzy_simulation.ctrl.antecedents[1] # Asumsi 'delta_error' adalah antecedent kedua
        
        clamped_error = np.clip(error, error_antecedent.universe.min(), error_antecedent.universe.max())
        clamped_delta_error = np.clip(delta_error, delta_error_antecedent.universe.min(), delta_error_antecedent.universe.max())
        
        fuzzy_simulation.input['error'] = clamped_error
        fuzzy_simulation.input['delta_error'] = clamped_delta_error
        try:
            fuzzy_simulation.compute()
            kontrol = fuzzy_simulation.output['kontrol']
            # if frame_count % 60 == 0: # Cetak sesekali jika ada clamping
            #    print(f"[FUZZY NOTE] Input clamped. Original E:{error}, DE:{delta_error}. Clamped E:{clamped_error}, DE:{clamped_delta_error}. Output: {kontrol}")
            return kontrol
        except Exception as e2:
            # print(f"[FUZZY ERROR] Failed even after clamping: {e2}. Returning 0.")
            return 0 # Default jika masih gagal

BASE_SPEED_PWM = 50  # Kecepatan dasar motor (0-100 atau 0-255, tergantung driver Anda)
PWM_MIN = 40         # PWM minimal agar motor bergerak
PWM_MAX = 70         # PWM maksimal yang diizinkan

def calculate_motor_pwm(kontrol_output):
    """
    Menghitung nilai PWM untuk motor kiri dan kanan berdasarkan output kontrol fuzzy.
    """
    # 'kontrol_output' adalah nilai dari -100 (belok kiri kuat) hingga +100 (belok kanan kuat) dari fuzzy
    
    # Semakin besar absolut kontrol_output, semakin besar perbedaan kecepatan antar roda
    # Jika kontrol_output negatif, robot belok kiri (PWM kanan > PWM kiri)
    # Jika kontrol_output positif, robot belok kanan (PWM kiri > PWM kanan)

    # Normalisasi kontrol_output ke rentang yang lebih sesuai untuk diferensial PWM jika perlu
    # Misalnya, jika output FLC [-100, 100], kita bisa memetakannya ke perbedaan PWM
    # Misal, maks perbedaan PWM adalah BASE_SPEED_PWM itu sendiri atau sebagian darinya
    
    # Versi sederhana:
    # pwm_kiri = BASE_SPEED_PWM - kontrol_output
    # pwm_kanan = BASE_SPEED_PWM + kontrol_output
    
    # Versi yang lebih mempertimbangkan arah belok:
    # Jika kontrol_output > 0 (belok kanan), kurangi kecepatan roda kanan, tambah kecepatan roda kiri
    # Jika kontrol_output < 0 (belok kiri), kurangi kecepatan roda kiri, tambah kecepatan roda kanan
    
    # Mari gunakan pendekatan di mana output kontrol langsung memodifikasi kecepatan dasar
    # Jika kontrol_output positif (FLC mau belok KIRI robot, error di KANAN), maka PWM KIRI << PWM KANAN
    # Jika kontrol_output negatif (FLC mau belok KANAN robot, error di KIRI), maka PWM KIRI >> PWM KANAN
    # Perlu diingat: output fuzzy 'kontrol' yang saya contohkan:
    #   Negatif = belok kiri kuat/belok kiri (artinya error di kanan, jadi motor kanan lebih cepat)
    #   Positif = belok kanan kuat/belok kanan (artinya error di kiri, jadi motor kiri lebih cepat)
    #   Ini terbalik dengan asumsi umum PID di mana error positif (kanan) -> output positif.
    #   Mari kita konsisten dengan output fuzzy yang didefinisikan:
    #   'belok_kiri_kuat' = -100, 'belok_kanan_kuat' = 100

    # Jika output fuzzy adalah + (ingin belok kanan), maka pwm_kiri harus lebih cepat
    # Jika output fuzzy adalah - (ingin belok kiri), maka pwm_kanan harus lebih cepat
    
    # Skala kontrol output ke setengah dari base_speed agar tidak terlalu agresif
    # atau skala ke rentang PWM_MAX - PWM_MIN
    turn_effect = kontrol_output # Asumsi kontrol_output sudah diskalakan dengan baik oleh Fuzzy
                               # Jika tidak, Anda mungkin perlu: turn_effect = kontrol_output * FAKTOR_SKALA

    pwm_kiri = BASE_SPEED_PWM + turn_effect
    pwm_kanan = BASE_SPEED_PWM - turn_effect

    # Clamping PWM ke rentang yang diizinkan
    pwm_kiri = int(np.clip(pwm_kiri, PWM_MIN, PWM_MAX))
    pwm_kanan = int(np.clip(pwm_kanan, PWM_MIN, PWM_MAX))
    
    return pwm_kiri, pwm_kanan


def send_motor_commands(serial_conn, pwm_left, pwm_right):
    """Mengirim perintah PWM ke mikrokontroler melalui serial."""
    if serial_conn and serial_conn.is_open:
        command = f"M{int(pwm_left):d},{int(pwm_right):d}\n" # Format perintah: M[PWM_KIRI],[PWM_KANAN]
        try:
            serial_conn.write(command.encode())
            # print(f"Sent: {command.strip()}") # Untuk debug
        except Exception as e:
            print(f"[SERIAL ERROR] Failed to send command {command.strip()}: {e}")
    else:
        # Simulasikan jika tidak ada koneksi serial
        # print(f"[SIMULATE CMD] L:{pwm_left}, R:{pwm_right}")
        pass # Tidak melakukan apa-apa jika serial tidak tersedia

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AKHIR DARI BAGIAN FUNGSI YANG PERLU ANDA PASTIKAN/DEFINISIKAN
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value # Pastikan bisa diakses

    fuzzy_ctrl_simulation = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial() # Sesuaikan port jika perlu
    error_filter = ErrorFilter(window_size=3) 
    line_recovery_handler = LineRecovery() 

    prev_error = 0
    frame_count = 0

    DISPLAY_GUI = True # Set True untuk menampilkan GUI, False untuk headless

    if DISPLAY_GUI:
        cv2.namedWindow("Threshold ROI", cv2.WINDOW_NORMAL) # Agar bisa di-resize
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL) # Agar bisa di-resize
        # cv2.resizeWindow("Camera View", 480, 360) # Ukuran window awal
        # cv2.resizeWindow("Threshold ROI", 480, 360)


    try:
        while True:
            frame_start_time = time.time() # Untuk mengukur FPS
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2 

            if DISPLAY_GUI:
                # Baca nilai trackbar hanya jika GUI ditampilkan
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            # Proses gambar utama ada di sini
            gray_full, binary_full, roi_binary, roi_start_y, roi_end_y = process_image(frame, display_mode=DISPLAY_GUI)
            
            current_recovery_action = "N/A" # Default jika tidak ada recovery

            if roi_binary is None or roi_binary.size == 0: # Jika ROI tidak valid
                current_recovery_action = line_recovery_handler.handle_line_lost(ser)
                prev_error = 0 # Reset error
                if frame_count % 30 == 0: # Cetak pesan error sesekali
                    print(f"[DEBUG] Gagal memproses frame: ROI tidak valid. Aksi pemulihan: {current_recovery_action}")
                
                # Bagian visualisasi saat ROI error (opsional tapi berguna untuk debug)
                if DISPLAY_GUI:
                    frame_for_display = frame.copy()
                    # Gambar garis tengah, batas ROI, dll. seperti di bawah jika diperlukan
                    cv2.putText(frame_for_display, f"ROI ERROR. Action: {current_recovery_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.imshow("Camera View", frame_for_display)
                    # Tampilkan gambar biner kosong atau frame grayscale jika binary_full tidak ada
                    if binary_full is not None:
                         cv2.imshow("Threshold ROI", binary_full)
                    else: # Jika binary_full juga None (kasus ekstrim)
                         cv2.imshow("Threshold ROI", gray_full if gray_full is not None else np.zeros((100,100), dtype=np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_count += 1
                continue # Lanjut ke iterasi berikutnya jika ROI error

            # Hitung posisi garis jika ROI valid
            line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
            
            if line_detected:
                error = cx - center_x_frame 
                line_recovery_handler.line_found(error) # Beritahu handler bahwa garis ditemukan & simpan error
                
                filtered_error = error_filter.filter_error(error) # Gunakan error mentah atau yang difilter
                delta_error = filtered_error - prev_error
                prev_error = filtered_error

                kontrol = compute_fuzzy_control(fuzzy_ctrl_simulation, filtered_error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0: # Cetak debug sesekali
                    print(f"[DEBUG] Err:{filtered_error:4d}, ΔErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            
            else: # Garis tidak terdeteksi di ROI yang valid
                current_recovery_action = line_recovery_handler.handle_line_lost(ser)
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan
                if frame_count % 20 == 0: # Cetak debug sesekali
                    print(f"[DEBUG] Garis tidak terdeteksi. Aksi pemulihan: {current_recovery_action}")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Garis tengah acuan (hijau)
                cv2.line(frame_for_display, (center_x_frame, 0), (center_x_frame, height), (0, 255, 0), 1)
                
                # Gambar kotak ROI (biru)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                # Garis bantu indikasi zona tengah untuk FLC (kuning) - sesuaikan dengan FLC 'tengah'
                flc_error_z_boundary = 15 # Misalnya, jika FLC 'tengah' adalah +/- 15
                cv2.line(frame_for_display, (center_x_frame - flc_error_z_boundary, roi_start_y), (center_x_frame - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (center_x_frame + flc_error_z_boundary, roi_start_y), (center_x_frame + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 7, (0, 0, 255), -1) # Titik centroid (merah)
                    cv2.line(frame_for_display, (center_x_frame, cy), (cx, cy), (0,0,255),2) # Garis error
                    cv2.putText(frame_for_display, f"E:{filtered_error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"DE:{delta_error}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"FLC:{kontrol:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame_for_display, f"PWM L:{pwm_kiri} R:{pwm_kanan}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                else: # Tampilkan status pemulihan jika garis hilang
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"ACTION: {current_recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                fps = 1.0 / (time.time() - frame_start_time)
                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame_for_display, f"FPS: {fps:.1f}", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary) 
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---
            else: # Jika headless, tambahkan sedikit delay agar tidak membebani CPU sepenuhnya
                if not line_detected and line_recovery_handler.lost_count > 15 : # Jika sedang spin mencari garis
                     time.sleep(0.01) # Delay kecil saat spin
                # else:
                #    time.sleep(0.001) # Delay sangat kecil saat mengikuti garis

            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Menghentikan motor dan menutup koneksi...")
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti
        if ser and ser.is_open:
            ser.close()
            print("[INFO] Serial connection closed.")
        picam2.stop()
        print("[INFO] Camera stopped.")
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
            print("[INFO] OpenCV windows closed.")
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
