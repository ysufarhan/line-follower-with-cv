from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
# Ini akan diperbarui oleh slider OpenCV
manual_threshold_value = 100 # Nilai default awal, bisa disesuaikan

# --- Callback Function untuk Trackbar (Slider) ---
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Class untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=2): # Ukuran jendela yang lebih kecil untuk responsivitas
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Definisi Universe (Rentang Nilai Input/Output)
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan)
    # Dioptimalkan untuk stabilitas lurus dan respons belok akurat

    # ERROR: Seberapa jauh garis dari tengah kamera (pusat: 0)
    # NL: Negative Large, NS: Negative Small, Z: Zero, PS: Positive Small, PL: Positive Large
    error['NL'] = fuzz.trimf(error.universe, [-200, -150, -60])
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -5])
    error['Z']  = fuzz.trimf(error.universe, [-10, 0, 10])     # Sangat sempit untuk akurasi lurus
    error['PS'] = fuzz.trimf(error.universe, [5, 30, 80])
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 200])

    # DELTA: Perubahan error dari waktu sebelumnya (kecepatan error)
    delta['NL'] = fuzz.trimf(delta.universe, [-150, -100, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -2])
    delta['Z']  = fuzz.trimf(delta.universe, [-5, 0, 5])       # Sangat sempit untuk stabilitas
    delta['PS'] = fuzz.trimf(delta.universe, [2, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 100, 150])

    # OUTPUT: Nilai kontrol yang akan mengubah PWM motor (L: Left, LS: Left Small, Z: Zero, RS: Right Small, R: Right)
    output['L']  = fuzz.trimf(output.universe, [-150, -100, -50])
    output['LS'] = fuzz.trimf(output.universe, [-60, -25, -5])
    output['Z']  = fuzz.trimf(output.universe, [-2, 0, 2])       # Sangat sempit untuk PWM yang seimbang
    output['RS'] = fuzz.trimf(output.universe, [5, 25, 60])
    output['R']  = fuzz.trimf(output.universe, [50, 100, 150])

    # --- RULES (Aturan Fuzzy) ---
    # Mendefinisikan bagaimana input (error, delta) memengaruhi output kontrol
    rules = [
        # Ketika error NL (garis jauh di kiri robot)
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'],  output['LS']), # Jauh kiri, stabil -> belok kanan sedikit
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),  # Jauh kiri, bergerak ke tengah -> lurus
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),  # Jauh kiri, cepat ke tengah -> lurus

        # Ketika error NS (garis agak di kiri robot)
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']),  # Agak kiri, makin jauh kiri -> lurus
        ctrl.Rule(error['NS'] & delta['Z'],  output['Z']),  # Agak kiri, stabil -> lurus (target)
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['NS'] & delta['PL'], output['R']),

        # Ketika error Z (garis di tengah robot)
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'],  output['Z']),  # PENTING: Robot lurus sempurna!
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        # Ketika error PS (garis agak di kanan robot)
        ctrl.Rule(error['PS'] & delta['NL'], output['L']),
        ctrl.Rule(error['PS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['PS'] & delta['Z'],  output['Z']),  # Agak kanan, stabil -> lurus (target)
        ctrl.Rule(error['PS'] & delta['PS'], output['Z']),  # Agak kanan, makin jauh kanan -> lurus
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        # Ketika error PL (garis jauh di kanan robot)
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),  # Jauh kanan, bergerak ke tengah -> lurus
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),  # Jauh kanan, bergerak ke tengah -> lurus
        ctrl.Rule(error['PL'] & delta['Z'],  output['RS']), # Jauh kanan, stabil -> belok kiri sedikit
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera ---
def setup_camera():
    picam2 = Picamera2()
    # Resolusi rendah untuk performa di Raspberry Pi 4
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

# --- Setup Komunikasi Serial ke Mikrokontroler (Arduino/ESP32) ---
def setup_serial():
    try:
        # Pastikan port serial '/dev/ttyAMA0' atau '/dev/serial0' sudah benar
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

# --- Fungsi Pemrosesan Gambar ---
def process_image(frame, display_mode=False):
    global manual_threshold_value # Mengakses nilai threshold dari slider

    # Definisikan Region of Interest (ROI) untuk meminimalkan area pemrosesan
    roi_start_y = 160 # Baris mulai ROI (dari atas gambar)
    roi_end_y = 240   # Baris akhir ROI (dari atas gambar)
    
    # Cek jika frame terlalu kecil untuk ROI yang ditentukan
    if frame.shape[0] < roi_end_y:
        print("[ERROR] Frame terlalu kecil untuk ROI yang ditentukan.")
        return None, None, None
        
    # Potong ROI dari frame berwarna
    roi_color = frame[roi_start_y:roi_end_y, :] 
    
    # Konversi ROI ke grayscale
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Median Blur untuk mengurangi noise (lebih cepat dari Gaussian untuk noise tertentu)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # Terapkan Fixed Thresholding menggunakan nilai dari slider
    # cv2.THRESH_BINARY_INV: Piksel di bawah threshold jadi putih, di atas jadi hitam (cocok untuk garis hitam di latar terang)
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Operasi morfologi untuk membersihkan gambar (optional, bisa dihapus jika tidak diperlukan)
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    if display_mode:
        # Jika mode display aktif, proses juga full frame untuk visualisasi
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean
    else:
        # Untuk performa maksimal, hanya kembalikan ROI yang dibutuhkan
        return None, None, binary_roi_clean

# --- Fungsi Menghitung Posisi Garis ---
def calculate_line_position(roi_binary):
    # Hitung momen gambar untuk menemukan pusat massa (center of gravity)
    M = cv2.moments(roi_binary)
    # M['m00'] adalah area total piksel putih. Jika sangat kecil, artinya tidak ada garis.
    if M['m00'] > 50: # Ambang batas minimal untuk deteksi garis yang valid
        cx = int(M['m10'] / M['m00']) # Koordinat X pusat garis
        cy_roi = int(M['m01'] / M['m00']) # Koordinat Y pusat garis (relatif ke ROI)
        # Mengembalikan True, cx (absolute), dan cy (absolute)
        return True, cx, cy_roi + 160 # Tambahkan offset Y ROI untuk koordinat absolut
    return False, 0, 0 # Tidak ada garis terdeteksi

# --- Fungsi Menghitung Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # Pastikan input berada dalam rentang universe yang didefinisikan
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute() # Jalankan inferensi fuzzy
        return np.clip(fuzzy_ctrl.output['output'], -150, 150) # Pastikan output dalam rentang
    except Exception as e:
        # print(f"[FLC ERROR] {e}") # Debugging error FLC
        return 0.0 # Kembali ke 0 jika terjadi error

# --- Fungsi Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=55, scaling_factor=0.4): # Sesuaikan nilai ini saat tuning
    # base_pwm: Kecepatan dasar saat robot lurus
    # scaling_factor: Seberapa agresif respons steering terhadap output kontrol
    
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    # Batasi nilai PWM agar tidak melebihi rentang yang aman/efektif untuk motor Anda
    pwm_kiri = max(20, min(85, pwm_kiri)) # Min 20, Max 85
    pwm_kanan = max(20, min(85, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Fungsi Mengirim Perintah Motor via Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open: # Pastikan port serial terbuka
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n" # Format perintah (misal: "55,55\n")
            ser.write(cmd.encode()) # Kirim data sebagai byte
            # ser.flush() # Opsional: memastikan semua data terkirim, bisa menambah sedikit delay
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value # Akses global variable untuk threshold

    # Inisialisasi sistem
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=2) # Filter error

    prev_error = 0 # Menyimpan error sebelumnya untuk menghitung delta
    frame_count = 0 # Penghitung frame untuk debug print

    # --- Konfigurasi Tampilan GUI (untuk Debugging dan Tuning) ---
    DISPLAY_GUI = True # Setel ke False untuk performa maksimal (headless mode)

    if DISPLAY_GUI:
        # Buat jendela untuk tampilan ROI biner dan slider
        cv2.namedWindow("Threshold ROI")
        # Tambahkan trackbar (slider) untuk mengatur nilai threshold secara real-time
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        # Buat jendela untuk tampilan kamera utama
        cv2.namedWindow("Camera View")

    try:
        while True:
            # Ambil frame dari kamera
            frame = picam2.capture_array()
            
            # Jika GUI aktif, ambil nilai threshold terbaru dari slider
            if DISPLAY_GUI:
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            # Proses gambar (ROI) untuk mendapatkan citra biner garis
            _, _, roi_binary = process_image(frame, display_mode=DISPLAY_GUI)
            
            # Handle jika frame tidak valid
            if roi_binary is None:
                send_motor_commands(ser, 0, 0) # Berhenti jika ada masalah
                if frame_count % 30 == 0:
                    print("[DEBUG] Gagal memproses frame: ROI tidak valid.")
                frame_count += 1
                continue

            # Hitung posisi garis dari ROI biner
            line_detected, cx, cy = calculate_line_position(roi_binary)
            
            if line_detected:
                # Hitung error (deviasi dari pusat: 160)
                error = cx - 160
                error = error_filter.filter_error(error) # Filter error
                delta_error = error - prev_error # Hitung perubahan error
                prev_error = error # Simpan error saat ini untuk iterasi berikutnya

                # Hitung output kontrol dari sistem fuzzy
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                # Hitung nilai PWM motor kanan dan kiri
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                # Kirim perintah PWM ke mikrokontroler
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                # Debugging print ke konsol
                if frame_count % 10 == 0: # Cetak setiap 10 frame untuk mengurangi overhead
                    print(f"[DEBUG] Error: {error:4d}, Delta: {delta_error:4d}, FLC: {kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                # Jika garis tidak terdeteksi, berhenti atau lakukan aksi pencarian
                send_motor_commands(ser, 0, 0) # Berhenti total
                if frame_count % 20 == 0: # Cetak lebih jarang jika tidak ada garis
                    print("[DEBUG] Garis tidak terdeteksi")

            # --- Bagian Tampilan GUI (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # --- Garis Bantu Indikasi Belok ---
                # Garis vertikal di tengah kamera (acuan error 0)
                cv2.line(frame_for_display, (160, 160), (160, 240), (0, 255, 0), 2) # Hijau
                
                # Garis kuning: menandai "zona lurus" (misal: 20 piksel kiri/kanan dari tengah)
                straight_zone_half_width = 20 # Sesuaikan lebar zona lurus
                cv2.line(frame_for_display, (160 - straight_zone_half_width, 160), (160 - straight_zone_half_width, 240), (0, 255, 255), 1) # Kuning
                cv2.line(frame_for_display, (160 + straight_zone_half_width, 160), (160 + straight_zone_half_width, 240), (0, 255, 255), 1) # Kuning

                if line_detected:
                    # Gambar lingkaran di posisi garis terdeteksi (cx, cy)
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    # Tampilkan nilai error di layar
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Tampilkan nilai threshold yang sedang aktif
                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Tampilkan jendela gambar
                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary)
                
                # Cek tombol 'q' untuk keluar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan GUI ---

            frame_count += 1
            # Tidak ada time.sleep() di sini, biarkan loop berjalan secepat mungkin
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        # Pastikan motor berhenti saat program berakhir
        send_motor_commands(ser, 0, 0)
        if ser and ser.is_open:
            ser.close() # Tutup port serial
        picam2.stop() # Hentikan kamera
        if DISPLAY_GUI:
            cv2.destroyAllWindows() # Tutup semua jendela OpenCV
        print("[INFO] Program selesai")

# --- Main Program Entry Point ---
if __name__ == "__main__":
    main()
