from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Global Variable untuk Threshold Manual ---
# Ini akan diperbarui oleh slider OpenCV
manual_threshold_value = 100 # Nilai default awal, sesuaikan setelah tuning pertama

# --- Callback Function untuk Trackbar (Slider) ---
def on_trackbar_change(val):
    global manual_threshold_value
    manual_threshold_value = val

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=3): # Sedikit tingkatkan window_size untuk stabilitas
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
    # Rentang error dan delta diperluas sedikit untuk menampung belokan tajam
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error') # DIUBAH: Rentang diperluas
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta') # DIUBAH: Rentang diperluas
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan)
    # DIUBAH: Penyesuaian Krusial untuk respons lebih halus dan stabil
    # Fokus pada lurus sempurna dan belok yang responsif namun tidak berlebihan

    # ERROR: Posisi garis relatif terhadap pusat kamera (pusat: 0)
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])  # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])     # DIUBAH: DILEBARKAN sedikit untuk stabilitas LURUS
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])     # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])   # Positive Large

    # DELTA: Perubahan error antar frame (kecepatan error)
    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])   # DIUBAH: DILEBARKAN sedikit untuk mengurangi sensitivitas noise
    delta['Z']  = fuzz.trimf(delta.universe, [-7, 0, 7])       # DIUBAH: DILEBARKAN sedikit untuk mengurangi sensitivitas noise
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    # OUTPUT: Nilai kontrol yang akan digunakan untuk menghitung PWM
    output['L']  = fuzz.trimf(output.universe, [-150, -100, -50]) # Lebih banyak ruang untuk respons belok
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   # Respons belok kiri kecil
    output['Z']  = fuzz.trimf(output.universe, [-3, 0, 3])       # DIUBAH: SANGAT SEMPIT untuk lurus sempurna
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])      # Respons belok kanan kecil
    output['R']  = fuzz.trimf(output.universe, [50, 100, 150])   # Lebih banyak ruang untuk respons belok

    # Rule Base (DIUBAH: Disempurnakan untuk belokan mulus dan stabilitas lurus)
    rules = [
        # Error Negative Large (Garis di kiri jauh dari pusat) -> Belok Kanan Kuat
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), # Jauh kiri, makin jauh kiri -> L (Koreksi Agresif)
        ctrl.Rule(error['NL'] & delta['NS'], output['L']), # Jauh kiri, sedikit menjauh -> L
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']), # Jauh kiri, stabil -> LS (Koreksi Sedang)
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']), # Jauh kiri, mendekat ke tengah -> Z (Pencegah Overshoot)
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']), # Jauh kiri, cepat mendekat -> Z

        # Error Negative Small (Garis di kiri sedikit dari pusat) -> Belok Kanan Ringan
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),# Agak kiri, menjauh cepat -> LS
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']), # DIUBAH: Agak kiri, menjauh -> Z (Mulai Luruskan)
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),  # DIUBAH: Agak kiri, stabil -> Z (TARGET LURUS)
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),# Agak kiri, mendekat -> RS
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),# Agak kiri, cepat mendekat -> RS

        # Error Zero (Garis di tengah) -> Prioritas Utama: Lurus Sempurna
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']), # Di tengah, tapi menjauh ke kiri -> LS (Koreksi Halus)
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),  # DIUBAH: Di tengah, menjauh sedikit ke kiri -> Z
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),   # DIUBAH: Sempurna di tengah, stabil -> Z (IDEAL!)
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),  # DIUBAH: Di tengah, menjauh sedikit ke kanan -> Z
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']), # Di tengah, tapi menjauh ke kanan -> RS (Koreksi Halus)

        # Error Positive Small (Garis di kanan sedikit dari pusat) -> Belok Kiri Ringan
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),# Agak kanan, cepat mendekat -> LS
        ctrl.Rule(error['PS'] & delta['NS'], output['RS']),# DIUBAH: Agak kanan, mendekat -> RS
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),  # DIUBAH: Agak kanan, stabil -> Z (TARGET LURUS)
        ctrl.Rule(error['PS'] & delta['PS'], output['Z']), # DIUBAH: Agak kanan, menjauh -> Z (Mulai Luruskan)
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),# Agak kanan, menjauh cepat -> RS

        # Error Positive Large (Garis di kanan jauh dari pusat) -> Belok Kiri Kuat
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']), # Jauh kanan, cepat mendekat -> Z (Pencegah Overshoot)
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']), # Jauh kanan, mendekat -> Z
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']), # Jauh kanan, stabil -> RS (Koreksi Sedang)
        ctrl.Rule(error['PL'] & delta['PS'], output['R']), # Jauh kanan, menjauh -> R
        ctrl.Rule(error['PL'] & delta['PL'], output['R']), # Jauh kanan, makin jauh kanan -> R (Koreksi Agresif)
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera Raspberry Pi ---
def setup_camera():
    picam2 = Picamera2()
    # Resolusi 320x240 sudah cukup baik untuk Pi 4, fokus pada framerate.
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Beri waktu kamera untuk stabil
    return picam2

# --- Setup Komunikasi Serial ---
def setup_serial():
    try:
        # Gunakan /dev/ttyAMA0 atau /dev/ttyS0, sesuaikan dengan konfigurasi Anda
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1) 
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        print("Pastikan ESP32 terhubung dan port serial benar.")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame, display_mode=False):
    global manual_threshold_value # Akses nilai threshold dari global variable

    roi_start_y = 160
    roi_end_y = 240
    
    if frame.shape[0] < roi_end_y:
        print("[ERROR] Frame terlalu kecil untuk ROI yang ditentukan.")
        return None, None, None
        
    roi_color = frame[roi_start_y:roi_end_y, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # MENGGUNAKAN NILAI THRESHOLD DARI SLIDER
    # THRESH_BINARY_INV: Piksel di bawah threshold jadi putih, di atas jadi hitam (untuk garis hitam di latar terang)
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) # Tambahkan Open untuk noise kecil

    if display_mode:
        # Tampilan full frame juga pakai threshold dari slider
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean
    else:
        return None, None, binary_roi_clean

# --- Menghitung Posisi Garis ---
def calculate_line_position(roi_binary):
    # Menggunakan cv2.moments untuk mencari pusat massa
    M = cv2.moments(roi_binary)
    # Ambang batas minimal untuk mendeteksi garis (sesuaikan jika noise sering terdeteksi)
    if M['m00'] > 100: # DIUBAH: Disesuaikan ambang batas
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + 160 # cx relatif terhadap ROI, tambahkan offset untuk posisi absolut
    return False, 0, 0

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # Pastikan input berada dalam rentang universe FLC
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) # DIUBAH: Sesuaikan dengan universe error
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) # DIUBAH: Sesuaikan dengan universe delta
        fuzzy_ctrl.compute()
        # Pastikan output berada dalam rentang yang valid
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        # print(f"[FLC ERROR] {e}. Input Error: {error_val}, Delta Error: {delta_error}")
        return 0.0

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=35, scaling_factor=0.08): # DIUBAH: Default PWM dan Scaling Factor
    # FLC Dead Zone: Jika kontrol sangat kecil, anggap robot harus lurus.
    # Membantu stabilitas di garis lurus, mencegah zigzag kecil.
    FLC_DEAD_ZONE = 10 # DIUBAH: Ukuran dead zone (dalam nilai output FLC)
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 # Tidak ada koreksi, PWM kiri = kanan
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Batasi nilai PWM agar tidak melebihi rentang yang aman atau efektif
    MIN_PWM_OUTPUT = 20 # DIUBAH: Batas bawah PWM
    MAX_PWM_OUTPUT = 55 # DIUBAH: Batas atas PWM

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush() # Pastikan data terkirim
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value # Akses nilai threshold dari global variable

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3) # DIUBAH: window_size filter

    prev_error = 0
    frame_count = 0

    DISPLAY_GUI = True # WAJIB TRUE untuk menggunakan slider dan visualisasi

    if DISPLAY_GUI:
        cv2.namedWindow("Threshold ROI")
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        cv2.namedWindow("Camera View")

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2 # Pusat x dari frame, yaitu 160 untuk 320x240

            if DISPLAY_GUI:
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            gray_full, binary_full, roi_binary = process_image(frame, display_mode=DISPLAY_GUI)
            
            if roi_binary is None:
                send_motor_commands(ser, 0, 0)
                if frame_count % 30 == 0:
                    print("[DEBUG] Gagal memproses frame: ROI tidak valid.")
                frame_count += 1
                continue

            line_detected, cx, cy = calculate_line_position(roi_binary)
            
            if line_detected:
                error = cx - center_x_frame # Hitung error relatif terhadap pusat frame
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    print(f"[DEBUG] Error: {error:4d}, Delta: {delta_error:4d}, FLC: {kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0) # Berhenti jika garis tidak terdeteksi
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Garis tengah acuan (hijau)
                cv2.line(frame_for_display, (center_x_frame, 0), (center_x_frame, height), (0, 255, 0), 2)
                
                # Garis bantu indikasi belok (kuning)
                # Ini menunjukkan zona di mana error masih dianggap "lurus" oleh FLC.
                # Sesuaikan 15 dengan nilai batas 'Z' pada error FLC Anda.
                flc_error_z_boundary = 15 # Dari definition error['Z'] [-15, 0, 15]
                cv2.line(frame_for_display, (center_x_frame - flc_error_z_boundary, 160), (center_x_frame - flc_error_z_boundary, 240), (0, 255, 255), 1)
                cv2.line(frame_for_display, (center_x_frame + flc_error_z_boundary, 160), (center_x_frame + flc_error_z_boundary, 240), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Threshold ROI", roi_binary)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            # Tidak ada time.sleep() di sini, biarkan loop berjalan secepat mungkin
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti
        if ser and ser.is_open:
            ser.close()
        picam2.stop()
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
