from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- PENGATURAN GLOBAL ---
# Set ke False untuk menjalankan robot dengan kecepatan penuh tanpa antarmuka grafis (GUI).
# Set ke True saat melakukan tuning atau debugging untuk melihat output visual.
DISPLAY_GUI = True

# --- Kelas untuk Filter Error (Rata-rata Bergerak) ---
class ErrorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Universe (rentang nilai) untuk input dan output
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # Fungsi Keanggotaan (Membership Functions) untuk ERROR
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10])  # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-15, 0, 15])     # Zero (di tengah)
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])     # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])   # Positive Large

    # Fungsi Keanggotaan untuk DELTA (Perubahan Error)
    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-7, 0, 7])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    # Fungsi Keanggotaan untuk OUTPUT (Koreksi Belok)
    output['L']  = fuzz.trimf(output.universe, [-150, -100, -50]) # Belok Kiri Kuat
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   # Belok Kiri Ringan
    output['Z']  = fuzz.trimf(output.universe, [-3, 0, 3])       # Lurus
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])      # Belok Kanan Ringan
    output['R']  = fuzz.trimf(output.universe, [50, 100, 150])   # Belok Kanan Kuat

    # Rule Base (Aturan Fuzzy) - 25 Aturan
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), ctrl.Rule(error['NL'] & delta['NS'], output['L']), ctrl.Rule(error['NL'] & delta['Z'], output['LS']), ctrl.Rule(error['NL'] & delta['PS'], output['Z']), ctrl.Rule(error['NL'] & delta['PL'], output['Z']),
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),ctrl.Rule(error['NS'] & delta['NS'], output['Z']), ctrl.Rule(error['NS'] & delta['Z'], output['Z']),  ctrl.Rule(error['NS'] & delta['PS'], output['RS']),ctrl.Rule(error['NS'] & delta['PL'], output['RS']),
        ctrl.Rule(error['Z']  & delta['NL'], output['LS']),ctrl.Rule(error['Z']  & delta['NS'], output['Z']), ctrl.Rule(error['Z']  & delta['Z'], output['Z']),  ctrl.Rule(error['Z']  & delta['PS'], output['Z']), ctrl.Rule(error['Z']  & delta['PL'], output['RS']),
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),ctrl.Rule(error['PS'] & delta['NS'], output['RS']),ctrl.Rule(error['PS'] & delta['Z'], output['Z']),  ctrl.Rule(error['PS'] & delta['PS'], output['Z']), ctrl.Rule(error['PS'] & delta['PL'], output['RS']),
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']), ctrl.Rule(error['PL'] & delta['NS'], output['Z']), ctrl.Rule(error['PL'] & delta['Z'], output['RS']),ctrl.Rule(error['PL'] & delta['PS'], output['R']), ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera Raspberry Pi ---
def setup_camera():
    picam2 = Picamera2()
    # Gunakan video_configuration untuk streaming berkelanjutan
    config = picam2.create_video_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("[INFO] Kamera berhasil diinisialisasi.")
    return picam2

# --- Setup Komunikasi Serial ke Mikrokontroler (ESP32/Arduino) ---
def setup_serial():
    try:
        # Sesuaikan port serial jika berbeda. Untuk RPi 3/4, /dev/ttyAMA0 biasanya untuk GPIO.
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        print("[INFO] Port serial berhasil dibuka.")
        return ser
    except Exception as e:
        print(f"[ERROR] Gagal membuka serial port: {e}")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame):
    # Tentukan Region of Interest (ROI) di bagian bawah gambar
    roi_start_y = 160
    roi_end_y = 240
    
    if frame.shape[0] < roi_end_y:
        return None, 0
        
    roi_color = frame[roi_start_y:roi_end_y, :]
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 5)
    
    # --- PERUBAHAN UTAMA: Thresholding Otomatis (Otsu) ---
    # Ini membuat robot adaptif terhadap perubahan pencahayaan.
    # THRESH_BINARY_INV digunakan karena garis hitam di latar terang.
    detected_threshold, binary_roi = cv2.threshold(
        blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Operasi morfologi untuk membersihkan noise kecil
    kernel = np.ones((5,5), np.uint8)
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)
    
    return binary_roi_clean, detected_threshold

# --- Menghitung Posisi Garis (Centroid) ---
def calculate_line_position(roi_binary):
    M = cv2.moments(roi_binary)
    if M['m00'] > 200: # Cek apakah ada cukup piksel putih (garis)
        cx = int(M['m10'] / M['m00'])
        return True, cx
    return False, 0

# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except:
        return 0.0

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=40, scaling_factor=0.1):
    # base_pwm: Kecepatan dasar robot saat lurus.
    # scaling_factor: Seberapa besar pengaruh FLC terhadap belokan (agresivitas).
    # Tune nilai-nilai ini sesuai dengan karakteristik fisik robot Anda.
    
    FLC_DEAD_ZONE = 10 # Jika output FLC di bawah ini, anggap lurus.
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm - kontrol_scaled # Jika kontrol positif (belok kanan), kurangi speed kiri
    pwm_kanan = base_pwm + kontrol_scaled # dan tambah speed kanan

    # Batasi nilai PWM ke rentang aman untuk motor Anda
    MIN_PWM_OUTPUT = 25
    MAX_PWM_OUTPUT = 60
    
    pwm_kiri = int(max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri)))
    pwm_kanan = int(max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan)))
    
    return pwm_kiri, pwm_kanan

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            # Format: "PWM_KIRI,PWM_KANAN\n"
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")

# --- Fungsi Utama Program ---
def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter()

    prev_error = 0
    last_known_error = 0 # Simpan error terakhir jika garis hilang

    if DISPLAY_GUI:
        cv2.namedWindow("Camera View")
        cv2.namedWindow("Binary ROI")

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2

            roi_binary, detected_thresh = process_image(frame)
            
            if roi_binary is None:
                if ser: send_motor_commands(ser, 0, 0)
                continue

            line_detected, cx = calculate_line_position(roi_binary)
            
            if line_detected:
                error = cx - center_x_frame
                last_known_error = error
                
                filtered_error = error_filter.filter_error(error)
                delta_error = filtered_error - prev_error
                prev_error = filtered_error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, filtered_error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                
                if ser: send_motor_commands(ser, pwm_kiri, pwm_kanan)

            else: # Strategi jika garis hilang
                if ser: send_motor_commands(ser, 0, 0) # Berhenti
                print("[WARN] Garis tidak terdeteksi.")
                # Anda bisa menambahkan logika lebih canggih di sini,
                # misalnya berputar ke arah 'last_known_error'.

            # --- Bagian Tampilan (Hanya berjalan jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Gambar box ROI pada frame utama
                cv2.rectangle(frame_for_display, (0, 160), (width-1, 240), (255, 0, 0), 2)
                
                if line_detected:
                    # Gambar titik pusat garis (centroid) di dalam ROI pada frame utama
                    cy_display = 160 + (roi_binary.shape[0] // 2)
                    cv2.circle(frame_for_display, (cx, cy_display), 7, (0, 0, 255), -1)
                    
                    # Tampilkan nilai error
                    cv2.putText(frame_for_display, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Tampilkan nilai threshold yang terdeteksi Otsu
                cv2.putText(frame_for_display, f"Otsu Thresh: {int(detected_thresh)}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow("Camera View", frame_for_display)
                cv2.imshow("Binary ROI", roi_binary)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna (Ctrl+C).")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        print("[INFO] Menghentikan motor dan menutup semua sumber daya...")
        if ser: send_motor_commands(ser, 0, 0)
        if ser and ser.is_open:
            ser.close()
        picam2.stop()
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()
content_copy
download
Use code with caution.
Python
