from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Class untuk Filtering Error ---
class ErrorFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.error_history = []
        self.alpha = 0.5 # Konservatif untuk smoothing

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        if not self.error_history:
            return 0

        # Exponential Moving Average (EMA)
        smoothed_error = self.error_history[0]
        for i in range(1, len(self.error_history)):
            smoothed_error = self.alpha * self.error_history[i] + (1 - self.alpha) * smoothed_error
            
        return int(smoothed_error)

# --- Setup Logika Fuzzy (Disesuaikan untuk Kamera Sangat Rendah) ---
def setup_fuzzy_logic():
    # Universe disesuaikan secara signifikan karena perubahan error yang lebih drastis
    # Dengan kamera ~1.5-2cm dari lantai, error bisa mencapai +-200 hingga +-300 atau lebih
    error = ctrl.Antecedent(np.arange(-350, 351, 1), 'error') 
    delta = ctrl.Antecedent(np.arange(-200, 201, 1), 'delta')
    output = ctrl.Consequent(np.arange(-200, 201, 1), 'output')

    # Membership Functions (Disesuaikan untuk Angle Kamera Rendah)
    # ERROR - Zona netral lebih lebar, range ekstrem diperluas
    error['NL'] = fuzz.trimf(error.universe, [-350, -250, -100]) # Lebih lebar
    error['NS'] = fuzz.trimf(error.universe, [-150, -50, -15])
    error['Z']  = fuzz.trimf(error.universe, [-25, 0, 25])      # Zona netral lebih lebar
    error['PS'] = fuzz.trimf(error.universe, [15, 50, 150])
    error['PL'] = fuzz.trimf(error.universe, [100, 250, 350])   # Lebih lebar

    # DELTA (Perubahan Error) - Zona netral lebih lebar, range diperluas
    delta['NL'] = fuzz.trimf(delta.universe, [-200, -120, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-70, -25, -8])
    delta['Z']  = fuzz.trimf(delta.universe, [-15, 0, 15])      # Zona netral lebih lebar
    delta['PS'] = fuzz.trimf(delta.universe, [8, 25, 70])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 120, 200])

    # OUTPUT (Kontrol Motor) - Range lebih besar untuk koreksi agresif saat dibutuhkan
    output['L']  = fuzz.trimf(output.universe, [-200, -140, -80])
    output['LS'] = fuzz.trimf(output.universe, [-100, -50, -20])
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])    # Zona netral lebih lebar
    output['RS'] = fuzz.trimf(output.universe, [20, 50, 100])
    output['R']  = fuzz.trimf(output.universe, [80, 140, 200])

    # Rules - Aturan inti tetap, tapi dengan MFs yang telah disesuaikan,
    # perilakunya akan berbeda.
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']), # Sedikit lebih agresif dari Z
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

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
        ctrl.Rule(error['PS'] & delta['NS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']), # Sedikit lebih agresif dari Z
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera ---
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (320, 240), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2) 
    print("[INFO] Kamera berhasil diinisialisasi pada resolusi 320x240.")
    return picam2

# --- Setup Komunikasi Serial (UART) ---
def setup_serial():
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        print("[UART] Port serial berhasil dibuka.")
        return ser
    except serial.SerialException as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}. Pastikan UART diaktifkan dan terhubung dengan benar.")
        return None
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan tidak terduga saat membuka port serial: {e}")
        return None

# --- Pemrosesan Gambar ---
def process_image(frame):
    # ROI: 60% dari tinggi gambar hingga ke bawah (pixel 144 hingga 239)
    # Ini sangat penting karena dengan kamera rendah, garis akan tampak sangat besar di bagian bawah frame.
    # Memilih ROI ini akan fokus pada area yang relevan di depan robot.
    roi_start_row = int(frame.shape[0] * 0.6) # Dimulai dari baris 144 dari 240
    
    # Crop frame
    cropped_frame = frame[roi_start_row:frame.shape[0], :]

    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Thresholding lebih baik untuk kondisi cahaya bervariasi
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary, roi_start_row

# --- Deteksi Posisi Garis ---
def calculate_line_position(roi_binary_frame, min_area_threshold=100):
    kernel = np.ones((3,3), np.uint8)
    roi_clean = cv2.morphologyEx(roi_binary_frame, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > min_area_threshold: 
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                return True, cx, area
    return False, 0, 0

# --- Koreksi Perspektif (Tetap dikomentari, pertimbangkan lagi jika ada osilasi kuat) ---
# Dengan kamera yang sangat rendah dan ROI sempit, koreksi cx mungkin tidak sekompleks
# yang Anda bayangkan. Cx langsung dari ROI cenderung sudah cukup representatif.
def perspective_correction(error, roi_height, frame_width):
    # return error # Pilihan default: tidak ada koreksi perspektif
    
    # Jika perlu, coba uncomment dan sesuaikan faktor di bawah
    # Misalnya, error yang sama di bagian bawah ROI (garis dekat)
    # secara fisik pergeserannya lebih kecil dari error yang sama di bagian atas ROI (garis jauh).
    # Namun karena kita hanya mengambil centroid dari seluruh ROI, efek ini mungkin tereduksi.
    # Contoh sederhana:
    # factor = 1.0 + (abs(error) / (frame_width / 2)) * 0.1
    # return int(error * factor)
    return error

# --- Komputasi Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # Pastikan input sesuai dengan universe yang didefinisikan
        fuzzy_ctrl.input['error'] = np.clip(error_val, -350, 350)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -200, 200)
        fuzzy_ctrl.compute()
        # Batasi output agar tidak melebihi range yang valid
        return np.clip(fuzzy_ctrl.output['output'], -200, 200)
    except Exception as e:
        print(f"[FLC ERROR] Error saat komputasi fuzzy: {e}")
        return 0.0 # Kembali 0 jika terjadi error

# --- Perhitungan PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.35): # Scaling factor sedikit ditingkatkan untuk respons
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    
    pwm_min = 25 # Mungkin perlu lebih rendah dari 30 untuk tikungan tajam
    pwm_max = 75 # Mungkin perlu lebih tinggi dari 70
    pwm_kiri = max(pwm_min, min(pwm_max, pwm_kiri))
    pwm_kanan = max(pwm_min, min(pwm_max, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Kontrol Kecepatan Adaptif ---
def adaptive_speed_control(error, base_pwm=50):
    abs_error = abs(error)
    # Ambang batas error disesuaikan dengan universe yang lebih besar
    if abs_error > 150: # Error sangat besar (tikungan sangat tajam/hampir keluar)
        return max(20, base_pwm - 20) # Turunkan kecepatan drastis
    elif abs_error > 70: # Error sedang (tikungan)
        return max(30, base_pwm - 10)
    else: # Error kecil (jalur lurus)
        return base_pwm

# --- Kirim Perintah Motor ke Arduino/Mikrokontroler ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser:
        try:
            cmd = f"M{pwm_kiri:03d},{pwm_kanan:03d}\n" 
            ser.write(cmd.encode())
            ser.flush()
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data serial: {e}")
            # ser.close() # Hindari menutup/membuka di sini, biar di main loop/finally
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan tidak terduga saat mengirim perintah motor: {e}")

# --- Fungsi Utama ---
def main():
    print("[INFO] Memulai inisialisasi sistem...")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=5)

    prev_error = 0
    frame_count = 0
    lost_line_count = 0
    last_valid_error = 0
    
    MAX_LOST_FRAMES = 50 # Toleransi lebih lama karena garis bisa hilang sesaat di tikungan tajam
    RECOVERY_PWM_SCALE = 0.25 # Skala recovery sedikit lebih besar
    RECOVERY_BASE_PWM = 45 # Base PWM saat recovery sedikit lebih cepat

    print("[INFO] Sistem siap! Tekan 'q' pada jendela tampilan untuk keluar.")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            binary_roi, roi_start_row = process_image(frame)

            line_detected, cx_in_roi, area = calculate_line_position(binary_roi)
            
            if line_detected:
                lost_line_count = 0 
                
                center_x_roi = binary_roi.shape[1] // 2
                raw_error = cx_in_roi - center_x_roi
                
                error = error_filter.filter_error(raw_error)
                
                # Gunakan koreksi perspektif jika Anda mengaktifkannya
                # error = perspective_correction(error, binary_roi.shape[0], frame.shape[1])
                
                delta_error = error - prev_error
                prev_error = error
                last_valid_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                
                adaptive_base_pwm = adaptive_speed_control(error, base_pwm=55) # Menaikkan sedikit base_pwm
                
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol, adaptive_base_pwm)
                
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0: # Cetak setiap 10 frame (lebih sering dari sebelumnya karena error lebih besar)
                    print(f"[DEBUG] Err:{error:4d} ΔErr:{delta_error:4d} FLC:{kontrol:6.1f} PWM:L{pwm_kiri:3d} R{pwm_kanan:3d} Area:{area:4.0f}")
                    
            else: # Garis tidak terdeteksi
                lost_line_count += 1
                
                if lost_line_count < MAX_LOST_FRAMES:
                    recovery_control = last_valid_error * RECOVERY_PWM_SCALE
                    pwm_kiri, pwm_kanan = calculate_motor_pwm(recovery_control, RECOVERY_BASE_PWM)
                    send_motor_commands(ser, pwm_kiri, pwm_kanan)
                    if frame_count % 10 == 0:
                        print(f"[RECOVERY] Garis hilang ({lost_line_count}/{MAX_LOST_FRAMES}). Mencari: {last_valid_error}. PWM: L{pwm_kiri:3d} R{pwm_kanan:3d}")
                else:
                    send_motor_commands(ser, 0, 0)
                    if frame_count % 30 == 0:
                        print("[WARNING] Garis hilang terlalu lama! Robot berhenti.")

            # --- Visualisasi ---
            if frame_count % 3 == 0: # Tampilkan lebih sering untuk debugging visual (misal 3 atau 4)
                frame_display = frame.copy()
                
                cv2.rectangle(frame_display, (0, roi_start_row), 
                              (frame.shape[1], frame.shape[0]), (0, 255, 0), 1)
                
                center_x_frame = frame.shape[1] // 2
                cv2.line(frame_display, (center_x_frame, 0), (center_x_frame, frame.shape[0]), (0, 255, 0), 1)
                
                if line_detected:
                    visual_cy = roi_start_row + (binary_roi.shape[0] // 2) 
                    cv2.circle(frame_display, (cx_in_roi, visual_cy), 5, (0, 0, 255), -1)
                    
                    cv2.putText(frame_display, f"E:{error}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_display, f"PWM:L{pwm_kiri} R{pwm_kanan}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Line Following - RPi4", cv2.resize(frame_display, (640, 480)))
                cv2.imshow("Binary ROI", cv2.resize(binary_roi, (320, 160)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            time.sleep(0.03) # Coba 30-33 FPS untuk RPi4 jika memungkinkan
            
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Terjadi kesalahan fatal: {e}")
    finally:
        print("[INFO] Membersihkan sistem...")
        send_motor_commands(ser, 0, 0)
        if ser:
            ser.close()
            print("[INFO] Port serial ditutup.")
        picam2.stop()
        print("[INFO] Kamera dihentikan.")
        cv2.destroyAllWindows()
        print("[INFO] Jendela OpenCV ditutup. Program selesai.")

if __name__ == "__main__":
    main()
