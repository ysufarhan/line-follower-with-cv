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
    def __init__(self, window_size=3): # Sedikit tingkatkan window_size untuk stabilitas
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        return int(sum(self.error_history) / len(self.error_history))

# --- Kelas LineRecovery Dihilangkan ---
# Kelas LineRecovery tidak lagi digunakan.
# Motor akan selalu bergerak berdasarkan fuzzy output.

# --- Setup Logika Fuzzy (FLC) ---
def setup_fuzzy_logic():
    # Definisi Universe (Rentang Nilai Input/Output)
    error = ctrl.Antecedent(np.arange(-250, 251, 1), 'error') 
    delta = ctrl.Antecedent(np.arange(-180, 181, 1), 'delta') 
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output') 

    # CUSTOM MEMBERSHIP FUNCTIONS (Fungsi Keanggotaan)
    error['NL'] = fuzz.trimf(error.universe, [-250, -150, -60]) 
    error['NS'] = fuzz.trimf(error.universe, [-80, -30, -10]) 
    error['Z'] = fuzz.trimf(error.universe, [-15, 0, 15])     
    error['PS'] = fuzz.trimf(error.universe, [10, 30, 80])    
    error['PL'] = fuzz.trimf(error.universe, [60, 150, 250])  

    delta['NL'] = fuzz.trimf(delta.universe, [-180, -90, -30])
    delta['NS'] = fuzz.trimf(delta.universe, [-40, -15, -5])   
    delta['Z'] = fuzz.trimf(delta.universe, [-7, 0, 7])       
    delta['PS'] = fuzz.trimf(delta.universe, [5, 15, 40])
    delta['PL'] = fuzz.trimf(delta.universe, [30, 90, 180])

    output['L'] = fuzz.trimf(output.universe, [-150, -100, -50]) 
    output['LS'] = fuzz.trimf(output.universe, [-60, -20, -5])   
    output['Z'] = fuzz.trimf(output.universe, [-3, 0, 3])      
    output['RS'] = fuzz.trimf(output.universe, [5, 20, 60])    
    output['R'] = fuzz.trimf(output.universe, [50, 100, 150])   

    # Rule Base (25 Aturan yang Diperbarui)
    rules = [
        # error['NL'] (Negative Large)
        ctrl.Rule(error['NL'] & delta['NL'], output['L']), 
        ctrl.Rule(error['NL'] & delta['NS'], output['L']), 
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']), 
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),  # DIUBAH: Output lebih kuat ke kiri
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']), 

        # error['NS'] (Negative Small)
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['Z']), 
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),   
        ctrl.Rule(error['NS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['NS'] & delta['PL'], output['R']), # DIUBAH: Output lebih kuat ke kanan

        # error['Z'] (Zero)
        ctrl.Rule(error['Z'] & delta['NL'], output['L']), # DIUBAH: Output lebih kuat ke kiri
        ctrl.Rule(error['Z'] & delta['NS'], output['LS']), # DIUBAH: Output sedikit ke kiri
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),   
        ctrl.Rule(error['Z'] & delta['PS'], output['RS']), # DIUBAH: Output sedikit ke kanan
        ctrl.Rule(error['Z'] & delta['PL'], output['R']), # DIUBAH: Output lebih kuat ke kanan

        # error['PS'] (Positive Small)
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']), # DIUBAH: Netralkan
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),   
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']), # DIUBAH: Output sedikit ke kanan
        ctrl.Rule(error['PS'] & delta['PL'], output['R']), # DIUBAH: Output lebih kuat ke kanan

        # error['PL'] (Positive Large)
        ctrl.Rule(error['PL'] & delta['NL'], output['L']), # DIUBAH: Output sangat kuat ke kiri
        ctrl.Rule(error['PL'] & delta['NS'], output['LS']), # DIUBAH: Output lebih kuat ke kiri
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['R']), 
        ctrl.Rule(error['PL'] & delta['PL'], output['R']), 
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

# --- Setup Kamera Raspberry Pi ---
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Beri waktu kamera untuk stabil
    return picam2

# --- Setup Komunikasi Serial ---
def setup_serial():
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=0.1) 
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        print("Pastikan ESP32 terhubung dan port serial benar.")
        return None

# --- Pemrosesan Citra Menggunakan OpenCV ---
def process_image(frame, display_mode=False):
    global manual_threshold_value 

    # Inisialisasi ROI values to be returned (default)
    roi_start_y_local = 120 
    roi_end_y_local = 240 
    
    if frame.shape[0] < roi_end_y_local:
        print("[ERROR] Frame is too small for the defined ROI. Using default ROI for visualization.")
        return None, None, None, roi_start_y_local, roi_end_y_local 
        
    roi_color = frame[roi_start_y_local:roi_end_y_local, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    _, binary_roi = cv2.threshold(blurred_roi, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) 

    if display_mode:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        blurred_full = cv2.GaussianBlur(gray_full, (5,5), 0)
        _, binary_full = cv2.threshold(blurred_full, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return gray_full, binary_full, binary_roi_clean, roi_start_y_local, roi_end_y_local 
    else:
        return None, None, binary_roi_clean, roi_start_y_local, roi_end_y_local 

# --- Menghitung Posisi Garis ---
def calculate_line_position(roi_binary, roi_start_y): 
    M = cv2.moments(roi_binary)
    # Gunakan ambang batas yang lebih rendah untuk m00 agar deteksi lebih 'lentur'
    # saat garis mulai menghilang atau sangat tipis.
    if M['m00'] > 50: # Ambang batas momen diturunkan dari 100 ke 50
        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        return True, cx, cy_roi + roi_start_y 
    return False, 0, 0 # Jika tidak ada piksel putih yang cukup, anggap garis tidak terdeteksi


# --- Menghitung Output Kontrol Fuzzy ---
def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error): 
    try:
        # Penting: Pastikan input selalu dalam rentang universe FLC
        fuzzy_ctrl.input['error'] = np.clip(error_val, -250, 250) 
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -180, 180) 
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        # Jika terjadi error komputasi fuzzy (misal input di luar range walaupun sudah di-clip), 
        # kembalikan output 0 atau nilai aman lainnya.
        print(f"[FUZZY ERROR] Gagal komputasi fuzzy: {e}")
        return 0.0

# --- Menghitung Nilai PWM Motor ---
def calculate_motor_pwm(kontrol, base_pwm=45, scaling_factor=0.08): 
    FLC_DEAD_ZONE = 10 
    
    if abs(kontrol) < FLC_DEAD_ZONE:
        kontrol_scaled = 0 
    else:
        kontrol_scaled = kontrol * scaling_factor

    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled

    # Tetap pastikan PWM dalam rentang yang aman untuk motor
    MIN_PWM_OUTPUT = 35 
    MAX_PWM_OUTPUT = 55 

    pwm_kiri = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kiri))
    pwm_kanan = max(MIN_PWM_OUTPUT, min(MAX_PWM_OUTPUT, pwm_kanan))
    
    return int(pwm_kiri), int(pwm_kanan)

# --- Mengirim Perintah Motor Melalui Serial ---
def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush() 
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value 

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3) 
    # LineRecovery tidak lagi diinisialisasi/digunakan

    prev_error = 0 # Tetap simpan prev_error untuk perhitungan delta_error

    frame_count = 0

    DISPLAY_GUI = True 

    if DISPLAY_GUI:
        cv2.namedWindow("Threshold ROI")
        cv2.createTrackbar("Threshold", "Threshold ROI", manual_threshold_value, 255, on_trackbar_change)
        cv2.namedWindow("Camera View")

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            center_x_frame = width // 2 

            if DISPLAY_GUI:
                manual_threshold_value = cv2.getTrackbarPos("Threshold", "Threshold ROI")

            gray_full, binary_full, roi_binary, roi_start_y, roi_end_y = process_image(frame, display_mode=DISPLAY_GUI)
            
            line_detected = False
            current_cx = 0 # Inisialisasi cx untuk digunakan di luar blok if/else
            current_cy = 0
            
            if roi_binary is not None:
                # Perbarui threshold untuk deteksi garis yang lebih 'lentur'
                line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
                if line_detected:
                    current_cx = cx
                    current_cy = cy

            # Jika garis tidak terdeteksi secara valid (momen terlalu kecil/ROI kosong),
            # gunakan error terakhir yang valid atau set ke nilai ekstrem untuk memicu putaran.
            if not line_detected:
                # Opsi 1: Pertahankan error terakhir yang valid dan biarkan FLC menghitung
                # Asumsi FLC Anda akan menghasilkan putaran kuat jika errornya ekstrem
                
                # Opsi 2 (Alternatif, bisa lebih agresif): 
                # Paksakan error ke nilai ekstrem untuk memicu putaran pencarian
                # Misalnya, jika terakhir di kanan, paksakan error ke kiri ekstrem, dan sebaliknya
                if prev_error > 0: # terakhir garis di kanan, putar ke kiri
                    error = -200 # Error ekstrem ke kiri
                else: # terakhir garis di kiri atau tengah, putar ke kanan (default)
                    error = 200 # Error ekstrem ke kanan
                
                # Delta error bisa di set ke 0 atau dibiarkan saja (error - prev_error)
                # Jika ingin putaran yang konstan, set delta error ke 0 atau nilai kecil.
                delta_error = 0 # Agar FLC fokus pada nilai error statis ekstrem untuk putaran

                if frame_count % 20 == 0:
                    print(f"[DEBUG] Garis tidak terdeteksi. Memicu putaran pencarian: Error={error}")
            else:
                # Garis terdeteksi, hitung error seperti biasa
                error = current_cx - center_x_frame 
                error = error_filter.filter_error(error) # Filter error
                delta_error = error - prev_error
                
            prev_error = error # Update prev_error setelah perhitungan

            # Komputasi FLC dan kontrol motor selalu berjalan
            kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
            pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
            send_motor_commands(ser, pwm_kiri, pwm_kanan)

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 
                flc_error_z_boundary = 15 
                cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected: 
                    cv2.circle(frame_for_display, (current_cx, current_cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: 
                    cv2.putText(frame_for_display, f"GARIS HILANG! SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"Forced Error: {error}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame_for_display, f"PWM L:{pwm_kiri} R:{pwm_kanan}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


                cv2.imshow("Camera View", frame_for_display)
                if binary_full is not None:
                    cv2.imshow("Threshold ROI", binary_full)
                elif roi_binary is not None: 
                    cv2.imshow("Threshold ROI", roi_binary)
                else: 
                    try:
                        cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                    except Exception as e:
                        print(f"[VISUALIZATION ERROR] Could not display dummy ROI: {e}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- Akhir Bagian Tampilan ---

            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna")
    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan tak terduga: {e}")
    finally:
        send_motor_commands(ser, 0, 0) # Pastikan motor berhenti saat program berakhir
        if ser and ser.is_open:
            ser.close()
        picam2.stop()
        if DISPLAY_GUI:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai")

if __name__ == "__main__":
    main()
