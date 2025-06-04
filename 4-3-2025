from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ErrorFilter:
    def __init__(self, window_size=2): # Mengurangi window_size untuk responsivitas lebih baik
        self.window_size = window_size
        self.error_history = []

    def filter_error(self, error):
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        # Menggunakan rata-rata sederhana, int() untuk performa
        return int(sum(self.error_history) / len(self.error_history))

def setup_fuzzy_logic():
    # Definisi universe yang diperluas
    error = ctrl.Antecedent(np.arange(-200, 201, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-150, 151, 1), 'delta')
    output = ctrl.Consequent(np.arange(-150, 151, 1), 'output')

    # CUSTOM MEMBERSHIP FUNCTIONS - Dioptimalkan untuk belokan 90 derajat
    # ERROR - Dipersempit range untuk respons lebih cepat
    error['NL'] = fuzz.trimf(error.universe, [-200, -180, -80])
    error['NS'] = fuzz.trimf(error.universe, [-120, -50, -10])
    error['Z']  = fuzz.trimf(error.universe, [-20, 0, 20])
    error['PS'] = fuzz.trimf(error.universe, [10, 50, 120])
    error['PL'] = fuzz.trimf(error.universe, [80, 180, 200])

    # DELTA (change of error) - Ditingkatkan sensitivitas
    delta['NL'] = fuzz.trimf(delta.universe, [-150, -120, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-70, -30, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-10, 0, 10])
    delta['PS'] = fuzz.trimf(delta.universe, [5, 30, 70])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 120, 150])

    # OUTPUT - Ditingkatkan respons steering
    output['L']  = fuzz.trimf(output.universe, [-150, -120, -70])
    output['LS'] = fuzz.trimf(output.universe, [-90, -40, -15])
    output['Z']  = fuzz.trimf(output.universe, [-8, 0, 8])
    output['RS'] = fuzz.trimf(output.universe, [15, 40, 90])
    output['R']  = fuzz.trimf(output.universe, [70, 120, 150])

    # Rules yang sama seperti sebelumnya
    rules = [
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),

        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),

        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ]

    control_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(control_system)

def setup_camera():
    picam2 = Picamera2()
    # Menurunkan resolusi dan preview stream jika hanya main stream yang dibutuhkan
    # Resolusi main stream: 320x240, di Raspberry Pi 4 mungkin lebih stabil
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    try:
        # Pastikan /dev/ttyAMA0 adalah port serial yang benar untuk Raspberry Pi 4
        # Terkadang di Pi 4 bisa juga /dev/serial0
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1) # Timeout lebih kecil
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame, display_mode=False):
    # Mengambil ROI langsung dari frame berwarna, lalu konversi ke grayscale
    # Mengurangi operasi cvtColor pada seluruh frame
    roi_start_y = 160
    roi_end_y = 240
    
    # Crop ROI lebih awal untuk mengurangi jumlah piksel yang diproses
    # Pastikan frame memiliki dimensi yang cukup
    if frame.shape[0] < roi_end_y:
        print("[ERROR] Frame terlalu kecil untuk ROI yang ditentukan.")
        return None, None, None # Handle error
        
    roi_color = frame[roi_start_y:roi_end_y, :] 
    
    gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    # Menggunakan median blur yang seringkali lebih cepat dari Gaussian untuk noise salt-and-pepper
    # Kernel yang lebih kecil (3,3)
    blurred_roi = cv2.medianBlur(gray_roi, 3) 
    
    # Thresholding OTSU untuk adaptasi cahaya
    _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations (opsional, bisa dihilangkan jika tidak terlalu diperlukan)
    # Kernel yang lebih kecil untuk performa
    kernel = np.ones((3,3), np.uint8) 
    binary_roi_clean = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    # binary_roi_clean = cv2.morphologyEx(binary_roi_clean, cv2.MORPH_OPEN, kernel, iterations=1) # Mungkin tidak diperlukan

    # Jika display_mode True, return full frame dan binary untuk debugging
    if display_mode:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_full = cv2.threshold(cv2.GaussianBlur(gray_full, (5,5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return gray_full, binary_full, binary_roi_clean
    else:
        return None, None, binary_roi_clean # Tidak perlu kembalikan full frame jika tidak display

def calculate_line_position(roi_binary):
    # Momen harus dari ROI yang sudah diproses
    M = cv2.moments(roi_binary)
    if M['m00'] > 50: # Mengurangi sensitivitas deteksi momen untuk noise
        cx = int(M['m10'] / M['m00'])
        # cy tidak lagi penting untuk error, karena ROI sudah fixed pada tinggi tertentu
        # Tapi jika dibutuhkan untuk visualisasi, bisa dihitung relatif terhadap ROI
        cy_roi = int(M['m01'] / M['m00']) # cy relatif terhadap ROI
        return True, cx, cy_roi + 160 # Mengembalikan cx dan cy absolut untuk visualisasi
    return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    try:
        # Menggunakan np.clip untuk memastikan nilai input berada dalam universe
        fuzzy_ctrl.input['error'] = np.clip(error_val, -200, 200)
        fuzzy_ctrl.input['delta'] = np.clip(delta_error, -150, 150)
        fuzzy_ctrl.compute()
        return np.clip(fuzzy_ctrl.output['output'], -150, 150)
    except Exception as e:
        # Menghindari print berlebihan di loop utama jika ada banyak error
        # print(f"[FLC ERROR] {e}") 
        return 0.0

def calculate_motor_pwm(kontrol, base_pwm=50, scaling_factor=0.35): # Sedikit tingkatkan scaling
    kontrol_scaled = kontrol * scaling_factor
    pwm_kiri = base_pwm + kontrol_scaled
    pwm_kanan = base_pwm - kontrol_scaled
    pwm_kiri = max(20, min(85, pwm_kiri))  # Tetap jaga rentang PWM
    pwm_kanan = max(20, min(85, pwm_kanan))
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    if ser and ser.is_open: # Cek apakah serial port terbuka
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            # ser.flush() # flush() mungkin tidak selalu diperlukan dan bisa memperlambat
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] Gagal mengirim data: {e}")
            # Mungkin perlu penanganan error lebih lanjut, misal re-open serial port
        except Exception as e:
            print(f"[SERIAL GENERAL ERROR] {e}")

def main():
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=2) 

    prev_error = 0
    frame_count = 0
    start_time = time.time()

    # *** Konfigurasi Tampilan (Untuk Debugging) ***
    # Setel ke True jika Anda ingin melihat jendela OpenCV (memakan CPU/GPU)
    # Setel ke False untuk mode headless (performa lebih baik)
    DISPLAY_GUI = True 
    # ==============================================

    try:
        while True:
            frame = picam2.capture_array() # Ambil frame
            
            # Memproses gambar, hanya ROI yang relevan
            # Jika DISPLAY_GUI True, maka process_image akan mengembalikan full frame binary juga
            _, _, roi_binary = process_image(frame, display_mode=DISPLAY_GUI)
            
            if roi_binary is None: # Handle jika frame terlalu kecil
                send_motor_commands(ser, 0, 0)
                if frame_count % 30 == 0:
                    print("[DEBUG] Gagal memproses frame: ROI tidak valid.")
                frame_count += 1
                continue

            line_detected, cx, cy = calculate_line_position(roi_binary)
            
            if line_detected:
                error = cx - 160 # cx adalah posisi pusat garis relatif terhadap gambar 320x240
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0: # Lebih sering print untuk monitoring
                    print(f"[DEBUG] Error: {error:4d}, Delta: {delta_error:4d}, FLC: {kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                send_motor_commands(ser, 0, 0) # Berhenti jika garis tidak terdeteksi
                if frame_count % 20 == 0:
                    print("[DEBUG] Garis tidak terdeteksi")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                cv2.line(frame_for_display, (160, 160), (160, 240), (0, 255, 0), 2)
                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
