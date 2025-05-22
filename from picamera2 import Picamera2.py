from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

def setup_fuzzy_logic():
    """
    Konfigurasi sistem fuzzy logic untuk kontrol robot - DIPERBAIKI
    """
    # Buat variabel fuzzy
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')
    
    # Definisikan membership functions - DIPERBAIKI dengan range yang lebih luas
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -60])  # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-100, -40, 0])    # Negative Small
    error['Z']  = fuzz.trimf(error.universe, [-30, 0, 30])      # Zero - diperluas
    error['PS'] = fuzz.trimf(error.universe, [0, 40, 100])     # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [60, 160, 160])   # Positive Large

    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -40])
    delta['NS'] = fuzz.trimf(delta.universe, [-60, -30, 0])
    delta['Z']  = fuzz.trimf(delta.universe, [-20, 0, 20])     # diperluas
    delta['PS'] = fuzz.trimf(delta.universe, [0, 30, 60])
    delta['PL'] = fuzz.trimf(delta.universe, [40, 100, 100])

    output['L']  = fuzz.trimf(output.universe, [-100, -100, -40])  # Left
    output['LS'] = fuzz.trimf(output.universe, [-60, -30, 0])     # Left Small
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])      # Zero - diperluas
    output['RS'] = fuzz.trimf(output.universe, [0, 30, 60])      # Right Small
    output['R']  = fuzz.trimf(output.universe, [40, 100, 100])   # Right
    
    # Definisikan rule base LENGKAP - 25 rules untuk sistem 5x5
    rules = [
        # Error NL (Negative Large)
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),   # Belok kiri keras
        ctrl.Rule(error['NL'] & delta['NS'], output['L']),   
        ctrl.Rule(error['NL'] & delta['Z'], output['L']),    
        ctrl.Rule(error['NL'] & delta['PS'], output['LS']),  
        ctrl.Rule(error['NL'] & delta['PL'], output['LS']),  
        
        # Error NS (Negative Small)
        ctrl.Rule(error['NS'] & delta['NL'], output['L']),   
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),  
        ctrl.Rule(error['NS'] & delta['Z'], output['LS']),   
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),  
        
        # Error Z (Zero) - PENTING untuk jalan lurus
        ctrl.Rule(error['Z'] & delta['NL'], output['LS']),   
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),     # LURUS - output nol
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['PL'], output['RS']),   
        
        # Error PS (Positive Small)
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),  
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   
        ctrl.Rule(error['PS'] & delta['Z'], output['RS']),   
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),  
        ctrl.Rule(error['PS'] & delta['PL'], output['R']),   
        
        # Error PL (Positive Large)
        ctrl.Rule(error['PL'] & delta['NL'], output['RS']),  
        ctrl.Rule(error['PL'] & delta['NS'], output['RS']),  
        ctrl.Rule(error['PL'] & delta['Z'], output['R']),    
        ctrl.Rule(error['PL'] & delta['PS'], output['R']),   
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),   # Belok kanan keras
    ]
    
    # Buat sistem kontrol
    control_system = ctrl.ControlSystem(rules)
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    
    return fuzzy_controller

def setup_camera():
    """
    Konfigurasi dan inisialisasi kamera
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

def setup_serial():
    """
    Konfigurasi dan inisialisasi komunikasi serial
    """
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)
        print("[UART] Port serial berhasil dibuka")
        return ser
    except Exception as e:
        print(f"[UART ERROR] Gagal membuka serial port: {e}")
        return None

def process_image(frame, use_otsu=True):
    """
    Memproses frame untuk mendeteksi jalur/garis - DIPERBAIKI
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_otsu:
        # Otsu threshold - lebih konsisten untuk kondisi pencahayaan stabil
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(f"[IMG] Menggunakan Otsu threshold")
    else:
        # Adaptive threshold - lebih baik untuk pencahayaan tidak merata
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        print(f"[IMG] Menggunakan Adaptive threshold")
    
    # ROI bagian bawah - diperluas sedikit
    roi = binary[160:240, :]
    
    return gray, binary, roi

def calculate_line_position(roi):
    """
    Menghitung posisi garis dari ROI menggunakan moments - DIPERBAIKI
    """
    # Morphological operations untuk noise reduction
    kernel = np.ones((3,3), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)
    
    M = cv2.moments(roi_clean)
    
    if M['m00'] > 100:  # Threshold minimum untuk menghindari noise
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + 160  # Offset ROI
        return True, cx, cy
    else:
        return False, 0, 0

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error):
    """
    Menghitung output kontrol berdasarkan fuzzy logic
    """
    try:
        # Batasi input dalam range yang valid
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = fuzzy_ctrl.output['output']
        
        print(f"[FLC] Error: {error_val:4d} | Delta: {delta_error:4d} | Output: {kontrol:6.2f}")
        return kontrol
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=45):  # DITURUNKAN dari 65 ke 45
    """
    Menghitung PWM untuk motor berdasarkan nilai kontrol - DIPERBAIKI
    """
    # Scaling factor untuk kontrol yang lebih halus
    kontrol_scaled = kontrol * 0.3  # DITURUNKAN dari 0.4 ke 0.3 untuk lebih smooth
    
    pwm_kiri = base_pwm - kontrol_scaled
    pwm_kanan = base_pwm + kontrol_scaled
    
    # Batasi PWM ke rentang yang lebih rendah untuk kecepatan lambat
    pwm_kiri = max(0, min(80, pwm_kiri))   # Max 80% bukan 100%
    pwm_kanan = max(0, min(80, pwm_kanan)) # Max 80% bukan 100%
    
    # Print output PWM dengan lebih detail
    print(f"[PWM] Kontrol: {kontrol:6.2f} | Kiri: {pwm_kiri:5.1f}% | Kanan: {pwm_kanan:5.1f}%")
    
    return int(pwm_kiri), int(pwm_kanan)

def send_motor_commands(ser, pwm_kiri, pwm_kanan):
    """
    Mengirim perintah ke motor melalui serial
    """
    if ser:
        try:
            cmd = f"{pwm_kiri},{pwm_kanan}\n"
            ser.write(cmd.encode())
            ser.flush()
            print(f"[UART] Sent: {cmd.strip()}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

def visualize_tracking(frame, line_detected, cx=0, cy=0, error_val=0, kontrol=0):
    """
    Membuat visualisasi tracking pada frame - DIPERBAIKI
    """
    # Gambar garis tengah referensi
    cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
    
    # Gambar ROI
    cv2.rectangle(frame, (0, 160), (320, 240), (0, 255, 255), 2)
    
    if line_detected:
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(frame, (160, cy), (cx, cy), (0, 255, 0), 2)
        
        # Status text dengan background
        status_text = f"Err:{error_val:3d} | Ctrl:{kontrol:5.1f}"
        cv2.rectangle(frame, (5, 5), (280, 35), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.rectangle(frame, (5, 5), (280, 35), (0, 0, 0), -1)
        cv2.putText(frame, "GARIS TIDAK DITEMUKAN", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    """
    Fungsi utama program
    """
    # Setup komponen
    print("Inisialisasi sistem...")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    
    prev_error = 0
    line_lost_timeout = 0
    MAX_LOST_TIME = 1.5  # Diperpendek untuk respons yang lebih cepat
    
    # Tunggu kamera stabil
    print("Menunggu kamera stabil...")
    time.sleep(2)
    
    try:
        print("Memulai loop utama...")
        while True:
            start_time = time.time()
            
            # Ambil dan proses gambar - dengan opsi threshold
            frame = picam2.capture_array()
            _, _, roi = process_image(frame, use_otsu=True)  # Gunakan Otsu threshold
            
            # Deteksi posisi garis
            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                line_lost_timeout = 0
                
                # Hitung error dan delta error
                error_val = cx - 160  # Error dari tengah frame
                delta_error = error_val - prev_error
                
                # Komputasi fuzzy
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error)
                
                # Hitung PWM motor
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                
                # Kirim ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                prev_error = error_val
                
            else:
                line_lost_timeout += 0.05
                if line_lost_timeout >= MAX_LOST_TIME:
                    send_motor_commands(ser, 0, 0)
                    print("[WARN] Garis hilang, motor dihentikan")
                else:
                    # Tetap jalan dengan PWM terakhir untuk waktu singkat
                    print("[WARN] Garis hilang, mencari...")
                    
                kontrol = 0
                error_val = 0
            
            # Visualisasi
            frame = visualize_tracking(frame, line_detected, cx, cy, error_val, kontrol)
            cv2.imshow("Line Following Robot", frame)
            cv2.imshow("ROI Binary", roi)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Kontrol frame rate - diperlambat untuk stabilitas
            elapsed = time.time() - start_time
            if elapsed < 0.1:  # 10 FPS - diperlambat dari 20 FPS
                time.sleep(0.1 - elapsed)
            
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna")
    
    finally:
        # Cleanup
        print("Membersihkan sumber daya...")
        send_motor_commands(ser, 0, 0)
        cv2.destroyAllWindows()
        picam2.stop()
        if ser:
            ser.close()
        print("Program selesai")

if __name__ == "__main__":
    main()
