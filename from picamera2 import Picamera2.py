from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime

class ErrorFilter:
    """
    Filter untuk menstabilkan error yang berfluktuasi
    """
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.error_history = []
    
    def filter_error(self, error):
        """Filter error dengan moving average"""
        self.error_history.append(error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
        
        # Return rata-rata untuk smooth error
        filtered_error = sum(self.error_history) / len(self.error_history)
        return int(filtered_error)

def setup_fuzzy_logic():
    """
    Konfigurasi sistem fuzzy logic untuk kontrol robot - DIPERBAIKI TOTAL
    """
    # Buat variabel fuzzy
    error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
    delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
    output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')
    
    # MEMBERSHIP FUNCTIONS DIPERBAIKI - lebih smooth dan stabil
    # Error - diperluas dead zone untuk stabilitas
    error['NL'] = fuzz.trimf(error.universe, [-160, -160, -80])   # Negative Large
    error['NS'] = fuzz.trimf(error.universe, [-120, -50, -10])   # Negative Small  
    error['Z']  = fuzz.trimf(error.universe, [-40, 0, 40])       # Zero - DIPERLUAS untuk stabilitas
    error['PS'] = fuzz.trimf(error.universe, [10, 50, 120])      # Positive Small
    error['PL'] = fuzz.trimf(error.universe, [80, 160, 160])     # Positive Large

    # Delta error - lebih toleran terhadap noise
    delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -50])
    delta['NS'] = fuzz.trimf(delta.universe, [-70, -25, -5])
    delta['Z']  = fuzz.trimf(delta.universe, [-20, 0, 20])       # DIPERLUAS
    delta['PS'] = fuzz.trimf(delta.universe, [5, 25, 70])
    delta['PL'] = fuzz.trimf(delta.universe, [50, 100, 100])

    # Output - range lebih kecil untuk kontrol yang lebih halus
    output['L']  = fuzz.trimf(output.universe, [-100, -100, -50])  # Left
    output['LS'] = fuzz.trimf(output.universe, [-70, -35, -10])   # Left Small
    output['Z']  = fuzz.trimf(output.universe, [-15, 0, 15])      # Zero - dead zone
    output['RS'] = fuzz.trimf(output.universe, [10, 35, 70])     # Right Small
    output['R']  = fuzz.trimf(output.universe, [50, 100, 100])   # Right
    
    # RULE BASE DIPERBAIKI - lebih konservatif dan smooth
    rules = [
        # Error NL (garis jauh di kiri) -> belok kiri tapi tidak terlalu ekstrem
        ctrl.Rule(error['NL'] & delta['NL'], output['L']),   
        ctrl.Rule(error['NL'] & delta['NS'], output['LS']),  # DIPERLEMBUT
        ctrl.Rule(error['NL'] & delta['Z'], output['LS']),   # DIPERLEMBUT
        ctrl.Rule(error['NL'] & delta['PS'], output['Z']),   # DIPERLEMBUT
        ctrl.Rule(error['NL'] & delta['PL'], output['Z']),   # DIPERLEMBUT
        
        # Error NS (garis agak di kiri) -> koreksi lembut
        ctrl.Rule(error['NS'] & delta['NL'], output['LS']),  
        ctrl.Rule(error['NS'] & delta['NS'], output['LS']),  
        ctrl.Rule(error['NS'] & delta['Z'], output['Z']),    # DIPERLEMBUT
        ctrl.Rule(error['NS'] & delta['PS'], output['Z']),   
        ctrl.Rule(error['NS'] & delta['PL'], output['RS']),  
        
        # Error Z (garis di tengah) -> prioritas LURUS
        ctrl.Rule(error['Z'] & delta['NL'], output['Z']),    # DIPERLEMBUT
        ctrl.Rule(error['Z'] & delta['NS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['Z'], output['Z']),     # LURUS PRIORITAS
        ctrl.Rule(error['Z'] & delta['PS'], output['Z']),    
        ctrl.Rule(error['Z'] & delta['PL'], output['Z']),    # DIPERLEMBUT
        
        # Error PS (garis agak di kanan) -> koreksi lembut
        ctrl.Rule(error['PS'] & delta['NL'], output['LS']),  
        ctrl.Rule(error['PS'] & delta['NS'], output['Z']),   
        ctrl.Rule(error['PS'] & delta['Z'], output['Z']),    # DIPERLEMBUT
        ctrl.Rule(error['PS'] & delta['PS'], output['RS']),  
        ctrl.Rule(error['PS'] & delta['PL'], output['RS']),  
        
        # Error PL (garis jauh di kanan) -> belok kanan tapi tidak ekstrem
        ctrl.Rule(error['PL'] & delta['NL'], output['Z']),   # DIPERLEMBUT
        ctrl.Rule(error['PL'] & delta['NS'], output['Z']),   # DIPERLEMBUT
        ctrl.Rule(error['PL'] & delta['Z'], output['RS']),   # DIPERLEMBUT
        ctrl.Rule(error['PL'] & delta['PS'], output['RS']),  # DIPERLEMBUT
        ctrl.Rule(error['PL'] & delta['PL'], output['R']),   
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
    Memproses frame untuk mendeteksi jalur/garis
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
    Menghitung posisi garis dari ROI menggunakan moments
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

def compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error, dead_zone=25):
    """
    Menghitung output kontrol berdasarkan fuzzy logic - DIPERBAIKI
    """
    try:
        # DEAD ZONE DIPERLUAS untuk stabilitas yang lebih baik
        if abs(error_val) <= dead_zone and abs(delta_error) <= 15:
            print(f"[FLC] DEAD ZONE BESAR - Error: {error_val:4d} | Delta: {delta_error:4d} | Output: 0.00 (LURUS STABIL)")
            return 0.0
        
        # Batasi input dalam range yang valid
        error_val = max(-160, min(160, error_val))
        delta_error = max(-100, min(100, delta_error))
        
        fuzzy_ctrl.input['error'] = error_val
        fuzzy_ctrl.input['delta'] = delta_error
        fuzzy_ctrl.compute()
        kontrol = fuzzy_ctrl.output['output']
        
        # SMOOTHING OUTPUT yang lebih agresif
        if abs(kontrol) < 8:  # Diperbesar threshold
            kontrol = 0.0
            
        # FILTER TAMBAHAN - cegah perubahan drastis
        kontrol = np.clip(kontrol, -80, 80)  # Batasi output maksimal
        
        print(f"[FLC] Error: {error_val:4d} | Delta: {delta_error:4d} | Output: {kontrol:6.2f}")
        return kontrol
    except Exception as e:
        print(f"[FLC ERROR] {e}")
        return 0

def calculate_motor_pwm(kontrol, base_pwm=55):
    """
    Menghitung PWM untuk motor - ALGORITMA DIPERBAIKI TOTAL
    """
    # SCALING DIPERBAIKI - jauh lebih halus dan progresif
    if abs(kontrol) <= 10:
        # Dead zone - tidak ada koreksi
        kontrol_scaled = 0
    elif abs(kontrol) <= 30:
        # Koreksi kecil - scaling minimal
        kontrol_scaled = kontrol * 0.08  # SANGAT KECIL untuk smooth
    elif abs(kontrol) <= 60:
        # Koreksi sedang - scaling bertahap
        kontrol_scaled = kontrol * 0.12
    else:
        # Koreksi besar - scaling maksimal tapi terbatas
        kontrol_scaled = kontrol * 0.15  # MAKSIMAL 15% dari base PWM
    
    # DEAD ZONE KETAT untuk PWM
    if abs(kontrol_scaled) < 1:
        pwm_kiri = base_pwm
        pwm_kanan = base_pwm
        arah = "LURUS"
        status = "DEAD_ZONE"
    else:
        # Aplikasi kontrol dengan batas yang ketat
        pwm_kiri = base_pwm + kontrol_scaled   
        pwm_kanan = base_pwm - kontrol_scaled  
        
        # Tentukan arah dan status
        if kontrol_scaled < -1:
            arah = "KIRI"
            status = f"CTRL_{abs(kontrol_scaled):.1f}"
        elif kontrol_scaled > 1:
            arah = "KANAN" 
            status = f"CTRL_{abs(kontrol_scaled):.1f}"
        else:
            arah = "LURUS"
            status = "MINIMAL"
    
    # BATASI PWM dengan range yang lebih ketat
    pwm_kiri = max(30, min(75, pwm_kiri))   # Range 30-75% untuk stabilitas
    pwm_kanan = max(30, min(75, pwm_kanan)) 
    
    print(f"[PWM] Kontrol: {kontrol:6.2f} | Scaled: {kontrol_scaled:5.2f} | {status:10s} | Kiri: {pwm_kiri:5.1f}% | Kanan: {pwm_kanan:5.1f}%")
    
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
    Membuat visualisasi tracking pada frame
    """
    # Gambar garis tengah referensi
    cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
    
    # Gambar ROI
    cv2.rectangle(frame, (0, 160), (320, 240), (0, 255, 255), 2)
    
    if line_detected:
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(frame, (160, cy), (cx, cy), (0, 255, 0), 2)
        
        # Status text dengan background - DIPERBAIKI untuk tampilan yang lebih baik
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
    Fungsi utama program - DIPERBAIKI dengan stabilitas maksimal
    """
    # Setup komponen
    print("=== Inisialisasi Sistem Line Following Robot ===")
    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3)  # Filter untuk smooth error
    
    # Variabel untuk kontrol yang stabil
    prev_error = 0
    prev_kontrol = 0  # TAMBAHAN: simpan kontrol sebelumnya untuk smoothing
    line_lost_timeout = 0
    MAX_LOST_TIME = 1.5
    
    # Statistik untuk monitoring
    frame_count = 0
    start_program = time.time()
    
    # Tunggu kamera stabil
    print("Menunggu kamera stabil...")
    time.sleep(2)
    
    try:
        print("=== Memulai Loop Utama ===")
        while True:
            start_time = time.time()
            frame_count += 1
            
            # Ambil dan proses gambar
            frame = picam2.capture_array()
            _, binary, roi = process_image(frame, use_otsu=True)
            
            # Deteksi posisi garis
            line_detected, cx, cy = calculate_line_position(roi)
            
            if line_detected:
                line_lost_timeout = 0
                
                # Hitung error RAW dan filter untuk stabilitas
                raw_error = cx - 160
                error_val = error_filter.filter_error(raw_error)  # ERROR DIFILTER
                delta_error = error_val - prev_error
                
                # Debug info untuk error filtering
                if frame_count % 20 == 0:  # Print setiap 20 frame
                    print(f"[DEBUG] Raw Error: {raw_error:4d} | Filtered: {error_val:4d} | Delta: {delta_error:4d}")
                
                # Komputasi fuzzy yang diperbaiki
                kontrol = compute_fuzzy_control(fuzzy_ctrl, error_val, delta_error, dead_zone=25)
                
                # SMOOTHING kontrol - cegah perubahan mendadak yang menyebabkan oscillation
                if abs(kontrol - prev_kontrol) > 20:  # Jika perubahan terlalu besar
                    kontrol_smooth = prev_kontrol + np.sign(kontrol - prev_kontrol) * 10  # Batasi perubahan
                    print(f"[SMOOTH] Kontrol di-smooth dari {prev_kontrol:.1f} ke {kontrol_smooth:.1f} (original: {kontrol:.1f})")
                    kontrol = kontrol_smooth
                
                # Hitung PWM dengan algoritma baru
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol, base_pwm=55)
                
                # Kirim ke motor
                send_motor_commands(ser, pwm_kiri, pwm_kanan)
                
                # Update history
                prev_error = error_val
                prev_kontrol = kontrol
                
            else:
                line_lost_timeout += 0.05
                if line_lost_timeout >= MAX_LOST_TIME:
                    send_motor_commands(ser, 0, 0)
                    print("[WARN] Garis hilang lama, motor dihentikan")
                else:
                    # Tetap jalan dengan PWM terakhir untuk waktu singkat
                    print(f"[WARN] Garis hilang {line_lost_timeout:.1f}s, mencari...")
                    
                kontrol = 0
                error_val = 0
            
            # Visualisasi dengan informasi lebih lengkap
            frame = visualize_tracking(frame, line_detected, cx, cy, error_val, kontrol)
            
            # Tambahkan info frame rate
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_program)
                cv2.putText(frame, f"FPS: {fps:.1f}", (5, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Tampilkan frame
            cv2.imshow("Line Following Robot - Improved", frame)
            cv2.imshow("ROI Binary", roi)
            
            # Kontrol exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n=== Program dihentikan oleh pengguna ===")
                break
            elif key == ord('r'):
                print("[INFO] Reset error filter")
                error_filter = ErrorFilter(window_size=3)
                prev_error = 0
                prev_kontrol = 0
            
            # Kontrol frame rate - 10 FPS untuk stabilitas maksimal
            elapsed = time.time() - start_time
            target_fps = 0.1  # 10 FPS
            if elapsed < target_fps:
                time.sleep(target_fps - elapsed)
            
            # Status report berkala
            if frame_count % 100 == 0:
                uptime = time.time() - start_program
                avg_fps = frame_count / uptime
                print(f"[STATUS] Frame: {frame_count} | Uptime: {uptime:.1f}s | Avg FPS: {avg_fps:.1f}")
            
    except KeyboardInterrupt:
        print("\n=== Program dihentikan dengan Ctrl+C ===")
    
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
    
    finally:
        # Cleanup yang lebih aman
        print("=== Membersihkan Sumber Daya ===")
        try:
            send_motor_commands(ser, 0, 0)
            time.sleep(0.5)  # Pastikan command terkirim
            print("[CLEANUP] Motor dihentikan")
        except:
            pass
            
        try:
            cv2.destroyAllWindows()
            print("[CLEANUP] Window ditutup")
        except:
            pass
            
        try:
            picam2.stop()
            print("[CLEANUP] Kamera dihentikan")
        except:
            pass
            
        try:
            if ser:
                ser.close()
                print("[CLEANUP] Serial port ditutup")
        except:
            pass
            
        # Statistik akhir
        total_time = time.time() - start_program
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"[STATS] Total Frame: {frame_count} | Runtime: {total_time:.1f}s | Avg FPS: {avg_fps:.1f}")
        
        print("=== Program Selesai ===")

if __name__ == "__main__":
    main()
