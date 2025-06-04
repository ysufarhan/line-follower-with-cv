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
        return int(sum(self.error_history) / len(self.error_history))

# --- Kelas untuk Pemulihan Garis (Diperbarui untuk Hold Last Direction) ---
class LineRecovery:
    def __init__(self):
        self.lost_count = 0 # Menghitung berapa frame garis hilang
        self.last_valid_error_direction = 0 # Menyimpan arah error terakhir (+:kanan, -:kiri)
        self.last_valid_pwm_kiri = 0 # Menyimpan PWM kiri terakhir
        self.last_valid_pwm_kanan = 0 # Menyimpan PWM kanan terakhir
        self.search_speed = 40 # Kecepatan PWM saat berputar mencari garis
        self.hold_direction_factor = 0.5 # Faktor untuk seberapa agresif belok saat hold direction
        self.hold_speed_base_pwm = 40 # Kecepatan dasar saat hold direction (maju)

    def handle_line_lost(self, ser_instance):
        self.lost_count += 1
        
        # --- Strategi Prediksi/Hold Last Direction ---
        # Tahap 1: Lanjutkan dengan arah terakhir yang diketahui (maju sambil belok)
        if self.lost_count < 15: # DIUBAH: Coba maju terus selama 15 frame sambil belok
            # Gunakan PWM terakhir jika memungkinkan, atau terapkan belok halus
            if self.last_valid_error_direction > 0: # Terakhir belok kanan/garis di kanan
                pwm_kiri_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
            elif self.last_valid_error_direction < 0: # Terakhir belok kiri/garis di kiri
                pwm_kiri_rec = self.hold_speed_base_pwm - (self.hold_direction_factor * abs(self.last_valid_error_direction))
                pwm_kanan_rec = self.hold_speed_base_pwm + (self.hold_direction_factor * abs(self.last_valid_error_direction))
            else: # Terakhir lurus, coba lurus terus
                pwm_kiri_rec = self.hold_speed_base_pwm
                pwm_kanan_rec = self.hold_speed_base_pwm
            
            # Pastikan PWM tidak keluar batas saat recovery
            pwm_kiri_rec = max(20, min(85, pwm_kiri_rec))
            pwm_kanan_rec = max(20, min(85, pwm_kanan_rec))

            send_motor_commands(ser_instance, int(pwm_kiri_rec), int(pwm_kanan_rec))
            return f"HOLD_DIR ({int(pwm_kiri_rec)},{int(pwm_kanan_rec)})"
        
        # Tahap 2: Jika masih hilang, beralih ke strategi putar di tempat (fallback)
        else:
            if self.last_valid_error_direction > 0: # Garis terakhir di kanan, putar ke kanan (motor kiri maju, kanan mundur)
                send_motor_commands(ser_instance, self.search_speed, -self.search_speed)
                return "SEARCH_RIGHT_SPIN"
            else: # Garis terakhir di kiri, putar ke kiri (motor kiri mundur, kanan maju)
                send_motor_commands(ser_instance, -self.search_speed, self.search_speed)
                return "SEARCH_LEFT_SPIN"
        
    def line_found(self, current_error):
        self.lost_count = 0
        # Simpan arah error terakhir yang signifikan
        if abs(current_error) > 5:
            self.last_valid_error_direction = current_error 
        else: # Jika error sangat kecil, anggap arahnya lurus
            self.last_valid_error_direction = 0 
        # Tidak perlu menyimpan last_valid_pwm_kiri/kanan di sini, cukup arah error.

# ... (Sisa kode setup_fuzzy_logic, setup_camera, setup_serial, process_image, calculate_line_position, compute_fuzzy_control, calculate_motor_pwm, send_motor_commands) tetap sama seperti program terakhir Anda) ...

# --- Fungsi Utama Program ---
def main():
    global manual_threshold_value 

    fuzzy_ctrl = setup_fuzzy_logic()
    picam2 = setup_camera()
    ser = setup_serial()
    error_filter = ErrorFilter(window_size=3) 
    line_recovery_handler = LineRecovery() 

    prev_error = 0
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
            
            if roi_binary is None:
                recovery_action = line_recovery_handler.handle_line_lost(ser) # Panggil recovery saat frame invalid
                prev_error = 0 # Reset error
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Gagal memproses frame: ROI tidak valid. Aksi pemulihan: {recovery_action}")
                frame_count += 1
                
                # Menampilkan visualisasi saat frame error
                if DISPLAY_GUI:
                    frame_for_display = frame.copy()
                    cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                    cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2)
                    flc_error_z_boundary = 15
                    cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                    
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"ACTION: {recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame_for_display, f"Thresh: {manual_threshold_value}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow("Camera View", frame_for_display)
                    if binary_full is not None:
                        cv2.imshow("Threshold ROI", binary_full)
                    else:
                        try:
                            cv2.imshow("Threshold ROI", np.zeros((roi_end_y - roi_start_y, width), dtype=np.uint8))
                        except Exception as e:
                            print(f"[VISUALIZATION ERROR] Could not display dummy ROI: {e}")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue 
            
            line_detected, cx, cy = calculate_line_position(roi_binary, roi_start_y)
            
            if line_detected:
                line_recovery_handler.line_found(cx - center_x_frame) # Reset recovery state & simpan arah
                
                error = cx - center_x_frame 
                error = error_filter.filter_error(error)
                delta_error = error - prev_error
                prev_error = error

                kontrol = compute_fuzzy_control(fuzzy_ctrl, error, delta_error)
                pwm_kiri, pwm_kanan = calculate_motor_pwm(kontrol)
                send_motor_commands(ser, pwm_kiri, pwm_kanan)

                if frame_count % 10 == 0:
                    print(f"[DEBUG] Err:{error:4d}, ΔErr:{delta_error:3d}, FLC:{kontrol:6.2f}, PWM: L{pwm_kiri} R{pwm_kanan}")
            else:
                # DIUBAH: Panggil LineRecovery saat garis tidak terdeteksi
                recovery_action = line_recovery_handler.handle_line_lost(ser)
                prev_error = 0 # Reset error untuk mencegah lonjakan besar saat garis ditemukan
                if frame_count % 20 == 0:
                    print(f"[DEBUG] Garis tidak terdeteksi. Aksi pemulihan: {recovery_action}")

            # --- Bagian Tampilan (Hanya aktif jika DISPLAY_GUI = True) ---
            if DISPLAY_GUI:
                frame_for_display = frame.copy()
                
                # Garis tengah acuan (hijau)
                cv2.line(frame_for_display, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
                
                # Gambar kotak ROI (biru)
                cv2.rectangle(frame_for_display, (0, roi_start_y), (width, roi_end_y), (255, 0, 0), 2) 

                # Garis bantu indikasi belok (kuning)
                flc_error_z_boundary = 15 
                cv2.line(frame_for_display, (width // 2 - flc_error_z_boundary, roi_start_y), (width // 2 - flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)
                cv2.line(frame_for_display, (width // 2 + flc_error_z_boundary, roi_start_y), (width // 2 + flc_error_z_boundary, roi_end_y), (0, 255, 255), 1)

                if line_detected:
                    cv2.circle(frame_for_display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_for_display, f"E:{error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: # Tampilkan status pemulihan jika garis hilang
                    cv2.putText(frame_for_display, f"LOST: {line_recovery_handler.lost_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # current_recovery_action sudah didapatkan dari panggilan handle_line_lost di atas
                    cv2.putText(frame_for_display, f"ACTION: {recovery_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Gunakan 'recovery_action' yang sudah ada


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
