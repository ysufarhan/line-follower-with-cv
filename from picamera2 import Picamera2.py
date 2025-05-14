from picamera2 import Picamera2
import cv2
import numpy as np
import time
import serial
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# === Fuzzy Logic Setup ===

error = ctrl.Antecedent(np.arange(-160, 161, 1), 'error')
delta = ctrl.Antecedent(np.arange(-100, 101, 1), 'delta')
output = ctrl.Consequent(np.arange(-100, 101, 1), 'output')

error['NL'] = fuzz.trimf(error.universe, [-160, -160, -80])
error['NS'] = fuzz.trimf(error.universe, [-160, -80, 0])
error['Z']  = fuzz.trimf(error.universe, [-40, 0, 40])
error['PS'] = fuzz.trimf(error.universe, [0, 80, 160])
error['PL'] = fuzz.trimf(error.universe, [80, 160, 160])

delta['NL'] = fuzz.trimf(delta.universe, [-100, -100, -50])
delta['NS'] = fuzz.trimf(delta.universe, [-100, -50, 0])
delta['Z']  = fuzz.trimf(delta.universe, [-20, 0, 20])
delta['PS'] = fuzz.trimf(delta.universe, [0, 50, 100])
delta['PL'] = fuzz.trimf(delta.universe, [50, 100, 100])

output['L']  = fuzz.trimf(output.universe, [-100, -100, -50])
output['LS'] = fuzz.trimf(output.universe, [-100, -50, 0])
output['Z']  = fuzz.trimf(output.universe, [-20, 0, 20])
output['RS'] = fuzz.trimf(output.universe, [0, 50, 100])
output['R']  = fuzz.trimf(output.universe, [50, 100, 100])

rules = [
    ctrl.Rule(error['NL'] & delta['NL'], output['L']),
    ctrl.Rule(error['NL'] & delta['Z'], output['L']),
    ctrl.Rule(error['NS'] & delta['NS'], output['LS']),
    ctrl.Rule(error['Z'] & delta['Z'], output['Z']),
    ctrl.Rule(error['PS'] & delta['PS'], output['RS']),
    ctrl.Rule(error['PL'] & delta['Z'], output['R']),
    ctrl.Rule(error['PL'] & delta['PL'], output['R']),
    ctrl.Rule(error['Z'] & delta['NL'], output['LS']),
    ctrl.Rule(error['Z'] & delta['PL'], output['RS']),
]

control_system = ctrl.ControlSystem(rules)
fuzzy_ctrl = ctrl.ControlSystemSimulation(control_system)

# === Kamera dan Serial Setup ===

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

try:
    ser = serial.Serial('/dev/serial0', 115200)
except Exception as e:
    print(f"[UART ERROR] Gagal membuka serial port: {e}")
    ser = None

prev_error = 0

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ROI bagian bawah (tempat jalur biasanya terlihat)
        roi = binary[180:240, :]
        M = cv2.moments(roi)

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) + 180
            error_val = cx - 160
            delta_error = error_val - prev_error
            kontrol = 0

            try:
                fuzzy_ctrl.input['error'] = error_val
                fuzzy_ctrl.input['delta'] = delta_error
                fuzzy_ctrl.compute()
                kontrol = int(fuzzy_ctrl.output['output'])

                # Tambahkan log ke terminal
                print(f"[LOG] Error: {error_val} | Delta: {delta_error} | Output Fuzzy: {kontrol}")

            except Exception as e:
                print(f"[FLC ERROR] {e}")
                kontrol = 0

            # Koreksi PWM motor kiri-kanan
            base_pwm = 60  # kecepatan dasar
            pwm_kiri = base_pwm - kontrol
            pwm_kanan = base_pwm + kontrol

            # Batasi PWM ke 0–100
            pwm_kiri = max(0, min(100, pwm_kiri))
            pwm_kanan = max(0, min(100, pwm_kanan))

            # Kirim ke ESP32 (misal format: "60,70\n")
            if ser:
                try:
                    ser.write(f"{int(pwm_kiri)},{int(pwm_kanan)}\n".encode())
                except Exception as e:
                    print(f"[SERIAL WRITE ERROR] {e}")

            prev_error = error_val

            # Visualisasi
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
            cv2.putText(frame, f"Err:{error_val} | Ctrl:{kontrol}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Garis tidak terdeteksi
            cv2.putText(frame, "Garis tidak ditemukan", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if ser:
                ser.write("0,0\n".encode())  # berhenti

        # Tampilkan hasil
        cv2.imshow("Deteksi Garis", frame)
        cv2.imshow("ROI", roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Dihentikan oleh pengguna")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    if ser:
        ser.close()
