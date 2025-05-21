import serial
import time

# Buka koneksi serial ke ESP32
# Ganti '/dev/ttyUSB0' sesuai dengan port ESP32 Anda
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Tunggu ESP32 reset setelah koneksi

try:
    while True:
        # Kirim PWM 100 untuk kiri dan kanan
        ser.write(b"100,100\n")
        print("Sent: 100,100")
        time.sleep(5)  # Jeda 5 detik

        # Kirim PWM 50 untuk kiri dan kanan
        ser.write(b"50,50\n")
        print("Sent: 50,50")
        time.sleep(5)  # Jeda 5 detik

except KeyboardInterrupt:
    print("Program dihentikan oleh pengguna.")

finally:
    ser.close()
