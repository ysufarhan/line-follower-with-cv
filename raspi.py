#!/usr/bin/env python3
import serial
import time
import os

# Cek port serial yang tersedia
def list_serial_ports():
    """Mencari port serial yang tersedia di Raspberry Pi"""
    available_ports = []
    for port in ['/dev/ttyS0', '/dev/ttyAMA0', '/dev/ttyUSB0', '/dev/ttyACM0']:
        if os.path.exists(port):
            available_ports.append(port)
    return available_ports

print("Port serial yang tersedia:", list_serial_ports())

# Konfigurasi port serial
# Jika menggunakan USB-to-Serial converter, mungkin portnya '/dev/ttyUSB0'
# Pastikan untuk mengaktifkan UART di raspi-config dengan: sudo raspi-config
port_to_use = '/dev/ttyAMA0'  # Coba ganti dengan port yang tersedia
print(f"Menggunakan port: {port_to_use}")

ser = serial.Serial(
    port=port_to_use,
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

def send_data(data):
    """Mengirim data ke ESP32"""
    ser.write((data + '\n').encode('utf-8'))  # Tambahkan newline
    print(f"Data terkirim: {data}")
    
def receive_data():
    """Menerima data dari ESP32"""
    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').rstrip()
            print(f"Data diterima: {data}")
            return data
        except UnicodeDecodeError:
            raw_data = ser.readline()
            print(f"Data mentah diterima (hex): {raw_data.hex()}")
            return None
    return None

try:
    print("Komunikasi UART Raspberry Pi dengan ESP32")
    print("Tekan Ctrl+C untuk keluar")
    
    # Bersihkan buffer serial terlebih dahulu
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    counter = 0
    while True:
        # Mengirim pesan dengan counter
        message = f"Pesan dari Raspberry Pi #{counter}"
        send_data(message)
        
        # Menunggu respons dengan timeout yang lebih lama
        timeout_counter = 0
        response = None
        while response is None and timeout_counter < 10:
            response = receive_data()
            if response is None:
                print("Menunggu respons...")
                timeout_counter += 1
                time.sleep(0.2)
        
        if response is None:
            print("Tidak ada respons dari ESP32!")
        
        # Menunggu sebelum mengirim pesan berikutnya
        time.sleep(1.5)
        counter += 1
        
except KeyboardInterrupt:
    print("Program dihentikan")
finally:
    ser.close()
    print("Port serial ditutup")