import serial
ser = serial.Serial('/dev/serial0', 115200)
ser.write(b"halo\n")
print(ser.readline().decode().strip())
