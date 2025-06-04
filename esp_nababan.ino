#include <Arduino.h>

// Pin Motor sesuai konfigurasi Anda pada driver L298N
#define ENA 14    // PWM Motor Kiri (Enable A)
#define IN1 27    // Motor Kiri Input 1
#define IN2 26    // Motor Kiri Input 2
#define IN3 25    // Motor Kanan Input 1
#define IN4 33    // Motor Kanan Input 2
#define ENB 32    // PWM Motor Kanan (Enable B)

// Variabel untuk parsing serial
String inputString = "";
bool stringComplete = false;

void setup() {
  // Inisialisasi Serial untuk komunikasi dengan Raspberry Pi dan debug
  Serial.begin(115200);
  inputString.reserve(50); // Alokasi memori untuk string input serial
  
  // Setup pin motor sebagai OUTPUT
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
  
  // Setup PWM Channels untuk ESP32 (LEDC)
  // ledcSetup(channel, freq, resolution_bits)
  ledcSetup(0, 5000, 8);  // Channel 0, Frekuensi 5 KHz, Resolusi 8 bit (0-255) untuk ENA (Motor Kiri)
  ledcSetup(1, 5000, 8);  // Channel 1, Frekuensi 5 KHz, Resolusi 8 bit (0-255) untuk ENB (Motor Kanan)
  
  // Kaitkan pin PWM ke channel yang sudah disetup
  ledcAttachPin(ENA, 0);
  ledcAttachPin(ENB, 1);
  
  // Pastikan motor berhenti saat startup
  stopMotors();
  
  Serial.println("ESP32 Motor Controller Ready");
}

void loop() {
  // Periksa apakah string serial sudah lengkap (diakhiri '\n')
  if (stringComplete) {
    // Cari koma untuk memisahkan nilai PWM kiri dan kanan
    int commaIndex = inputString.indexOf(',');
    if (commaIndex > 0) {
      // Ekstrak dan konversi nilai PWM ke integer
      int pwmLeft = inputString.substring(0, commaIndex).toInt();
      int pwmRight = inputString.substring(commaIndex + 1).toInt();
      
      // Kontrol motor berdasarkan nilai PWM yang diterima
      setMotors(pwmLeft, pwmRight);
      
      // Debug output ke Serial Monitor ESP32 (bisa juga dilihat dari Raspberry Pi jika terhubung)
      Serial.print("Received - Left: ");
      Serial.print(pwmLeft);
      Serial.print(", Right: ");
      Serial.println(pwmRight);
    } else {
        Serial.println("Invalid serial format received.");
    }
    
    // Reset string dan flag untuk menerima perintah berikutnya
    inputString = "";
    stringComplete = false;
  }
}

// Fungsi untuk menggerakkan motor
// Menerima nilai PWM -100 hingga 100 dari Raspberry Pi
void setMotors(int pwmLeft, int pwmRight) {
  // --- Motor Kiri ---
  if (pwmLeft > 0) {
    // Maju
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    // Map PWM dari 0-100 ke 0-255 untuk ledcWrite
    ledcWrite(0, map(pwmLeft, 0, 100, 0, 255)); 
  } else if (pwmLeft < 0) {
    // Mundur
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    // Gunakan nilai absolut PWM untuk kecepatan, map dari 0-100 ke 0-255
    ledcWrite(0, map(abs(pwmLeft), 0, 100, 0, 255));
  } else {
    // Berhenti
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    ledcWrite(0, 0);
  }
  
  // --- Motor Kanan ---
  if (pwmRight > 0) {
    // Maju
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    ledcWrite(1, map(pwmRight, 0, 100, 0, 255));
  } else if (pwmRight < 0) {
    // Mundur
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    ledcWrite(1, map(abs(pwmRight), 0, 100, 0, 255));
  } else {
    // Berhenti
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    ledcWrite(1, 0);
  }
}

// Fungsi untuk menghentikan semua motor
void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(0, 0); // Matikan PWM channel 0
  ledcWrite(1, 0); // Matikan PWM channel 1
}

// Fungsi untuk menangani data serial yang masuk
// Ini adalah fungsi callback yang dipanggil otomatis saat ada data serial tersedia
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    
    // Jika karakter yang diterima adalah newline, berarti string sudah lengkap
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
