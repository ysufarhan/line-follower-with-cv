#include <Arduino.h>
#include <Wire.h>          // Diperlukan untuk komunikasi I2C
#include <Adafruit_VL53L0X.h> // Library untuk sensor VL53L0X

// --- Definisi Pin Motor ---
// Sesuaikan pin-pin ini dengan koneksi hardware Anda pada ESP32
#define ENA 14     // PWM Motor Kiri (Enable A)
#define IN1 27     // Input 1 Motor Kiri
#define IN2 26     // Input 2 Motor Kiri
#define IN3 25     // Input 3 Motor Kanan
#define IN4 33     // Input 4 Motor Kanan
#define ENB 32     // PWM Motor Kanan (Enable B)

// --- Variabel untuk Parsing Serial ---
String inputString = "";        // String untuk menyimpan data serial yang masuk
bool stringComplete = false;    // Flag yang menandakan string lengkap (diakhiri newline)

// --- Variabel Watchdog Timer ---
// Berfungsi untuk menghentikan motor secara otomatis jika tidak ada perintah yang diterima dari Pi
unsigned long lastCommandTime = 0;      // Waktu terakhir perintah diterima (dalam ms)
const unsigned long WATCHDOG_TIMEOUT = 500; // Timeout watchdog dalam milidetik (misal: 500ms = 0.5 detik)

// --- Konfigurasi Sensor VL53L0X ---
Adafruit_VL53L0X lox = Adafruit_VL53L0X(); // Membuat objek sensor VL53L0X
const int OBSTACLE_DISTANCE_CM = 5;       // Jarak ambang batas (dalam cm) untuk berhenti
const int VL53L0X_READ_INTERVAL = 100;    // Interval membaca sensor (ms) untuk menghindari overloading I2C bus
unsigned long lastVL53L0XReadTime = 0;    // Waktu terakhir pembacaan sensor VL53L0X

// --- Variabel State ---
bool obstacleDetected = false; // Flag untuk menandakan apakah ada halangan

// --- Fungsi Setup Arduino (Dijalankan sekali saat booting) ---
void setup() {
  // Inisialisasi komunikasi Serial pada Baud Rate 115200
  Serial.begin(115200);
  inputString.reserve(50); // Mengalokasikan memori untuk string input, agar lebih efisien

  // Setup pin motor
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Setup PWM Channels
  ledcSetup(0, 5000, 8);  // Channel 0 untuk ENA (PWM Motor Kiri)
  ledcSetup(1, 5000, 8);  // Channel 1 untuk ENB (PWM Motor Kanan)
  ledcAttachPin(ENA, 0);
  ledcAttachPin(ENB, 1);

  // Menghentikan motor saat startup
  stopMotors();
  lastCommandTime = millis(); // Menginisialisasi waktu terakhir perintah (untuk watchdog)

  Serial.println("ESP32 Motor Controller Siap");

  // --- Inisialisasi Sensor VL53L0X ---
  Wire.begin(); // Memulai komunikasi I2C
  Serial.println("Memulai sensor VL53L0X...");
  if (!lox.begin()) {
    Serial.println("Gagal menemukan sensor VL53L0X. Pastikan wiring benar!");
    while(1); // Berhenti di sini jika sensor tidak terdeteksi
  }
  Serial.println("Sensor VL53L0X terdeteksi!");

  // Opsional: Sesuaikan mode rentang dan presisi sensor
  // lox.setMeasurementTimingBudget(20000); // Waktu pengukuran (mikrodetik). Lebih lama = lebih akurat
}

// --- Fungsi Loop Arduino (Dijalankan berulang-ulang) ---
void loop() {
  // --- Baca Data Serial dan Kontrol Motor ---
  if (stringComplete) {
    int commaIndex = inputString.indexOf(',');
    if (commaIndex > 0) {
      String pwmLeftStr = inputString.substring(0, commaIndex);
      String pwmRightStr = inputString.substring(commaIndex + 1);

      if (pwmLeftStr.length() > 0 && pwmRightStr.length() > 0) {
        int pwmLeft = pwmLeftStr.toInt();
        int pwmRight = pwmRightStr.toInt();
        
        // Hanya set motor jika tidak ada halangan
        if (!obstacleDetected) {
          setMotors(pwmLeft, pwmRight);
        } else {
          stopMotors(); // Pastikan motor berhenti jika ada halangan
          // Serial.println("Obstacle detected, motors stopped."); // Debug
        }
        
        lastCommandTime = millis(); // Reset watchdog timer
        
        // Debug output (opsional, bisa dihapus untuk produksi)
        Serial.print("Diterima - Kiri: ");
        Serial.print(pwmLeft);
        Serial.print(", Kanan: ");
        Serial.print(pwmRight);
        Serial.print(" | Halangan: ");
        Serial.println(obstacleDetected ? "YA" : "TIDAK");
      } else {
        Serial.println("Error: Nilai PWM kosong diterima.");
      }
    } else {
      Serial.println("Error: Format serial tidak valid (tidak ada koma).");
    }
    
    inputString = "";
    stringComplete = false;
  }

  // --- Watchdog Timer ---
  if (millis() - lastCommandTime > WATCHDOG_TIMEOUT) {
    stopMotors();
    static bool watchdogActive = false;
    if (!watchdogActive) {
      Serial.println("Watchdog: Tidak ada perintah, motor dihentikan.");
      watchdogActive = true;
    }
  } else {
    static bool watchdogActive = false; 
    if (watchdogActive) { 
        // Serial.println("Watchdog: Perintah kembali aktif.");
        watchdogActive = false;
    }
  }

  // --- Pembacaan Sensor VL53L0X ---
  if (millis() - lastVL53L0XReadTime > VL53L0X_READ_INTERVAL) {
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false); // Baca pengukuran tanpa menampilkan output default

    if (measure.RangeStatus != 4) {  // Status 4 berarti "out of range" atau "signal failed"
      // Konversi jarak dari mm ke cm
      int distanceCm = measure.RangeMilliMeter / 10;
      Serial.print("Jarak: ");
      Serial.print(distanceCm);
      Serial.println(" cm");

      if (distanceCm > 0 && distanceCm <= OBSTACLE_DISTANCE_CM) {
        obstacleDetected = true;
        stopMotors(); // Hentikan motor segera jika ada halangan
        // Serial.println("!!! HALANGAN TERDETEKSI - MOTOR DIHENTIKAN !!!");
      } else {
        obstacleDetected = false;
      }
    } else {
      // Jika sensor tidak mendapatkan pengukuran valid (misal terlalu jauh atau terlalu dekat)
      // Kita asumsikan tidak ada halangan (atau diluar jangkauan deteksi ambang batas)
      obstacleDetected = false;
      // Serial.println("Jarak: Di luar jangkauan/Invalid");
    }
    lastVL53L0XReadTime = millis();
  }
}

// --- Fungsi untuk Menggerakkan Motor ---
// Menerima nilai PWM dari -100 (mundur) hingga 100 (maju)
void setMotors(int pwmLeft, int pwmRight) {
  // Hanya gerakkan motor jika tidak ada halangan
  if (obstacleDetected) {
    stopMotors();
    return; // Keluar dari fungsi setMotors
  }

  // Kontrol Motor Kiri
  if (pwmLeft > 0) { // Jika PWM positif, motor maju
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    ledcWrite(0, map(pwmLeft, 0, 100, 0, 255));
  } else if (pwmLeft < 0) { // Jika PWM negatif, motor mundur
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    ledcWrite(0, map(abs(pwmLeft), 0, 100, 0, 255));
  } else { // Jika PWM nol, motor diam
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    ledcWrite(0, 0);
  }

  // Kontrol Motor Kanan
  if (pwmRight > 0) { // Maju
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    ledcWrite(1, map(pwmRight, 0, 100, 0, 255));
  } else if (pwmRight < 0) { // Mundur
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    ledcWrite(1, map(abs(pwmRight), 0, 100, 0, 255));
  } else { // Diam
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    ledcWrite(1, 0);
  }
}

// --- Fungsi untuk Menghentikan Motor ---
void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(0, 0); // Atur PWM ke nol
  ledcWrite(1, 0);
}

// --- Fungsi serialEvent() ---
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
