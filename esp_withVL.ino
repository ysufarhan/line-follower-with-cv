#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>

// Pin Motor sesuai konfigurasi Anda
#define ENA 14   // PWM Motor Kiri
#define IN1 27   // Motor Kiri
#define IN2 26   // Motor Kiri
#define IN3 25   // Motor Kanan
#define IN4 33   // Motor Kanan
#define ENB 32   // PWM Motor Kanan

// Pin I2C untuk VL53L0X (default ESP32)
#define SDA_PIN 21
#define SCL_PIN 22

// Obstacle detection parameters
#define OBSTACLE_DISTANCE 20   // Jarak deteksi obstacle dalam mm (2cm)
#define SENSOR_TIMEOUT 500     // Timeout sensor dalam ms

// Variabel untuk parsing serial
String inputString = "";
bool stringComplete = false;

// Variabel untuk motor control dan obstacle
int requestedPwmLeft = 0;
int requestedPwmRight = 0;
bool obstacleDetected = false;

// VL53L0X sensor object
VL53L0X sensor;
bool sensorInitialized = false;

void setup() {
  // Inisialisasi Serial
  Serial.begin(115200);
  inputString.reserve(50);
  
  // Setup pin motor
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
  
  // Setup PWM
  ledcSetup(0, 5000, 8);  // Channel 0 untuk ENA
  ledcSetup(1, 5000, 8);  // Channel 1 untuk ENB
  ledcAttachPin(ENA, 0);
  ledcAttachPin(ENB, 1);
  
  // Motor diam awal
  stopMotors();
  
  // Inisialisasi I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  
  // Inisialisasi VL53L0X
  Serial.println("Initializing VL53L0X sensor...");
  sensor.setTimeout(SENSOR_TIMEOUT);
  
  if (!sensor.init()) {
    Serial.println("WARNING: VL53L0X sensor not found! Robot will work without obstacle detection.");
    sensorInitialized = false;
  } else {
    Serial.println("VL53L0X sensor initialized successfully");
    sensor.setMeasurementTimingBudget(20000); // 20ms untuk update cepat
    sensor.startContinuous();
    sensorInitialized = true;
  }
  
  Serial.println("ESP32 Motor Controller Ready - Obstacle Detection at 2cm");
}

void loop() {
  // Baca sensor jarak jika tersedia
  if (sensorInitialized) {
    checkObstacle();
  }
  
  // Baca data serial dari Raspberry Pi
  if (stringComplete) {
    // Parse data (format: "pwm_kiri,pwm_kanan")
    int commaIndex = inputString.indexOf(',');
    if (commaIndex > 0) {
      requestedPwmLeft = inputString.substring(0, commaIndex).toInt();
      requestedPwmRight = inputString.substring(commaIndex + 1).toInt();
      
      // Kontrol motor berdasarkan status obstacle
      if (obstacleDetected) {
        // Jika ada obstacle, stop motor dan kirim status ke Raspi
        setMotors(0, 0);
        Serial.print("OBSTACLE_DETECTED:");
        Serial.print(requestedPwmLeft);
        Serial.print(",");
        Serial.println(requestedPwmRight);
      } else {
        // Jika tidak ada obstacle, jalankan motor normal
        setMotors(requestedPwmLeft, requestedPwmRight);
        Serial.print("NORMAL:");
        Serial.print(requestedPwmLeft);
        Serial.print(",");
        Serial.println(requestedPwmRight);
      }
    }
    
    // Reset string
    inputString = "";
    stringComplete = false;
  }
  
  delay(10); // Small delay untuk stabilitas
}

// Fungsi untuk mengecek obstacle
void checkObstacle() {
  static unsigned long lastSensorRead = 0;
  const unsigned long sensorInterval = 30; // Baca sensor setiap 30ms
  
  if (millis() - lastSensorRead >= sensorInterval) {
    uint16_t distance = sensor.readRangeContinuousMillimeters();
    
    if (sensor.timeoutOccurred()) {
      // Jika sensor timeout, anggap tidak ada obstacle
      obstacleDetected = false;
    } else if (distance <= OBSTACLE_DISTANCE && distance > 0) {
      // Obstacle terdeteksi pada jarak 2cm atau kurang
      if (!obstacleDetected) {
        Serial.print("OBSTACLE_ALERT:");
        Serial.print(distance);
        Serial.println("mm");
        obstacleDetected = true;
      }
    } else {
      // Tidak ada obstacle
      if (obstacleDetected) {
        Serial.println("OBSTACLE_CLEARED");
        obstacleDetected = false;
      }
    }
    
    lastSensorRead = millis();
    
    // Debug output jarak sensor (berkala)
    static unsigned long lastDistanceOutput = 0;
    if (millis() - lastDistanceOutput >= 2000) { // Setiap 2 detik
      Serial.print("DISTANCE:");
      Serial.print(distance);
      Serial.println("mm");
      lastDistanceOutput = millis();
    }
  }
}

// Fungsi untuk menggerakkan motor
void setMotors(int pwmLeft, int pwmRight) {
  // Motor Kiri
  if (pwmLeft > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    ledcWrite(0, map(pwmLeft, 0, 100, 0, 255));
  } else if (pwmLeft < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    ledcWrite(0, map(abs(pwmLeft), 0, 100, 0, 255));
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    ledcWrite(0, 0);
  }
  
  // Motor Kanan
  if (pwmRight > 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    ledcWrite(1, map(pwmRight, 0, 100, 0, 255));
  } else if (pwmRight < 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    ledcWrite(1, map(abs(pwmRight), 0, 100, 0, 255));
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    ledcWrite(1, 0);
  }
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(0, 0);
  ledcWrite(1, 0);
}

// SerialEvent untuk menangani data masuk
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}