#include <Arduino.h>

// Pin Motor sesuai konfigurasi Anda
#define ENA 14   // PWM Motor Kiri
#define IN1 27    // Motor Kiri
#define IN2 26    // Motor Kiri
#define IN3 25    // Motor Kanan
#define IN4 33    // Motor Kanan
#define ENB 32    // PWM Motor Kanan

// Variabel untuk parsing serial
String inputString = "";
bool stringComplete = false;

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
  
  Serial.println("ESP32 Motor Controller Ready");
}

void loop() {
  // Baca data serial
  if (stringComplete) {
    // Parse data (format: "pwm_kiri,pwm_kanan")
    int commaIndex = inputString.indexOf(',');
    if (commaIndex > 0) {
      int pwmLeft = inputString.substring(0, commaIndex).toInt();
      int pwmRight = inputString.substring(commaIndex + 1).toInt();
      
      // Kontrol motor
      setMotors(pwmLeft, pwmRight);
      
      // Debug output
      Serial.print("Received - Left: ");
      Serial.print(pwmLeft);
      Serial.print(", Right: ");
      Serial.println(pwmRight);
    }
    
    // Reset string
    inputString = "";
    stringComplete = false;
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