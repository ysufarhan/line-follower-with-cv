#include <Arduino.h>

// Pilih salah satu UART untuk komunikasi dengan Raspberry Pi
#define USE_UART0 true   // Set true untuk menggunakan UART0 (Serial)
#define USE_UART2 false  // Set true untuk menggunakan UART2

// Pin untuk UART2 (jika digunakan)
#define RXD2 16
#define TXD2 17

// Untuk debugging
#define DEBUG_BAUD 115200

// LED built-in untuk indikator visual
#define LED_PIN 2

void setup() {
  // LED untuk indikator visual
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Inisialisasi Serial untuk komunikasi dengan Raspberry Pi
  if (USE_UART0) {
    Serial.begin(115200);
    Serial.setTimeout(50);  // Timeout untuk readString/readStringUntil
  }
  
  // Inisialisasi Serial2 jika digunakan
  if (USE_UART2) {
    Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);
    Serial2.setTimeout(50);  // Timeout untuk readString/readStringUntil
    
    // Gunakan Serial untuk debugging saja
    Serial.begin(DEBUG_BAUD);
    Serial.println("ESP32 menggunakan UART2 untuk komunikasi dengan Raspberry Pi");
  } else if (!USE_UART0) {
    // Jika tidak menggunakan UART sama sekali
    Serial.begin(DEBUG_BAUD);
    Serial.println("ESP32 tidak menggunakan UART");
  }
  
  // Tambahkan delay startup untuk stabilitas
  delay(1000);
  
  // Bersihkan buffer serial
  while (Serial.available()) {
    Serial.read();
  }
  
  if (USE_UART2) {
    while (Serial2.available()) {
      Serial2.read();
    }
  }
  
  // Kirim pesan startup
  if (USE_UART0) {
    Serial.println("ESP32 siap berkomunikasi dengan Raspberry Pi");
  } else if (USE_UART2) {
    Serial2.println("ESP32 siap berkomunikasi dengan Raspberry Pi");
    Serial.println("Pesan siap terkirim ke Raspberry Pi");
  }
  
  // Flash LED sebagai indikator bahwa setup selesai
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

void loop() {
  // Indikasi bahwa masih hidup
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink > 2000) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
    lastBlink = millis();
  }
  
  // UART0: Menerima data dari Raspberry Pi
  if (USE_UART0 && Serial.available() > 0) {
    String receivedData = Serial.readStringUntil('\n');
    receivedData.trim();  // Hapus whitespace dan newline
    
    // Flash LED sebagai indikator data diterima
    digitalWrite(LED_PIN, HIGH);
    
    // Cek apakah data valid
    if (receivedData.length() > 0) {
      // Kirim respons kembali ke Raspberry Pi
      String response = "ESP32 menerima: " + receivedData;
      Serial.println(response);
    }
    
    digitalWrite(LED_PIN, LOW);
  }
  
  // UART2: Menerima data dari Raspberry Pi (jika digunakan)
  if (USE_UART2 && Serial2.available() > 0) {
    String receivedData = Serial2.readStringUntil('\n');
    receivedData.trim();  // Hapus whitespace dan newline
    
    // Flash LED sebagai indikator data diterima
    digitalWrite(LED_PIN, HIGH);
    
    // Debug: Print data yang diterima ke Serial
    Serial.print("Data diterima dari Raspberry Pi: '");
    Serial.print(receivedData);
    Serial.println("'");
    
    // Cek apakah data valid
    if (receivedData.length() > 0) {
      // Kirim respons kembali ke Raspberry Pi
      String response = "ESP32 menerima: " + receivedData;
      Serial2.println(response);
      Serial.println("Respons terkirim: " + response);
    }
    
    digitalWrite(LED_PIN, LOW);
  }
  
  // Delay kecil untuk stabilitas
  delay(10);
}
