/*
 * ==============================================================================
 * AIoT Driver Drowsiness Detection System - ESP32 Physical IMU Node Firmware
 * Target Microcontroller: ESP32 Dev Module / ESP32-WROOM-32
 * Sensor: MPU-6050 6-DOF IMU (Accelerometer + Gyroscope)
 * Default I2C Address: 0x68 (AD0 connected to GND)
 * Serial Baud Rate: 115200 baud
 * Sampling Interval: ~200 ms (5 Hz JSON output)
 * ==============================================================================
 *
 * HARDWARE PINOUT / WIRING:
 *   ESP32 Dev Board    MPU-6050 Module     Description
 *   ---------------    ---------------     -----------
 *   3V3 / 5V           VCC                 Power (3.3V recommended)
 *   GND                GND                 Common Ground
 *   GPIO 21            SDA                 I2C Data Line
 *   GPIO 22            SCL                 I2C Clock Line
 *   GND                AD0                 Selects 0x68 I2C Address (pull to GND)
 *
 * PROTOCOL & FORMAT (1 JSON line per ~200 ms):
 *   {"status":"OK","chip_id":"0x68","chip_verified":true,"ax":0.05,"ay":-0.12,"az":9.81,"gx":0.0,"gy":-0.1,"gz":0.0,"pitch":0.7,"roll":-0.3,"uptime_ms":2400}
 * ==============================================================================
 */

#include <Wire.h>

#define MPU6050_I2C_ADDR      0x68
#define MPU6050_SMPLRT_DIV    0x19
#define MPU6050_CONFIG        0x1A
#define MPU6050_GYRO_CONFIG   0x1B
#define MPU6050_ACCEL_CONFIG  0x1C
#define MPU6050_PWR_MGMT_1    0x6B
#define MPU6050_WHO_AM_I      0x75
#define MPU6050_ACCEL_XOUT_H  0x3B

// Conversion scale factors for default ranges (+-2g and +-250 deg/s)
const float ACCEL_SCALE_G = 16384.0; // 16384 LSB/g
const float GRAVITY = 9.80665;       // 1g in m/s^2
const float GYRO_SCALE = 131.0;      // 131 LSB / (deg/s)

bool sensorVerified = false;
uint8_t chipId = 0x00;

// Stationary Gyroscope Offsets
float gyro_offset_x = 0.0;
float gyro_offset_y = 0.0;
float gyro_offset_z = 0.0;

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 200; // ~200 ms output interval (~5 Hz)

void writeI2C(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

uint8_t readI2C(uint8_t reg) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(reg);
  uint8_t err = Wire.endTransmission(false);
  if (err != 0) {
    return 0xFF;
  }
  Wire.requestFrom((uint8_t)MPU6050_I2C_ADDR, (uint8_t)1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0xFF;
}

bool initMPU6050() {
  Wire.begin(21, 22, 100000); // SDA = GPIO21, SCL = GPIO22, 100kHz standard I2C
  delay(100);

  // 1. Check WHO_AM_I register (0x75) - Default MPU-6050 returns 0x68
  chipId = readI2C(MPU6050_WHO_AM_I);
  if (chipId == 0x68 || chipId == 0x70 || chipId == 0x72) {
    sensorVerified = true;
  } else {
    sensorVerified = false;
    return false;
  }

  // 2. Wake up MPU-6050 (clears sleep bit in PWR_MGMT_1, use X-axis gyro clock)
  writeI2C(MPU6050_PWR_MGMT_1, 0x01);
  delay(20);

  // 3. Sample Rate Divider: 1kHz / (1 + 9) = 100Hz internal sampling
  writeI2C(MPU6050_SMPLRT_DIV, 0x09);

  // 4. Digital Low Pass Filter: DLPF_CFG = 2 (~44Hz bandwidth for vehicular motion)
  writeI2C(MPU6050_CONFIG, 0x02);

  // 5. Gyro Full Scale Range: +-250 deg/s
  writeI2C(MPU6050_GYRO_CONFIG, 0x00);

  // 6. Accel Full Scale Range: +-2g
  writeI2C(MPU6050_ACCEL_CONFIG, 0x00);

  return true;
}

void calibrateGyroscope() {
  Serial.println("{\"event\":\"CALIBRATING_GYRO\",\"samples\":100,\"message\":\"Keep sensor stationary...\"}");
  long sum_gx = 0;
  long sum_gy = 0;
  long sum_gz = 0;
  int validSamples = 0;

  for (int i = 0; i < 100; i++) {
    Wire.beginTransmission(MPU6050_I2C_ADDR);
    Wire.write(0x43); // GYRO_XOUT_H
    if (Wire.endTransmission(false) == 0) {
      Wire.requestFrom((uint8_t)MPU6050_I2C_ADDR, (uint8_t)6);
      if (Wire.available() >= 6) {
        int16_t gx = (Wire.read() << 8) | Wire.read();
        int16_t gy = (Wire.read() << 8) | Wire.read();
        int16_t gz = (Wire.read() << 8) | Wire.read();
        sum_gx += gx;
        sum_gy += gy;
        sum_gz += gz;
        validSamples++;
      }
    }
    delay(5);
  }

  if (validSamples > 50) {
    gyro_offset_x = ((float)sum_gx / validSamples) / GYRO_SCALE;
    gyro_offset_y = ((float)sum_gy / validSamples) / GYRO_SCALE;
    gyro_offset_z = ((float)sum_gz / validSamples) / GYRO_SCALE;
  } else {
    gyro_offset_x = 0.0;
    gyro_offset_y = 0.0;
    gyro_offset_z = 0.0;
  }

  Serial.print("{\"event\":\"CALIBRATION_COMPLETE\",\"valid_samples\":");
  Serial.print(validSamples);
  Serial.print(",\"offset_gx\":");
  Serial.print(gyro_offset_x, 2);
  Serial.print(",\"offset_gy\":");
  Serial.print(gyro_offset_y, 2);
  Serial.print(",\"offset_gz\":");
  Serial.print(gyro_offset_z, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  Serial.println("{\"event\":\"BOOT\",\"system\":\"AIoT-Drowsiness-ESP32-IMU\",\"baud\":115200}");
  
  bool initialized = initMPU6050();
  if (initialized) {
    Serial.print("{\"event\":\"INIT_SUCCESS\",\"chip\":\"MPU-6050\",\"who_am_i\":\"0x");
    Serial.print(chipId, HEX);
    Serial.println("\"}");
    
    // Calibrate stationary offsets
    calibrateGyroscope();
  } else {
    Serial.print("{\"event\":\"INIT_FAILED\",\"error\":\"Sensor identity mismatch or I2C bus error\",\"read_who_am_i\":\"0x");
    Serial.print(chipId, HEX);
    Serial.println("\"}");
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTime = now;

  if (!sensorVerified) {
    // Attempt re-verification every 2 seconds if sensor was not found
    static unsigned long lastRetry = 0;
    if (now - lastRetry > 2000) {
      lastRetry = now;
      if (initMPU6050()) {
        calibrateGyroscope();
      }
    }
    Serial.print("{\"status\":\"UNVERIFIED\",\"chip_verified\":false,\"error\":\"Sensor not initialized or detached\",\"uptime_ms\":");
    Serial.print(now);
    Serial.println("}");
    return;
  }

  // Read 14 sequential registers: Accel (6) + Temp (2) + Gyro (6)
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  uint8_t txErr = Wire.endTransmission(false);

  if (txErr != 0) {
    Serial.print("{\"status\":\"READ_ERROR\",\"error\":\"I2C transmission error\",\"err_code\":");
    Serial.print(txErr);
    Serial.print(",\"uptime_ms\":");
    Serial.print(now);
    Serial.println("}");
    return;
  }

  uint8_t bytesRead = Wire.requestFrom((uint8_t)MPU6050_I2C_ADDR, (uint8_t)14);
  if (bytesRead < 14) {
    Serial.print("{\"status\":\"READ_ERROR\",\"error\":\"I2C read timeout, insufficient bytes\",\"uptime_ms\":");
    Serial.print(now);
    Serial.println("}");
    return;
  }

  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  int16_t raw_temp = (Wire.read() << 8) | Wire.read();
  int16_t raw_gx = (Wire.read() << 8) | Wire.read();
  int16_t raw_gy = (Wire.read() << 8) | Wire.read();
  int16_t raw_gz = (Wire.read() << 8) | Wire.read();

  // Convert raw readings to physical units
  float ax_ms2 = ((float)raw_ax / ACCEL_SCALE_G) * GRAVITY;
  float ay_ms2 = ((float)raw_ay / ACCEL_SCALE_G) * GRAVITY;
  float az_ms2 = ((float)raw_az / ACCEL_SCALE_G) * GRAVITY;

  // Subtract calibrated stationary zero-offsets
  float gx_degs = ((float)raw_gx / GYRO_SCALE) - gyro_offset_x;
  float gy_degs = ((float)raw_gy / GYRO_SCALE) - gyro_offset_y;
  float gz_degs = ((float)raw_gz / GYRO_SCALE) - gyro_offset_z;

  // Attitude calculations (degrees)
  float pitch = atan2(-ax_ms2, sqrt(ay_ms2 * ay_ms2 + az_ms2 * az_ms2)) * 180.0 / PI;
  float roll  = atan2(ay_ms2, az_ms2) * 180.0 / PI;

  // Output clean, continuous, single-line JSON telemetry
  Serial.print("{\"status\":\"OK\"");
  Serial.print(",\"chip_id\":\"0x");
  Serial.print(chipId, HEX);
  Serial.print("\"");
  Serial.print(",\"chip_verified\":true");
  Serial.print(",\"ax\":"); Serial.print(ax_ms2, 2);
  Serial.print(",\"ay\":"); Serial.print(ay_ms2, 2);
  Serial.print(",\"az\":"); Serial.print(az_ms2, 2);
  Serial.print(",\"gx\":"); Serial.print(gx_degs, 1);
  Serial.print(",\"gy\":"); Serial.print(gy_degs, 1);
  Serial.print(",\"gz\":"); Serial.print(gz_degs, 1);
  Serial.print(",\"pitch\":"); Serial.print(pitch, 1);
  Serial.print(",\"roll\":"); Serial.print(roll, 1);
  Serial.print(",\"uptime_ms\":"); Serial.print(now);
  Serial.println("}");
}
