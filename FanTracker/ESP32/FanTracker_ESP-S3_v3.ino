#include <WiFi.h>
#include <Wire.h>
#include <SPIFFS.h>
#include <ESP32Servo.h>
#include <math.h>
#include <esp_sleep.h>
#include "MS5837.h"

// ===== WiFi (SoftAP) =====
const char* AP_SSID = "FanTracker-ESP32S3";
const char* AP_PASS = "fantracker123";
WiFiServer server(3333);
WiFiClient client;

// ===== MS5837 / Bar30 =====
const int SDA_PIN = 47;
const int SCL_PIN = 48;
MS5837 sensor;

// ===== Batería =====
const int BATTERY_PIN = 2; // GPIO2 (ADC1) libre en ESP32-C3 mini
const float ADC_REF_V = 3.3;
const float BATTERY_DIVIDER_RATIO = 6.0; // (100k + 20k) / 20k
const float BATTERY_CALIBRATION = 1.0577; // Ajuste según multímetro
const float BATTERY_FULL_V = 16.8;  // 4S a 4.2V/celda
const float BATTERY_EMPTY_V = 12.0; // 4S a 3.0V/celda
const float BATTERY_CUTOFF_PERCENT = 20.0;
bool batteryLockout = false;

// ===== Motor T200 =====
const int MOTOR_PIN = 20;  // GPIO para PWM del ESC
const int NEUTRAL_US = 1500;
const int MIN_US = 1100;
const int MAX_US = 1900;
Servo thruster;

// ===== Misión =====
enum MissionState { IDLE, ARMING, DESCENT, HOLD, ASCENDING, COMPLETED };
MissionState missionState = IDLE;
float targetDepth = 0.0;
float descentPWM_us = 1625;  // PWM para descender
float holdSeconds = 3.0;    // tiempo manteniendo profundidad
float armSeconds = 4.0;      // tiempo para armar ESC
float surfaceBand = 0.3;     // metros cerca de superficie
float stagnationTimeout = 10.0;  // segundos sin cambio = apagar
float stagnationDelta = 0.05;    // metros mínimos de cambio
unsigned long missionStartTime = 0;
unsigned long holdStartTime = 0;
unsigned long lastDepthChangeTime = 0;
float lastDepth = 0.0;
bool autoMissionEnabled = false;
float autoMissionDepth = 0.8;
unsigned long autoMissionIntervalMs = 180000;
unsigned long nextAutoMissionMs = 0;
const unsigned long autoMissionStartDelayMs = 90000;
String missionEndReason = "";
bool sleepPending = false;
unsigned long sleepRequestMs = 0;
const unsigned long MISSION_SLEEP_THRESHOLD_MS = 1800000;
const unsigned long POST_MISSION_SLEEP_DELAY_MS = 300000;

// ===== Logging =====
String currentLogFile = "/log_actual.csv";
bool loggingEnabled = false;
File logFile;
bool logFinalReasonWritten = false;

// ===== Runtime params =====
volatile bool streamEnabled = true;
volatile uint32_t samplePeriodMs = 1000;
volatile float fluidDensity = 997.0f;

// ===== Filtros de sensor =====
const float PRESSURE_JUMP_MAX_MBAR = 2500.0f;
float lastValidPressure = NAN;
float lastValidDepth = NAN;
float lastValidTemp = NAN;
unsigned long lastSpikeWarningMs = 0;
float lastIgnoredPressure = NAN;
float lastIgnoredDepth = NAN;
float lastIgnoredTemp = NAN;
unsigned long lastIgnoredMs = 0;
unsigned long ignoredWindowStartMs = 0;
int ignoredCount = 0;
const unsigned long IGNORED_WINDOW_MS = 60000;
const int IGNORED_LIMIT = 10;
bool motorActive = false;
float missionZeroDepth = 0.0f;
bool missionZeroValid = false;

// ----- timestamp -----
String getTimestamp() {
  uint32_t totalSeconds = millis() / 1000;
  uint32_t hours = totalSeconds / 3600;
  uint32_t minutes = (totalSeconds % 3600) / 60;
  uint32_t seconds = totalSeconds % 60;
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%02lu:%02lu:%02lu",
           static_cast<unsigned long>(hours),
           static_cast<unsigned long>(minutes),
           static_cast<unsigned long>(seconds));
  return String(buffer);
}

// ----- helpers -----
void sendLine(const String& s) {
  Serial.println(s);
  if (client && client.connected()) client.println(s);
}

void setThrusterPWM(int us) {
  us = constrain(us, MIN_US, MAX_US);
  thruster.writeMicroseconds(us);
  motorActive = (us != NEUTRAL_US);
}

void stopThruster() {
  setThrusterPWM(NEUTRAL_US);
  delay(500);
  sendLine("[MOTOR] Detenido");
}

void i2cScan() {
  sendLine("I2C scan:");
  int found = 0;
  for (int addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      String line = "  Found 0x";
      if (addr < 16) line += "0";
      line += String(addr, HEX);
      sendLine(line);
      found++;
    }
  }
  if (found == 0) sendLine("  No I2C devices found");
}

String getMissionStateStr() {
  switch(missionState) {
    case IDLE: return "idle";
    case ARMING: return "arming";
    case DESCENT: return "descent";
    case HOLD: return "hold";
    case ASCENDING: return "ascending";
    case COMPLETED: return "idle";
    default: return "unknown";
  }
}

bool readSensorFiltered() {
  sensor.read();
  float pressure = sensor.pressure();
  float depth = sensor.depth();
  float temperature = sensor.temperature();

  if (isnan(lastValidPressure)) {
    lastValidPressure = pressure;
    lastValidDepth = depth;
    lastValidTemp = temperature;
    return true;
  }

  float pressureDelta = fabs(pressure - lastValidPressure);
  if (pressureDelta > PRESSURE_JUMP_MAX_MBAR) {
    unsigned long now = millis();
    lastIgnoredPressure = pressure;
    lastIgnoredDepth = depth;
    lastIgnoredTemp = temperature;
    lastIgnoredMs = now;
    if (ignoredWindowStartMs == 0 || now - ignoredWindowStartMs > IGNORED_WINDOW_MS) {
      ignoredWindowStartMs = now;
      ignoredCount = 0;
    }
    ignoredCount++;
    if (motorActive && ignoredCount >= IGNORED_LIMIT) {
      sendLine("[FAILSAFE] Demasiados saltos de presión. Apagando motor.");
      missionEndReason = "sensor_jump_failsafe";
      stopThruster();
      missionState = ASCENDING;
    }
    if (now - lastSpikeWarningMs > 1000) {
      sendLine("[SENSOR] Salto de presión ignorado: " + String(pressure, 2) +
               " mbar (delta " + String(pressureDelta, 1) + ")");
      lastSpikeWarningMs = now;
    }
    return false;
  }

  lastValidPressure = pressure;
  lastValidDepth = depth;
  lastValidTemp = temperature;
  return true;
}

String readBar30Line() {
  if (!readSensorFiltered()) {
    return String("[SENSOR] Lectura ignorada por salto");
  }
  String line;
  line.reserve(120);
  line += getTimestamp();
  line += " [" + getMissionStateStr() + "] ";
  line += "P: ";
  line += String(lastValidPressure, 1);
  line += " mbar | Prof: ";
  line += String(lastValidDepth, 3);
  line += " m | T: ";
  line += String(lastValidTemp, 2);
  line += " C";
  return line;
}

String readBar30CSV() {
  if (!readSensorFiltered()) {
    float batteryVoltage = readBatteryVoltage();
    float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
    float depthZeroed = missionZeroValid ? (lastIgnoredDepth - missionZeroDepth) : 0.0f;
    String line;
    line.reserve(140);
    line += getTimestamp();
    line += ",error,";
    line += String(lastIgnoredPressure, 2);
    line += ",";
    line += String(lastIgnoredTemp, 2);
    line += ",";
    line += String(lastIgnoredDepth, 3);
    line += ",";
    line += String(depthZeroed, 3);
    line += ",";
    line += String(batteryVoltage, 2);
    line += ",";
    line += String(batteryPercent, 1);
    return line;
  }
  float batteryVoltage = readBatteryVoltage();
  float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
  float depthZeroed = missionZeroValid ? (lastValidDepth - missionZeroDepth) : 0.0f;
  String line;
  line.reserve(140);
  line += getTimestamp();
  line += ",";
  line += getMissionStateStr();
  line += ",";
  line += String(lastValidPressure, 2);
  line += ",";
  line += String(lastValidTemp, 2);
  line += ",";
  line += String(lastValidDepth, 3);
  line += ",";
  line += String(depthZeroed, 3);
  line += ",";
  line += String(batteryVoltage, 2);
  line += ",";
  line += String(batteryPercent, 1);
  return line;
}

float readBatteryVoltage() {
  int raw = analogRead(BATTERY_PIN);
  float v_adc = (raw / 4095.0) * ADC_REF_V;
  return v_adc * BATTERY_DIVIDER_RATIO * BATTERY_CALIBRATION;
}

float batteryPercentFromVoltage(float voltage) {
  float percent = (voltage - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V) * 100.0;
  if (percent < 0.0) percent = 0.0;
  if (percent > 100.0) percent = 100.0;
  return percent;
}

void writeLogHeader() {
  float batteryVoltage = readBatteryVoltage();
  float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
  logFile.println("=== ESP32 FanTracker Controller ===");
  logFile.println("Escribe 'help' para comandos.");
  logFile.println("bateria_v=" + String(batteryVoltage, 2) + " V");
  logFile.println("bateria_pct=" + String(batteryPercent, 1) + " %");
  logFile.println("pwm_descenso_us=" + String((int)descentPWM_us) + " us");
  logFile.println("profundidad_objetivo_m=" + String(targetDepth, 2) + " m");
  logFile.println("hold_time_s=" + String(holdSeconds, 1) + " s");
  logFile.println("arm_time_s=" + String(armSeconds, 1) + " s");
  logFile.println("surface_band_m=" + String(surfaceBand, 2) + " m");
  logFile.println("stagnation_timeout_s=" + String(stagnationTimeout, 1) + " s");
  logFile.println("stagnation_delta_m=" + String(stagnationDelta, 2) + " m");
  logFile.println("auto_depth_m=" + String(autoMissionDepth, 2) + " m");
  logFile.println("auto_interval_s=" + String(autoMissionIntervalMs / 1000.0, 1) + " s");
  logFile.println("auto_start_delay_s=" + String(autoMissionStartDelayMs / 1000.0, 1) + " s");
  logFile.println("sample_period_s=" + String(samplePeriodMs / 1000.0, 2) + " s");
  logFile.println("fluid_density_kg_m3=" + String(fluidDensity, 1) + " kg/m3");
  logFile.println("pressure_jump_max_mbar=" + String(PRESSURE_JUMP_MAX_MBAR, 1) + " mbar");
  logFile.println("pressure_jump_window_s=" + String(IGNORED_WINDOW_MS / 1000.0, 1) + " s");
  logFile.println("pressure_jump_limit=" + String(IGNORED_LIMIT));
  logFile.println("sleep_threshold_s=" + String(MISSION_SLEEP_THRESHOLD_MS / 1000.0, 1) + " s");
  logFile.println("sleep_delay_s=" + String(POST_MISSION_SLEEP_DELAY_MS / 1000.0, 1) + " s");
  logFile.println("=== BEGIN FILE: " + currentLogFile + " ===");
  logFile.println("timestamp,state,pressure_mbar,temp_C,depth_real_m,depth_zeroed_m,battery_v,battery_pct");
}

void writeLogEndReason() {
  if (!logFile || logFinalReasonWritten) return;
  String reason = missionEndReason.length() > 0 ? missionEndReason : "fin_manual";
  logFile.println("=== MISION FINALIZADA: " + reason + " ===");
  logFinalReasonWritten = true;
}

void startLogging() {
  if (loggingEnabled) {
    sendLine("Log ya está activo");
    return;
  }
  
  currentLogFile = "/log_" + String(millis()) + ".csv";
  
  logFile = SPIFFS.open(currentLogFile, FILE_WRITE);
  if (!logFile) {
    sendLine("ERROR: no se pudo crear archivo log");
    return;
  }
  
  logFinalReasonWritten = false;
  writeLogHeader();
  loggingEnabled = true;
  sendLine("Logging iniciado: " + currentLogFile);

  if (logFile) {
    String firstLine = readBar30CSV();
    if (firstLine.length() > 0) {
      logFile.println(firstLine);
    }
    logFile.flush();
  }
}

void stopLogging() {
  if (!loggingEnabled) {
    sendLine("Log no está activo");
    return;
  }
  
  if (logFile) {
    writeLogEndReason();
    logFile.close();
  }
  loggingEnabled = false;
  sendLine("Logging detenido: " + currentLogFile);
}

void startMission(float depth_m) {
  if (missionState != IDLE && missionState != COMPLETED) {
    sendLine("ERROR: Misión ya en curso. Usa 'abort' primero.");
    return;
  }

  if (batteryLockout) {
    sendLine("ERROR: Batería baja. Misión bloqueada.");
    return;
  }

  targetDepth = depth_m;
  missionEndReason = "";
  sleepPending = false;
  missionState = ARMING;
  missionStartTime = millis();
  lastDepth = 0.0;
  lastDepthChangeTime = millis();
  missionZeroValid = false;
  
  // Auto-start logging
  if (!loggingEnabled) {
    startLogging();
  }

  float batteryVoltage = readBatteryVoltage();
  float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
  sendLine("[BATERÍA] Voltaje=" + String(batteryVoltage, 2) +
           " V | " + String(batteryPercent, 1) + "%");
  if (batteryPercent < BATTERY_CUTOFF_PERCENT) {
    sendLine("[BATERÍA] < 20%. Auto OFF y misión cancelada.");
    missionEndReason = "bateria_baja";
    stopThruster();
    autoMissionEnabled = false;
    batteryLockout = true;
    if (loggingEnabled) {
      stopLogging();
    }
    missionState = IDLE;
    return;
  }

  String msg = "=== MISIÓN INICIADA ===";
  msg += "\nProfundidad objetivo: " + String(targetDepth, 2) + " m";
  msg += "\nPWM descenso: " + String((int)descentPWM_us) + " us";
  msg += "\nHold: " + String(holdSeconds, 1) + " s";
  sendLine(msg);

  if (readSensorFiltered()) {
    missionZeroDepth = lastValidDepth;
    missionZeroValid = true;
  } else if (!isnan(lastValidDepth)) {
    missionZeroDepth = lastValidDepth;
    missionZeroValid = true;
  } else {
    missionZeroDepth = 0.0f;
    missionZeroValid = false;
  }
}

void abortMission() {
  if (missionState == IDLE) {
    sendLine("No hay misión activa");
    return;
  }
  
  stopThruster();
  missionState = IDLE;
  missionZeroValid = false;
  if (loggingEnabled) {
    stopLogging();
  }
  sendLine("=== MISIÓN ABORTADA ===");
}

void updateStagnationDetection(float depth) {
  if (abs(depth - lastDepth) >= stagnationDelta) {
    lastDepthChangeTime = millis();
    lastDepth = depth;
  }
}

void processMission() {
  if (missionState == IDLE || missionState == COMPLETED) return;
  
  if (!readSensorFiltered()) return;
  float depth = lastValidDepth;
  updateStagnationDetection(depth);
  
  unsigned long now = millis();
  float stagnationAge = (now - lastDepthChangeTime) / 1000.0;
  
  // Failsafe: estancamiento bajo el agua
  if (depth > surfaceBand && stagnationAge >= stagnationTimeout) {
    sendLine("[FAILSAFE] Sin cambio de profundidad por " + String(stagnationAge, 1) + 
             "s. Apagando motor para emerger.");
    missionEndReason = "estancamiento_sin_cambio";
    stopThruster();
    missionState = ASCENDING;
    return;
  }
  
  switch(missionState) {
    case ARMING:
      setThrusterPWM(NEUTRAL_US);
      if ((now - missionStartTime) >= (armSeconds * 1000)) {
        sendLine("[ESC] Armado completo. Iniciando descenso...");
        missionState = DESCENT;
      }
      break;
      
    case DESCENT:
      setThrusterPWM(descentPWM_us);
      if (depth >= targetDepth) {
        sendLine("[INFO] Profundidad alcanzada. Iniciando desaceleración controlada.");
        missionEndReason = "profundidad_objetivo_alcanzada";
        missionState = HOLD;
        holdStartTime = millis();
      }
      break;
      
    case HOLD:
      {
        float elapsed = (now - holdStartTime) / 1000.0;
        if (elapsed < holdSeconds) {
          // Rampa lineal desde descentPWM_us a NEUTRAL_US
          float ratio = elapsed / holdSeconds;
          int target_us = descentPWM_us + (NEUTRAL_US - descentPWM_us) * ratio;
          setThrusterPWM(target_us);
        } else {
          sendLine("[INFO] Hold completado. Apagando motor para ascenso por flotabilidad.");
          if (missionEndReason.length() == 0) {
            missionEndReason = "hold_completado";
          }
          stopThruster();
          missionState = ASCENDING;
        }
      }
      break;
      
    case ASCENDING:
      // Motor apagado, esperando llegar a superficie
      if (depth <= surfaceBand) {
        sendLine("=== MISIÓN COMPLETADA ===");
        sendLine("Superficie alcanzada.");
        if (missionEndReason.length() == 0) {
          missionEndReason = "superficie_alcanzada";
        }
        missionState = COMPLETED;
        if (loggingEnabled) {
          stopLogging();
        }
        unsigned long missionDurationMs = millis() - missionStartTime;
        if (missionDurationMs >= MISSION_SLEEP_THRESHOLD_MS) {
          sleepPending = true;
          sleepRequestMs = millis();
          sendLine("[SLEEP] Misión larga. Entrando en deep sleep en " +
                   String(POST_MISSION_SLEEP_DELAY_MS / 1000.0, 0) + "s.");
        }
        if (autoMissionEnabled && autoMissionIntervalMs > 0) {
          nextAutoMissionMs = millis() + autoMissionIntervalMs;
        }
        missionZeroValid = false;
        missionState = IDLE;
      }
      break;
      
    default:
      break;
  }
}

void processAutoMission() {
  if (!autoMissionEnabled) return;
  if (missionState != IDLE) return;

  unsigned long now = millis();
  if (now >= nextAutoMissionMs) {
    startMission(autoMissionDepth);
  }
}

void listFiles() {
  File root = SPIFFS.open("/");
  if (!root || !root.isDirectory()) {
    sendLine("ERROR: no se pudo abrir SPIFFS");
    return;
  }
  
  sendLine("=== Archivos en SPIFFS ===");
  File file = root.openNextFile();
  int totalSize = 0;
  int count = 0;
  
  while (file) {
    String name = String(file.name());
    int size = file.size();
    totalSize += size;
    count++;
    
    String line = name + " (" + String(size/1024.0, 2) + " KB)";
    sendLine(line);
    
    file = root.openNextFile();
  }
  
  sendLine("Total: " + String(count) + " archivos, " + 
           String(totalSize/1024.0, 2) + " KB");
  
  size_t total = SPIFFS.totalBytes();
  size_t used = SPIFFS.usedBytes();
  sendLine("Espacio: " + String(used/1024) + "/" + String(total/1024) + " KB");
}

void downloadFile(const String& filename) {
  if (!client || !client.connected()) {
    Serial.println("No hay cliente conectado para download");
    return;
  }
  
  File file = SPIFFS.open(filename, FILE_READ);
  if (!file) {
    client.println("ERROR: archivo no encontrado");
    return;
  }
  
  client.println("=== BEGIN FILE: " + filename + " ===");
  
  while (file.available()) {
    String line = file.readStringUntil('\n');
    client.println(line);
  }
  
  client.println("=== END FILE ===");
  file.close();
}

void deleteFile(const String& filename) {
  if (SPIFFS.remove(filename)) {
    sendLine("Archivo eliminado: " + filename);
  } else {
    sendLine("ERROR: no se pudo eliminar " + filename);
  }
}

void formatSPIFFS() {
  sendLine("Formateando SPIFFS...");
  if (loggingEnabled) stopLogging();
  
  if (SPIFFS.format()) {
    sendLine("SPIFFS formateado OK");
  } else {
    sendLine("ERROR al formatear SPIFFS");
  }
}

void acceptClient() {
  WiFiClient newClient = server.available();
  if (newClient) {
    if (client && client.connected()) client.stop();
    
    client = newClient;
    client.setNoDelay(true);
    client.println("=== ESP32 FanTracker Controller ===");
    client.println("Escribe 'help' para comandos.");
    Serial.println("Cliente TCP conectado.");
  }
  
  if (client && !client.connected()) client.stop();
}

String readClientLine() {
  if (!client || !client.connected() || !client.available()) return "";
  String cmd = client.readStringUntil('\n');
  cmd.trim();
  return cmd;
}

void handleCommand(const String& cmdRaw) {
  if (cmdRaw.length() == 0) return;
  
  String cmd = cmdRaw;
  cmd.trim();
  cmd.toLowerCase();
  
  if (cmd == "help") {
    client.println("=== Comandos ===");
    client.println("--- General ---");
    client.println("  help          - mostrar ayuda");
    client.println("  ping          - test conexion");
    client.println("  read          - leer sensor una vez");
    client.println("  stream on|off - activar/desactivar streaming");
    client.println("  rate <ms>     - intervalo de muestreo (ej: rate 200)");
    client.println("  density <val> - densidad fluido (dulce=997, mar=1029)");
    client.println("  battery       - leer estado batería");
    client.println("  scan          - escanear I2C");
    client.println("--- Misión ---");
    client.println("  dive <depth>  - iniciar misión a profundidad (ej: dive 5.0)");
    client.println("  auto on|off   - activar/desactivar misión automática");
    client.println("  abort         - abortar misión actual");
    client.println("  status        - estado de misión");
    client.println("  config <param> <val> - configurar parámetros");
    client.println("    params: descent_pwm, hold_time, arm_time, surface_band,");
    client.println("            stagnation_timeout, stagnation_delta, auto_depth,");
    client.println("            auto_interval");
    client.println("--- Logging ---");
    client.println("  log start     - iniciar grabación");
    client.println("  log stop      - detener grabación");
    client.println("  log status    - estado del logging");
    client.println("  files         - listar archivos guardados");
    client.println("  download <archivo> - descargar archivo");
    client.println("  delete <archivo>   - eliminar archivo");
    client.println("  format        - formatear memoria (¡BORRA TODO!)");
    client.println("  reboot        - reiniciar ESP32");
    return;
  }
  
  if (cmd == "ping") { client.println("pong"); return; }
  
  if (cmd == "read") {
    client.println(readBar30Line());
    return;
  }
  
  if (cmd.startsWith("stream ")) {
    if (cmd.endsWith("on"))  { streamEnabled = true;  client.println("stream=ON");  return; }
    if (cmd.endsWith("off")) { streamEnabled = false; client.println("stream=OFF"); return; }
    client.println("Uso: stream on|off");
    return;
  }
  
  if (cmd.startsWith("rate ")) {
    long ms = cmd.substring(5).toInt();
    if (ms < 50) ms = 50;
    samplePeriodMs = (uint32_t)ms;
    client.println("rate_ms=" + String(samplePeriodMs));
    return;
  }
  
  if (cmd.startsWith("density ")) {
    float d = cmd.substring(8).toFloat();
    if (d < 500) d = 500;
    if (d > 2000) d = 2000;
    fluidDensity = d;
    sensor.setFluidDensity(fluidDensity);
    client.println("density=" + String(fluidDensity, 1));
    return;
  }
  
  if (cmd == "scan") {
    i2cScan();
    return;
  }

  if (cmd == "battery") {
    float batteryVoltage = readBatteryVoltage();
    float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
    client.println("battery_v=" + String(batteryVoltage, 2) +
                   " V | battery_pct=" + String(batteryPercent, 1) + "%");
    return;
  }
  
  if (cmd.startsWith("dive ")) {
    float depth = cmd.substring(5).toFloat();
    if (depth <= 0 || depth > 50) {
      client.println("ERROR: profundidad inválida (0-50m)");
      return;
    }
    startMission(depth);
    return;
  }

  if (cmd == "auto on") {
    autoMissionEnabled = true;
    nextAutoMissionMs = millis() + autoMissionStartDelayMs;
    client.println("auto=ON");
    return;
  }

  if (cmd == "auto off") {
    autoMissionEnabled = false;
    client.println("auto=OFF");
    return;
  }

  if (cmd == "abort") {
    missionEndReason = "abortar_usuario";
    abortMission();
    return;
  }
  
  if (cmd == "status") {
    client.println("=== Estado Misión ===");
    client.println("Estado: " + getMissionStateStr());
    client.println("Profundidad objetivo: " + String(targetDepth, 2) + " m");
    if (readSensorFiltered()) {
      float depthZeroed = missionZeroValid ? (lastValidDepth - missionZeroDepth) : 0.0f;
      client.println("Profundidad actual: " + String(lastValidDepth, 3) + " m");
      client.println("Profundidad inicio: " + String(depthZeroed, 3) + " m");
    } else {
      client.println("Profundidad actual: (lectura ignorada)");
      client.println("Profundidad inicio: (lectura ignorada)");
    }
    float batteryVoltage = readBatteryVoltage();
    float batteryPercent = batteryPercentFromVoltage(batteryVoltage);
    client.println("Batería: " + String(batteryVoltage, 2) + " V | " +
                   String(batteryPercent, 1) + "%");
    client.println("PWM descenso: " + String((int)descentPWM_us) + " us");
    client.println("Hold time: " + String(holdSeconds, 1) + " s");
    client.println("Surface band: " + String(surfaceBand, 2) + " m");
    client.println("Stagnation timeout: " + String(stagnationTimeout, 1) + " s");
    client.println("Stagnation delta: " + String(stagnationDelta, 2) + " m");
    client.println("Auto mission: " + String(autoMissionEnabled ? "on" : "off"));
    client.println("Auto depth: " + String(autoMissionDepth, 2) + " m");
    client.println("Auto interval: " + String(autoMissionIntervalMs / 1000.0, 1) + " s");
    return;
  }
  
  if (cmd.startsWith("config ")) {
    String params = cmd.substring(7);
    int spaceIdx = params.indexOf(' ');
    if (spaceIdx > 0) {
      String param = params.substring(0, spaceIdx);
      float value = params.substring(spaceIdx + 1).toFloat();
      
      if (param == "descent_pwm") {
        descentPWM_us = constrain(value, MIN_US, MAX_US);
        client.println("descent_pwm=" + String((int)descentPWM_us));
      } else if (param == "hold_time") {
        holdSeconds = max(0.0f, value);
        client.println("hold_time=" + String(holdSeconds, 1));
      } else if (param == "arm_time") {
        armSeconds = max(0.0f, value);
        client.println("arm_time=" + String(armSeconds, 1));
      } else if (param == "surface_band") {
        surfaceBand = max(0.0f, value);
        client.println("surface_band=" + String(surfaceBand, 2));
      } else if (param == "stagnation_timeout") {
        stagnationTimeout = max(0.0f, value);
        client.println("stagnation_timeout=" + String(stagnationTimeout, 1));
      } else if (param == "stagnation_delta") {
        stagnationDelta = max(0.0f, value);
        client.println("stagnation_delta=" + String(stagnationDelta, 2));
      } else if (param == "auto_depth") {
        if (value <= 0 || value > 50) {
          client.println("ERROR: auto_depth inválido (0-50m)");
          return;
        }
        autoMissionDepth = value;
        client.println("auto_depth=" + String(autoMissionDepth, 2));
      } else if (param == "auto_interval") {
        if (value < 1) value = 1;
        autoMissionIntervalMs = (unsigned long)(value * 1000.0f);
        if (missionState == IDLE) {
          nextAutoMissionMs = millis() + autoMissionIntervalMs;
        }
        client.println("auto_interval=" + String(autoMissionIntervalMs / 1000.0, 1));
      } else {
        client.println("Parámetro desconocido. Usa: descent_pwm, hold_time, arm_time,");
        client.println("surface_band, stagnation_timeout, stagnation_delta, auto_depth,");
        client.println("auto_interval");
      }
    } else {
      client.println("Uso: config <param> <value>");
    }
    return;
  }
  
  if (cmd == "log start") {
    startLogging();
    return;
  }
  
  if (cmd == "log stop") {
    stopLogging();
    return;
  }
  
  if (cmd == "log status") {
    if (loggingEnabled) {
      client.println("Logging ACTIVO: " + currentLogFile);
    } else {
      client.println("Logging INACTIVO");
    }
    return;
  }
  
  if (cmd == "files") {
    listFiles();
    return;
  }
  
  if (cmd.startsWith("download ")) {
    String filename = cmdRaw.substring(9);
    filename.trim();
    if (!filename.startsWith("/")) filename = "/" + filename;
    downloadFile(filename);
    return;
  }
  
  if (cmd.startsWith("delete ")) {
    String filename = cmdRaw.substring(7);
    filename.trim();
    if (!filename.startsWith("/")) filename = "/" + filename;
    deleteFile(filename);
    return;
  }
  
  if (cmd == "format") {
    formatSPIFFS();
    return;
  }
  
  if (cmd == "reboot") {
    client.println("Reiniciando...");
    client.flush();
    client.stop();
    delay(200);
    ESP.restart();
  }
  
  client.println("Comando no reconocido. Usa: help");
}

void setup() {
  Serial.begin(115200);
  delay(200);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(BATTERY_PIN, INPUT);

  // --- SPIFFS ---
  if (!SPIFFS.begin(true)) {
    Serial.println("ERROR: SPIFFS mount failed");
    while (1) delay(1000);
  }
  Serial.println("SPIFFS montado OK");
  
  // --- Motor ---
  ESP32PWM::allocateTimer(0);
  thruster.setPeriodHertz(50);
  thruster.attach(MOTOR_PIN, MIN_US, MAX_US);
  thruster.writeMicroseconds(0);
  Serial.println("Motor T200 configurado en GPIO " + String(MOTOR_PIN));
  
  // --- WiFi ---
  WiFi.mode(WIFI_AP);
  bool apOk = WiFi.softAP(AP_SSID, AP_PASS);
  if (!apOk) {
    Serial.println("ERROR: No se pudo iniciar SoftAP");
  }
  IPAddress apIP = WiFi.softAPIP();
  Serial.println("SoftAP activo");
  Serial.print("SSID: ");
  Serial.println(AP_SSID);
  Serial.print("IP ESP32: ");
  Serial.println(apIP);
  server.begin();
  server.setNoDelay(true);
  Serial.println("TCP console lista en puerto 3333");
  Serial.println("Conecta desde PC con: nc " + apIP.toString() + " 3333");
  
  // --- I2C / Sensor ---
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  
  i2cScan();
  
  sensor.setModel(MS5837::MS5837_30BA);
  if (!sensor.init()) {
    Serial.println("ERROR: MS5837 init failed. Revisa Vin/GND/SDA/SCL.");
    while (1) delay(1000);
  }
  
  sensor.setFluidDensity(fluidDensity);
  Serial.println("Bar30 OK.");
  Serial.println("\n=== FanTracker listo ===");
  Serial.println("Usa 'dive <depth>' para iniciar misión");
  if (autoMissionEnabled) {
    nextAutoMissionMs = millis() + autoMissionStartDelayMs;
  } else {
    nextAutoMissionMs = millis();
  }
}

void loop() {
  acceptClient();
  
  // comandos
  String cmd = readClientLine();
  if (cmd.length() > 0) {
    Serial.print("CMD: ");
    Serial.println(cmd);
    handleCommand(cmd);
  }
  
  // procesar misión
  processMission();
  processAutoMission();

  if (sleepPending) {
    unsigned long elapsed = millis() - sleepRequestMs;
    if (elapsed >= POST_MISSION_SLEEP_DELAY_MS) {
      if (loggingEnabled) {
        stopLogging();
      }
      sendLine("[SLEEP] Entrando en deep sleep...");
      client.flush();
      client.stop();
      delay(100);
      if (autoMissionEnabled && autoMissionIntervalMs > 0) {
        esp_sleep_enable_timer_wakeup(static_cast<uint64_t>(autoMissionIntervalMs) * 1000ULL);
      }
      esp_deep_sleep_start();
    }
  }

  // streaming y logging
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last >= samplePeriodMs) {
    last = now;
    
    // streaming a cliente TCP
    if (streamEnabled) {
      String line = readBar30Line();
      if (line.length() > 0) {
        sendLine(line);
      }
    }
    
    // guardar en archivo
    if (loggingEnabled && logFile) {
      String csvLine = readBar30CSV();
      if (csvLine.length() > 0) {
        logFile.println(csvLine);
        logFile.flush();
      }
    }
  }
  
  delay(2);
}
