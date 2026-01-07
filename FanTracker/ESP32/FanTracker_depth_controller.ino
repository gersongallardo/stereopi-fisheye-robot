#include <WiFi.h>
#include <Wire.h>
#include <SPIFFS.h>
#include <time.h>
#include <ESP32Servo.h>
#include "MS5837.h"

// ===== WiFi =====
const char* SSID = "losrobles800";
const char* PASS = "losrobles800";
WiFiServer server(3333);
WiFiClient client;

// ===== NTP para sincronizar hora =====
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = -10800;  // Chile: UTC-3
const int daylightOffset_sec = 0;

// ===== MS5837 / Bar30 =====
const int SDA_PIN = 10;
const int SCL_PIN = 1;
MS5837 sensor;

// ===== Motor T200 =====
const int MOTOR_PIN = 2;  // GPIO para PWM del ESC
const int NEUTRAL_US = 1500;
const int MIN_US = 1100;
const int MAX_US = 1900;
Servo thruster;

// ===== Misión =====
enum MissionState { IDLE, ARMING, DESCENT, HOLD, ASCENDING, COMPLETED };
MissionState missionState = IDLE;
float targetDepth = 0.0;
float descentPWM_us = 1600;  // PWM para descender
float holdSeconds = 15.0;    // tiempo manteniendo profundidad
float armSeconds = 4.0;      // tiempo para armar ESC
float surfaceBand = 0.3;     // metros cerca de superficie
float stagnationTimeout = 20.0;  // segundos sin cambio = apagar
float stagnationDelta = 0.05;    // metros mínimos de cambio
unsigned long missionStartTime = 0;
unsigned long holdStartTime = 0;
unsigned long lastDepthChangeTime = 0;
float lastDepth = 0.0;

// ===== Logging =====
String currentLogFile = "/log_actual.csv";
bool loggingEnabled = false;
File logFile;

// ===== Runtime params =====
volatile bool streamEnabled = true;
volatile uint32_t samplePeriodMs = 500;
volatile float fluidDensity = 997.0f;

// ----- timestamp -----
String getTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return String(millis()/1000) + "s";
  }
  char buffer[24];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
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
}

void stopThruster() {
  setThrusterPWM(NEUTRAL_US);
  delay(500);
  thruster.writeMicroseconds(0);
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
    case COMPLETED: return "completed";
    default: return "unknown";
  }
}

String readBar30Line() {
  sensor.read();
  String line;
  line.reserve(120);
  line += getTimestamp();
  line += " [" + getMissionStateStr() + "] ";
  line += "P: ";
  line += String(sensor.pressure(), 1);
  line += " mbar | Prof: ";
  line += String(sensor.depth(), 3);
  line += " m | T: ";
  line += String(sensor.temperature(), 2);
  line += " C";
  return line;
}

String readBar30CSV() {
  sensor.read();
  String line;
  line.reserve(100);
  line += getTimestamp();
  line += ",";
  line += getMissionStateStr();
  line += ",";
  line += String(sensor.pressure(), 2);
  line += ",";
  line += String(sensor.temperature(), 2);
  line += ",";
  line += String(sensor.depth(), 3);
  return line;
}

void startLogging() {
  if (loggingEnabled) {
    sendLine("Log ya está activo");
    return;
  }
  
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    char filename[32];
    strftime(filename, sizeof(filename), "/log_%Y%m%d_%H%M%S.csv", &timeinfo);
    currentLogFile = String(filename);
  } else {
    currentLogFile = "/log_" + String(millis()) + ".csv";
  }
  
  logFile = SPIFFS.open(currentLogFile, FILE_WRITE);
  if (!logFile) {
    sendLine("ERROR: no se pudo crear archivo log");
    return;
  }
  
  logFile.println("timestamp,state,pressure_mbar,temp_C,depth_m");
  loggingEnabled = true;
  sendLine("Logging iniciado: " + currentLogFile);
}

void stopLogging() {
  if (!loggingEnabled) {
    sendLine("Log no está activo");
    return;
  }
  
  if (logFile) {
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
  
  targetDepth = depth_m;
  missionState = ARMING;
  missionStartTime = millis();
  lastDepth = 0.0;
  lastDepthChangeTime = millis();
  
  // Auto-start logging
  if (!loggingEnabled) {
    startLogging();
  }
  
  String msg = "=== MISIÓN INICIADA ===";
  msg += "\nProfundidad objetivo: " + String(targetDepth, 2) + " m";
  msg += "\nPWM descenso: " + String((int)descentPWM_us) + " us";
  msg += "\nHold: " + String(holdSeconds, 1) + " s";
  sendLine(msg);
}

void abortMission() {
  if (missionState == IDLE) {
    sendLine("No hay misión activa");
    return;
  }
  
  stopThruster();
  missionState = COMPLETED;
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
  
  sensor.read();
  float depth = sensor.depth();
  updateStagnationDetection(depth);
  
  unsigned long now = millis();
  float stagnationAge = (now - lastDepthChangeTime) / 1000.0;
  
  // Failsafe: estancamiento bajo el agua
  if (depth > surfaceBand && stagnationAge >= stagnationTimeout) {
    sendLine("[FAILSAFE] Sin cambio de profundidad por " + String(stagnationAge, 1) + 
             "s. Apagando motor para emerger.");
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
        missionState = COMPLETED;
      }
      break;
      
    default:
      break;
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
    client.println("  scan          - escanear I2C");
    client.println("--- Misión ---");
    client.println("  dive <depth>  - iniciar misión a profundidad (ej: dive 5.0)");
    client.println("  abort         - abortar misión actual");
    client.println("  status        - estado de misión");
    client.println("  config <param> <val> - configurar parámetros");
    client.println("    params: descent_pwm, hold_time, arm_time");
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
  
  if (cmd.startsWith("dive ")) {
    float depth = cmd.substring(5).toFloat();
    if (depth <= 0 || depth > 50) {
      client.println("ERROR: profundidad inválida (0-50m)");
      return;
    }
    startMission(depth);
    return;
  }
  
  if (cmd == "abort") {
    abortMission();
    return;
  }
  
  if (cmd == "status") {
    client.println("=== Estado Misión ===");
    client.println("Estado: " + getMissionStateStr());
    client.println("Profundidad objetivo: " + String(targetDepth, 2) + " m");
    client.println("Profundidad actual: " + String(sensor.depth(), 3) + " m");
    client.println("PWM descenso: " + String((int)descentPWM_us) + " us");
    client.println("Hold time: " + String(holdSeconds, 1) + " s");
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
      } else {
        client.println("Parámetro desconocido. Usa: descent_pwm, hold_time, arm_time");
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
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASS);
  
  Serial.print("Conectando a WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(400);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi OK");
    Serial.print("IP ESP32: ");
    Serial.println(WiFi.localIP());
    
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    Serial.println("Sincronizando hora con NTP...");
    delay(2000);
    
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      Serial.print("Hora actual: ");
      Serial.println(getTimestamp());
    }
    
    server.begin();
    server.setNoDelay(true);
    Serial.println("TCP console lista en puerto 3333");
    Serial.println("Conecta desde PC con: nc " + WiFi.localIP().toString() + " 3333");
  } else {
    Serial.println("\nWiFi no disponible - modo standalone");
    Serial.println("El sistema puede operar sin WiFi");
  }
  
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
  
  // streaming y logging
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last >= samplePeriodMs) {
    last = now;
    
    // streaming a cliente TCP
    if (streamEnabled) {
      sendLine(readBar30Line());
    }
    
    // guardar en archivo
    if (loggingEnabled && logFile) {
      logFile.println(readBar30CSV());
      logFile.flush();
    }
  }
  
  delay(2);
}
