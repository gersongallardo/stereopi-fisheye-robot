#include <Arduino.h>
#include <WiFi.h>
#include "esp_sleep.h"

// ===== WiFi (SoftAP) =====
const char* AP_SSID = "FanTracker-sleep";
const char* AP_PASS = "fantracker123";
WiFiServer server(3333);

// ===== Timers =====
static const uint32_t IDLE_TO_SLEEP_MS = 5UL * 60UL * 1000UL;      // 5 min
static const uint64_t WAKE_AFTER_US    = 5ULL * 60ULL * 1000000ULL; // 5 min

static uint32_t lastActivityMs = 0;

// RTC memory survives deep sleep
static RTC_DATA_ATTR uint32_t bootCount = 0;

void printWakeInfo(Stream &out) {
  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  out.println("\n=== BOOT / WAKE ===");
  out.print("Wakeup cause: ");
  switch (cause) {
    case ESP_SLEEP_WAKEUP_TIMER: out.println("TIMER"); break;
    case ESP_SLEEP_WAKEUP_EXT0:  out.println("EXT0"); break;
    case ESP_SLEEP_WAKEUP_EXT1:  out.println("EXT1"); break;
    case ESP_SLEEP_WAKEUP_GPIO:  out.println("GPIO"); break;
    case ESP_SLEEP_WAKEUP_UART:  out.println("UART"); break;
    default: out.println("UNDEFINED (power-on/reset)"); break;
  }
  bootCount++;
  out.print("Boot count (RTC): ");
  out.println(bootCount);
}

void startSoftAP() {
  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(AP_SSID, AP_PASS);
  delay(200);

  Serial.println("\n=== SoftAP ===");
  Serial.print("softAP start: ");
  Serial.println(ok ? "OK" : "FAIL");

  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP SSID: "); Serial.println(AP_SSID);
  Serial.print("AP PASS: "); Serial.println(AP_PASS);
  Serial.print("AP IP:   "); Serial.println(ip);

  server.begin();
  server.setNoDelay(true);
  Serial.println("TCP server: port 3333");
  Serial.println("Try:  echo status | nc 192.168.4.1 3333");
}

void goToDeepSleep(const char* reason) {
  Serial.println();
  Serial.println("=== GOING TO DEEP SLEEP ===");
  Serial.print("Reason: "); Serial.println(reason);
  Serial.print("Wake after (s): "); Serial.println((uint32_t)(WAKE_AFTER_US / 1000000ULL));
  Serial.flush();

  // Cleanly stop WiFi (optional but nice)
  server.end();
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_OFF);

  // Timer wakeup
  esp_sleep_enable_timer_wakeup(WAKE_AFTER_US);
  delay(200);

  esp_deep_sleep_start();
}

void replyHelp(Stream &out) {
  out.println("Commands:");
  out.println("  status  -> uptime, idle time, clients");
  out.println("  sleep   -> deep sleep now (wake in 5 min)");
  out.println("  help    -> this help");
  out.println("");
  out.println("Notes:");
  out.println("  When it sleeps, WiFi/TCP drops. Reconnect after wake.");
}

void replyStatus(Stream &out) {
  uint32_t now = millis();
  out.print("Uptime(s): "); out.println(now / 1000);
  out.print("Idle(s):   "); out.println((now - lastActivityMs) / 1000);
  out.print("AP IP:     "); out.println(WiFi.softAPIP());
  out.print("Stations:  "); out.println(WiFi.softAPgetStationNum());
  out.print("BootCount: "); out.println(bootCount);
}

String readLineFromClient(WiFiClient &c) {
  // read until newline or timeout
  String line;
  unsigned long t0 = millis();
  while (millis() - t0 < 1000) {
    while (c.available()) {
      char ch = (char)c.read();
      if (ch == '\r') continue;
      if (ch == '\n') {
        line.trim();
        return line;
      }
      line += ch;
    }
    delay(5);
  }
  line.trim();
  return line;
}

void handleClient(WiFiClient &c) {
  c.setTimeout(1);
  c.println("FanTracker sleep test ready. Type 'help'.");

  // Allow multiple commands in one connection
  unsigned long lastSeen = millis();
  while (c.connected()) {
    if (c.available()) {
      String cmd = readLineFromClient(c);
      if (cmd.length() == 0) continue;

      lastActivityMs = millis();
      lastSeen = millis();
      cmd.toLowerCase();

      if (cmd == "help") {
        replyHelp(c);
      } else if (cmd == "status") {
        replyStatus(c);
      } else if (cmd == "sleep") {
        c.println("OK -> sleeping now, reconnect after wake.");
        c.flush();
        delay(50);
        c.stop();
        goToDeepSleep("TCP command");
      } else {
        c.println("Unknown. Try: help");
      }
    }

    // close idle tcp connection after 60s
    if (millis() - lastSeen > 60000) {
      c.println("Closing idle connection.");
      c.stop();
      break;
    }

    delay(10);
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  printWakeInfo(Serial);
  startSoftAP();
  lastActivityMs = millis();

  Serial.println("\nCommands over TCP (examples):");
  Serial.println("  echo status | nc 192.168.4.1 3333");
  Serial.println("  echo sleep  | nc 192.168.4.1 3333");
}

void loop() {
  // Accept new client
  WiFiClient client = server.available();
  if (client) {
    Serial.println("\nClient connected.");
    handleClient(client);
    Serial.println("Client disconnected.");
  }

  // Auto-sleep on inactivity (no TCP commands) after 5 minutes
  if (millis() - lastActivityMs > IDLE_TO_SLEEP_MS) {
    goToDeepSleep("Idle timeout (no TCP activity)");
  }

  // Heartbeat
  static uint32_t lastBeat = 0;
  if (millis() - lastBeat > 10000) {
    lastBeat = millis();
    Serial.print("alive, stations=");
    Serial.print(WiFi.softAPgetStationNum());
    Serial.print(", uptime(s)=");
    Serial.println(millis() / 1000);
  }

  delay(10);
}
