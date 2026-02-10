#include <Arduino.h>

// GPIO que controlará el Gate del MOSFET (elige uno libre)
static const int MOSFET_GATE_PIN = 18;

static bool sw_on = false;

// High-side P-MOSFET control:
// ON  -> Gate = 0V (OUTPUT LOW)
// OFF -> Gate = Hi-Z (INPUT), y una R 100k (G<->S) lo sube a VBAT
void setSwitch(bool on) {
  sw_on = on;

  if (on) {
    pinMode(MOSFET_GATE_PIN, OUTPUT);
    digitalWrite(MOSFET_GATE_PIN, LOW);
    Serial.println("SWITCH: ON (circuit closed)");
  } else {
    pinMode(MOSFET_GATE_PIN, INPUT);  // alta impedancia
    Serial.println("SWITCH: OFF (circuit open)");
  }
}

void printHelp() {
  Serial.println("\n=== MOSFET High-Side Switch Control (Serial) ===");
  Serial.println("Commands:");
  Serial.println("  on        -> close circuit (turn ON)");
  Serial.println("  off       -> open circuit (turn OFF)");
  Serial.println("  toggle    -> toggle state");
  Serial.println("  pulse X   -> ON for X ms then OFF (e.g. pulse 1000)");
  Serial.println("  status    -> show state");
  Serial.println("  help      -> show this help");
  Serial.println("------------------------------------------------");
  Serial.println("Wiring notes:");
  Serial.println("  - MUST have 100k between Gate and Source (G<->S)");
  Serial.println("  - ESP32 GND must be common with battery/Oxy GND");
}

String readLine() {
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      String out = line;
      line = "";
      out.trim();
      return out;
    }
    line += c;
  }
  return "";
}

void setup() {
  Serial.begin(115200);
  delay(200);

  // OFF por defecto
  pinMode(MOSFET_GATE_PIN, INPUT);
  sw_on = false;

  printHelp();
  Serial.println("Ready.");
}

void loop() {
  String cmd = readLine();
  if (cmd.length() == 0) return;
  cmd.toLowerCase();

  if (cmd == "help") {
    printHelp();
  } else if (cmd == "on") {
    setSwitch(true);
  } else if (cmd == "off") {
    setSwitch(false);
  } else if (cmd == "toggle") {
    setSwitch(!sw_on);
  } else if (cmd == "status") {
    Serial.print("State: ");
    Serial.println(sw_on ? "ON" : "OFF");
  } else if (cmd.startsWith("pulse")) {
    int ms = 500;
    int sp = cmd.indexOf(' ');
    if (sp > 0) ms = cmd.substring(sp + 1).toInt();
    if (ms < 10) ms = 10;
    if (ms > 10000) ms = 10000;

    Serial.print("Pulse ON for ");
    Serial.print(ms);
    Serial.println(" ms...");
    setSwitch(true);
    delay(ms);
    setSwitch(false);
  } else {
    Serial.println("Unknown command. Type 'help'.");
  }
}
