#!/usr/bin/env python3
"""
Control sencillo para inmersión vertical con sensor Bar30 (MS5837) y un T200.

- Lee profundidad en agua dulce (por defecto) usando ms5837.
- Envía PWM al ESC vía pigpio para descender hasta la profundidad objetivo.
- Mantiene la profundidad reduciendo gradualmente el empuje y luego apaga
  el motor para que el encapsulado suba por flotabilidad.
- Incluye un failsafe de “posición estancada”: si la profundidad no cambia
  durante cierto tiempo bajo la superficie, apaga los motores.
"""
import argparse
import os
import signal
import sys
import time
from datetime import datetime

# Permite usar la copia local de ms5837 incluida en el repositorio.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MS5837_PATH = os.path.join(CURRENT_DIR, "ms5837-python")
if MS5837_PATH not in sys.path:
    sys.path.insert(0, MS5837_PATH)

import ms5837
import pigpio

NEUTRAL_US = 1500
MIN_US = 1100
MAX_US = 1900


class DepthDiveController:
    def __init__(
        self,
        pi,
        sensor,
        gpio_pwm,
        target_depth_m,
        hold_seconds=15.0,
        descent_us=1600,
        sample_period=0.2,
        stagnation_timeout=20.0,
        stagnation_delta=0.05,
        arm_seconds=4.0,
        fluid_density=ms5837.DENSITY_FRESHWATER,
        surface_band_m=0.3,
    ):
        self.pi = pi
        self.sensor = sensor
        self.gpio_pwm = gpio_pwm
        self.target_depth_m = target_depth_m
        self.hold_seconds = hold_seconds
        self.descent_us = self._clamp_us(descent_us)
        self.sample_period = sample_period
        self.stagnation_timeout = stagnation_timeout
        self.stagnation_delta = stagnation_delta
        self.arm_seconds = arm_seconds
        self.surface_band_m = surface_band_m
        self.running = True

        self.sensor.setFluidDensity(fluid_density)
        self._last_depth_change_t = time.monotonic()
        self._last_depth = None

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _clamp_us(self, us):
        return max(MIN_US, min(MAX_US, int(us)))

    def _set_pwm(self, us):
        self.pi.set_servo_pulsewidth(self.gpio_pwm, us)

    def _stop_thruster(self):
        # Neutral breve y luego apaga señal
        self._set_pwm(NEUTRAL_US)
        time.sleep(0.5)
        self._set_pwm(0)

    def _arm_esc(self):
        print(f"[ESC] Armando {self.arm_seconds:.1f}s en {NEUTRAL_US}us ...")
        self._set_pwm(NEUTRAL_US)
        time.sleep(self.arm_seconds)
        print("[ESC] Listo.\n")

    def _read_depth(self):
        if not self.sensor.read():
            raise RuntimeError("Lectura del sensor falló")
        return self.sensor.depth()

    def _update_stagnation(self, depth):
        now = time.monotonic()
        if self._last_depth is None or abs(depth - self._last_depth) >= self.stagnation_delta:
            self._last_depth_change_t = now
            self._last_depth = depth
        return now - self._last_depth_change_t

    def run(self):
        self._arm_esc()

        state = "descent"
        hold_start = None

        print(
            f"Objetivo: {self.target_depth_m:.2f} m | PWM descenso: {self.descent_us}us | "
            f"Hold: {self.hold_seconds:.1f}s"
        )

        try:
            while self.running:
                depth = self._read_depth()
                stagnation_age = self._update_stagnation(depth)

                print(
                    f"[{self._now_str()}] [{state}] "
                    f"P: {self.sensor.pressure():.1f} mbar | "
                    f"Prof: {depth:.3f} m | "
                    f"T: {self.sensor.temperature():.2f} C | "
                    f"Altura: {self.sensor.altitude():.2f} m | "
                    f"Δt sin cambio: {stagnation_age:.1f}s"
                )

                if depth > self.surface_band_m and stagnation_age >= self.stagnation_timeout:
                    print(
                        f"[FAILSAFE] Sin cambio de profundidad por {stagnation_age:.1f}s "
                        f"(>{self.stagnation_timeout}s). Apagando motor para emerger."
                    )
                    break

                if state == "descent":
                    self._set_pwm(self.descent_us)
                    if depth >= self.target_depth_m:
                        state = "hold"
                        hold_start = time.monotonic()
                        print("[INFO] Profundidad alcanzada. Iniciando desaceleración controlada.")
                elif state == "hold":
                    elapsed = time.monotonic() - hold_start
                    if elapsed < self.hold_seconds:
                        # Ramp lineal desde descent_us a NEUTRAL_US
                        ratio = max(0.0, min(1.0, elapsed / self.hold_seconds))
                        target_us = int(self.descent_us + (NEUTRAL_US - self.descent_us) * ratio)
                        self._set_pwm(target_us)
                    else:
                        print("[INFO] Hold completado. Apagando motor para ascenso por flotabilidad.")
                        break

                time.sleep(self.sample_period)
        finally:
            self.running = False
            self._stop_thruster()
            self.pi.stop()
            print("[SALIDA] Motor apagado y pigpio liberado.")


def parse_args():
    ap = argparse.ArgumentParser(description="Bajada controlada con Bar30 + T200 (pigpio).")
    ap.add_argument("--gpio", type=int, default=26, help="GPIO BCM para la señal PWM (ej. 26 = pin 37).")
    ap.add_argument("--target-depth", type=float, default=2.0, help="Profundidad objetivo en metros.")
    ap.add_argument("--hold-seconds", type=float, default=15.0, help="Tiempo manteniendo la profundidad.")
    ap.add_argument("--descent-us", type=int, default=1600, help="PWM us para descender (1100-1900).")
    ap.add_argument("--sample-period", type=float, default=0.2, help="Segundos entre lecturas del sensor.")
    ap.add_argument(
        "--stagnation-timeout",
        type=float,
        default=20.0,
        help="Si no cambia la profundidad por N segundos bajo la superficie, apaga el motor.",
    )
    ap.add_argument(
        "--stagnation-delta",
        type=float,
        default=0.05,
        help="Cambio mínimo de profundidad (m) que resetea el temporizador de estancamiento.",
    )
    ap.add_argument("--arm-seconds", type=float, default=4.0, help="Tiempo inicial en 1500us para armar el ESC.")
    ap.add_argument(
        "--saltwater", action="store_true", help="Usar densidad de agua salada (ms5837.DENSITY_SALTWATER)."
    )
    ap.add_argument(
        "--surface-band",
        type=float,
        default=0.3,
        help="Banda cercana a superficie (m) donde el failsafe por estancamiento no se aplica.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    fluid_density = ms5837.DENSITY_SALTWATER if args.saltwater else ms5837.DENSITY_FRESHWATER

    sensor = ms5837.MS5837_30BA()
    if not sensor.init():
        print("No se pudo inicializar el Bar30 (MS5837).")
        sys.exit(1)
    if not sensor.read():
        print("Lectura inicial falló.")
        sys.exit(1)

    pi = pigpio.pi()
    if not pi.connected:
        print("No se pudo conectar a pigpiod. ¿Está corriendo el servicio?")
        sys.exit(1)

    controller = DepthDiveController(
        pi=pi,
        sensor=sensor,
        gpio_pwm=args.gpio,
        target_depth_m=args.target_depth,
        hold_seconds=args.hold_seconds,
        descent_us=args.descent_us,
        sample_period=args.sample_period,
        stagnation_timeout=args.stagnation_timeout,
        stagnation_delta=args.stagnation_delta,
        arm_seconds=args.arm_seconds,
        fluid_density=fluid_density,
        surface_band_m=args.surface_band,
    )

    def handle_signal(signum, _frame):
        print(f"\n[Signal {signum}] Deteniendo control...")
        controller.running = False

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handle_signal)

    controller.run()


if __name__ == "__main__":
    main()
