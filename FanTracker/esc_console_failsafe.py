#!/usr/bin/env python3
import time
import signal
import threading
import pigpio
import argparse
import sys

NEUTRAL = 1500
MIN_US  = 1100
MAX_US  = 1900

class ESCFailsafe:
    def __init__(self, pi, gpio_pwm, arm_seconds=4.0):
        self.pi = pi
        self.gpio = gpio_pwm
        self.arm_seconds = arm_seconds

        self.lock = threading.Lock()
        self.current_us = NEUTRAL
        self.last_input_t = time.monotonic()
        self.last_enable_t = time.monotonic()  # último “refresh” de comando no-neutral
        self.running = True

    def stop_thruster(self):
        # Neutral un momento y luego apaga señal
        self.pi.set_servo_pulsewidth(self.gpio, NEUTRAL)
        time.sleep(0.5)
        self.pi.set_servo_pulsewidth(self.gpio, 0)

    def set_us(self, us):
        us = int(us)
        if us != 0 and (us < MIN_US or us > MAX_US):
            print(f"[WARN] Fuera de rango ({MIN_US}-{MAX_US} us): {us}")
            return

        with self.lock:
            self.last_input_t = time.monotonic()

            # 0 = apaga señal; para el ESC lo más seguro es ir a neutral primero,
            # pero acá respetamos 0 si el usuario lo pide explícitamente.
            if us == 0:
                self.current_us = 0
                self.pi.set_servo_pulsewidth(self.gpio, 0)
                return

            self.current_us = us
            self.pi.set_servo_pulsewidth(self.gpio, us)

            # Si estás mandando algo distinto a neutral, cuenta como “keep-alive”
            if us != NEUTRAL:
                self.last_enable_t = time.monotonic()

    def arm(self):
        print(f"Inicializando ESC: {NEUTRAL}us por {self.arm_seconds:.1f}s ...")
        self.pi.set_servo_pulsewidth(self.gpio, NEUTRAL)
        time.sleep(self.arm_seconds)
        with self.lock:
            self.current_us = NEUTRAL
            now = time.monotonic()
            self.last_input_t = now
            self.last_enable_t = now
        print("ESC inicializado.\n")

    def watchdog_loop(self, max_run_s, idle_timeout_s, check_period=0.1):
        """
        max_run_s: tiempo máximo que se permite mantener un PWM no-neutral sin “refresh”
        idle_timeout_s: si no hay input por este tiempo, se detiene (aunque estés en neutral)
        """
        while self.running:
            time.sleep(check_period)
            now = time.monotonic()

            with self.lock:
                cur = self.current_us
                last_in = self.last_input_t
                last_en = self.last_enable_t

            # Si no hay interacción por mucho tiempo => STOP y salir
            if idle_timeout_s > 0 and (now - last_in) > idle_timeout_s:
                print(f"\n[FAILSAFE] Sin input por {idle_timeout_s}s -> STOP")
                self.stop_thruster()
                self.running = False
                break

            # Si estás en throttle (no neutral) y no se “refresca” en max_run_s => STOP
            if max_run_s > 0 and cur not in (0, NEUTRAL) and (now - last_en) > max_run_s:
                print(f"\n[FAILSAFE] PWM no-neutral sin refresh por {max_run_s}s -> STOP (volviendo a 1500)")
                self.pi.set_servo_pulsewidth(self.gpio, NEUTRAL)
                with self.lock:
                    self.current_us = NEUTRAL
                # no salimos; permitimos que sigas usando consola

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpio", type=int, default=26, help="GPIO BCM para señal PWM (ej: 26 = pin físico 37)")
    ap.add_argument("--max-run", type=float, default=8.0,
                    help="Segundos máximos que permite mantener PWM != 1500 sin re-enviarlo (failsafe). 0 desactiva.")
    ap.add_argument("--idle-timeout", type=float, default=30.0,
                    help="Si no escribes nada por N segundos, STOP y sale. 0 desactiva.")
    ap.add_argument("--arm-seconds", type=float, default=4.0, help="Tiempo inicial en 1500us para armar ESC")
    args = ap.parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        print("No pude conectar a pigpiod. Ejecuta:")
        print("  sudo systemctl start pigpiod")
        sys.exit(1)

    esc = ESCFailsafe(pi, args.gpio, arm_seconds=args.arm_seconds)

    def handle_signal(signum, frame):
        print(f"\n[Signal {signum}] STOP")
        esc.running = False
        try:
            esc.stop_thruster()
        finally:
            pi.stop()
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, handle_signal)

    esc.arm()

    # Lanza watchdog en segundo plano (importante: funciona aunque input() quede bloqueado)
    th = threading.Thread(target=esc.watchdog_loop, args=(args.max_run, args.idle_timeout), daemon=True)
    th.start()

    print("Consola lista.")
    print(" - 1500 = stop, >1500 forward, <1500 reverse")
    print(" - Para mantener un PWM activo, re-envía el mismo valor antes de --max-run segundos.")
    print(" - 'k' = keep-alive (re-envía el PWM actual), 's' = stop (1500), '0' = apaga señal, 'q' = salir\n")

    try:
        while esc.running:
            try:
                s = input("PWM us> ").strip().lower()
            except EOFError:
                # si la entrada se corta, activamos failsafe por idle_timeout y/o paramos
                print("\n[EOF] Entrada cortada -> STOP")
                esc.stop_thruster()
                break

            if s in ("q", "quit", "exit"):
                break
            if s == "k":
                # keep-alive: re-envía el PWM actual y refresca timers
                with esc.lock:
                    cur = esc.current_us
                esc.set_us(cur if cur != 0 else NEUTRAL)
                continue
            if s == "s":
                esc.set_us(NEUTRAL)
                continue

            try:
                esc.set_us(int(s))
            except ValueError:
                print("Ingresa un número (1100-1900), o 'k', 's', '0', 'q'.")
    finally:
        esc.running = False
        esc.stop_thruster()
        pi.stop()
        print("Salida limpia (STOP).")

if __name__ == "__main__":
    main()
