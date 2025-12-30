#!/usr/bin/env python3
import time
import signal
import sys
import pigpio
import ms5837
from datetime import datetime

# Constantes PWM
NEUTRAL = 1500
MIN_US = 1100
MAX_US = 1900

# Parámetros de operación
DESCENT_PWM = 1700      # PWM para descender (ajustar según necesidad)
SPIN_PWM = 1550         # PWM suave para girar motores en profundidad
SPIN_DURATION = 7.5     # Segundos girando los motores (5-10s)

# Parámetros de seguridad (failsafe)
STUCK_TIMEOUT = 45.0    # Segundos sin cambio de profundidad antes de activar failsafe
STUCK_THRESHOLD = 0.05  # Cambio mínimo de profundidad (metros) para considerar movimiento
MAX_DESCENT_TIME = 120.0  # Tiempo máximo de descenso antes de abortar (segundos)

class DepthDevice:
    def __init__(self, pi, gpio_pwm, target_depth=1.0, arm_seconds=4.0):
        self.pi = pi
        self.gpio = gpio_pwm
        self.target_depth = target_depth
        self.arm_seconds = arm_seconds
        
        # Estado
        self.running = True
        self.current_depth = 0.0
        self.current_pwm = NEUTRAL
        
        # Inicializar sensor de presión
        print("Inicializando sensor MS5837 (Bar30)...")
        self.sensor = ms5837.MS5837_30BA()
        if not self.sensor.init():
            print("ERROR: No se pudo inicializar el sensor")
            sys.exit(1)
        
        if not self.sensor.read():
            print("ERROR: Fallo la lectura inicial del sensor")
            sys.exit(1)
        
        # Configurar para agua dulce
        self.sensor.setFluidDensity(1000)  # kg/m^3 agua dulce
        
        print(f"✓ Sensor inicializado correctamente")
        print(f"✓ Configurado para AGUA DULCE")
        print(f"  Presión inicial: {self.sensor.pressure():.1f} mbar")
        print(f"  Temperatura: {self.sensor.temperature():.2f} °C")
        print(f"  Profundidad inicial: {self.sensor.depth():.3f} m\n")
    
    def arm_esc(self):
        """Inicializa el ESC enviando señal neutral"""
        print(f"Armando ESC: {NEUTRAL}us por {self.arm_seconds:.1f}s...")
        self.pi.set_servo_pulsewidth(self.gpio, NEUTRAL)
        time.sleep(self.arm_seconds)
        print("✓ ESC armado.\n")
    
    def set_pwm(self, us):
        """Establece el PWM del thruster con límites de seguridad"""
        us = int(us)
        us = max(MIN_US, min(MAX_US, us))  # Limitar rango
        self.current_pwm = us
        self.pi.set_servo_pulsewidth(self.gpio, us)
    
    def stop_motor(self):
        """Detiene el motor (neutral y luego apaga)"""
        self.pi.set_servo_pulsewidth(self.gpio, NEUTRAL)
        time.sleep(0.3)
        self.pi.set_servo_pulsewidth(self.gpio, 0)
    
    def read_depth(self):
        """Lee la profundidad actual del sensor"""
        if self.sensor.read():
            self.current_depth = self.sensor.depth()
            return True
        return False
    
    def descend_to_target(self):
        """Desciende hasta alcanzar la profundidad objetivo"""
        print("=" * 60)
        print("  FASE 1: DESCENSO A PROFUNDIDAD OBJETIVO")
        print("=" * 60)
        print(f"Profundidad objetivo: {self.target_depth:.2f} m")
        print(f"PWM de descenso: {DESCENT_PWM} us")
        print(f"⚠ Failsafe activo: {STUCK_TIMEOUT:.0f}s sin movimiento = EMERGENCIA")
        print(f"⚠ Tiempo máximo de descenso: {MAX_DESCENT_TIME:.0f}s")
        print("\nDescendiendo...\n")
        
        # Activar motor para descender
        self.set_pwm(DESCENT_PWM)
        
        # Variables para detección de atasco
        last_significant_depth = 0.0
        last_movement_time = time.time()
        descent_start_time = time.time()
        
        reached = False
        try:
            while self.running and not reached:
                current_time = time.time()
                
                # Leer profundidad actual
                if not self.read_depth():
                    print("ERROR: Fallo lectura del sensor")
                    time.sleep(0.2)
                    continue
                
                # Verificar si ha habido movimiento significativo
                depth_change = abs(self.current_depth - last_significant_depth)
                if depth_change >= STUCK_THRESHOLD:
                    # Hay movimiento significativo, actualizar referencias
                    last_significant_depth = self.current_depth
                    last_movement_time = current_time
                
                # FAILSAFE 1: Detectar si está atascado (sin movimiento)
                time_stuck = current_time - last_movement_time
                if time_stuck > STUCK_TIMEOUT:
                    print("\n" + "!" * 60)
                    print("  ⚠⚠⚠ FAILSAFE ACTIVADO ⚠⚠⚠")
                    print(f"  Sin movimiento por {time_stuck:.1f}s (umbral: {STUCK_TIMEOUT:.0f}s)")
                    print(f"  Profundidad estable en: {self.current_depth:.3f} m")
                    print("  APAGANDO MOTORES - Ascenso por flotabilidad")
                    print("!" * 60 + "\n")
                    self.stop_motor()
                    return False
                
                # FAILSAFE 2: Tiempo máximo de descenso excedido
                time_descending = current_time - descent_start_time
                if time_descending > MAX_DESCENT_TIME:
                    print("\n" + "!" * 60)
                    print("  ⚠⚠⚠ FAILSAFE ACTIVADO ⚠⚠⚠")
                    print(f"  Tiempo máximo de descenso excedido: {time_descending:.1f}s")
                    print(f"  Profundidad alcanzada: {self.current_depth:.3f} m")
                    print(f"  Objetivo: {self.target_depth:.2f} m (no alcanzado)")
                    print("  APAGANDO MOTORES - Ascenso por flotabilidad")
                    print("!" * 60 + "\n")
                    self.stop_motor()
                    return False
                
                # Mostrar estado
                timestamp = datetime.now().strftime("%H:%M:%S")
                remaining = self.target_depth - self.current_depth
                progress = (self.current_depth / self.target_depth * 100) if self.target_depth > 0 else 0
                
                # Indicador de movimiento
                movement_status = "✓" if time_stuck < 5 else "⚠"
                
                print(f"[{timestamp}] Prof: {self.current_depth:.3f}m | "
                      f"Obj: {self.target_depth:.2f}m | "
                      f"Falta: {remaining:.3f}m | "
                      f"Prog: {progress:.1f}% | "
                      f"{movement_status} Sin mov: {time_stuck:.1f}s")
                
                # Verificar si alcanzó la profundidad objetivo
                if self.current_depth >= self.target_depth:
                    reached = True
                    print(f"\n✓ Profundidad objetivo alcanzada: {self.current_depth:.3f} m\n")
                    break
                
                time.sleep(0.2)  # Leer cada 0.2s
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupción durante descenso...")
            self.stop_motor()
            return False
        
        return reached
    
    def spin_motors(self):
        """Gira los motores lentamente por unos segundos"""
        print("=" * 60)
        print("  FASE 2: ROTACIÓN DE MOTORES EN PROFUNDIDAD")
        print("=" * 60)
        print(f"PWM de rotación: {SPIN_PWM} us")
        print(f"Duración: {SPIN_DURATION:.1f} segundos")
        print(f"⚠ Failsafe activo: {STUCK_TIMEOUT:.0f}s sin movimiento = EMERGENCIA")
        print("\nGirando motores...\n")
        
        # Activar motor suave
        self.set_pwm(SPIN_PWM)
        
        # Variables para detección de atasco durante rotación
        initial_depth = self.current_depth
        last_check_time = time.time()
        last_check_depth = initial_depth
        
        start_time = time.time()
        try:
            while self.running:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = SPIN_DURATION - elapsed
                
                if elapsed >= SPIN_DURATION:
                    break
                
                # Leer profundidad mientras gira
                if self.read_depth():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Verificar cambio significativo de profundidad cada 5 segundos
                    if current_time - last_check_time >= 5.0:
                        depth_change = abs(self.current_depth - last_check_depth)
                        
                        # FAILSAFE: Si se está hundiendo o subiendo mucho durante rotación
                        depth_drift = abs(self.current_depth - initial_depth)
                        if depth_drift > 0.3:  # Más de 30cm de deriva
                            print("\n" + "!" * 60)
                            print("  ⚠⚠⚠ FAILSAFE ACTIVADO ⚠⚠⚠")
                            print(f"  Deriva excesiva de profundidad: {depth_drift:.2f}m")
                            print(f"  Profundidad inicial: {initial_depth:.3f}m")
                            print(f"  Profundidad actual: {self.current_depth:.3f}m")
                            print("  APAGANDO MOTORES - Ascenso por flotabilidad")
                            print("!" * 60 + "\n")
                            self.stop_motor()
                            return False
                        
                        last_check_time = current_time
                        last_check_depth = self.current_depth
                    
                    print(f"[{timestamp}] Girando... {remaining:.1f}s restantes | "
                          f"Prof: {self.current_depth:.3f}m | "
                          f"PWM: {SPIN_PWM}us")
                
                time.sleep(0.5)  # Actualizar cada 0.5s
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupción durante rotación...")
            self.stop_motor()
            return False
        
        print(f"\n✓ Rotación completada ({SPIN_DURATION:.1f}s)\n")
        return True
    
    def surface_ascent(self):
        """Apaga motores y permite ascenso por flotabilidad"""
        print("=" * 60)
        print("  FASE 3: ASCENSO A SUPERFICIE (FLOTABILIDAD)")
        print("=" * 60)
        print("Motores apagados - El dispositivo subirá por flotabilidad\n")
        
        # Apagar motor
        self.stop_motor()
        
        print("Monitoreando ascenso...\n")
        
        try:
            # Monitorear ascenso por 30 segundos o hasta llegar cerca de superficie
            start_time = time.time()
            while self.running and (time.time() - start_time) < 30:
                if self.read_depth():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Subiendo... Profundidad: {self.current_depth:.3f} m | "
                          f"Motor: APAGADO")
                    
                    # Si llegó cerca de la superficie (< 0.1m), terminar
                    if self.current_depth < 0.1:
                        print(f"\n✓ Dispositivo en superficie ({self.current_depth:.3f} m)\n")
                        break
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupción durante ascenso...")
        
        print("=" * 60)
        print("  MISIÓN COMPLETADA")
        print("=" * 60)
    
    def run_mission(self):
        """Ejecuta la secuencia completa: descender, girar, subir"""
        print("\n" + "=" * 60)
        print("  INICIO DE MISIÓN - DISPOSITIVO VERTICAL")
        print("=" * 60)
        print(f"Profundidad objetivo: {self.target_depth} m")
        print(f"Modo: Agua dulce (río)")
        print(f"Secuencia: Descenso → Rotación → Ascenso flotante")
        print("=" * 60 + "\n")
        
        try:
            # FASE 1: Descender a profundidad objetivo
            if not self.descend_to_target():
                print("Misión abortada durante descenso")
                return
            
            # FASE 2: Girar motores lentamente
            if not self.spin_motors():
                print("Misión abortada durante rotación")
                return
            
            # FASE 3: Subir por flotabilidad
            self.surface_ascent()
            
        except Exception as e:
            print(f"\n\nERROR durante misión: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpieza y parada segura"""
        print("\n Limpiando sistema...")
        self.running = False
        self.stop_motor()
        self.pi.stop()
        print("✓ Sistema detenido de forma segura\n")

def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Control de profundidad para dispositivo vertical boyante en río")
    ap.add_argument("--gpio", type=int, default=26, 
                    help="GPIO BCM para señal PWM (default: 26)")
    ap.add_argument("--target", type=float, default=1.0,
                    help="Profundidad objetivo en metros (default: 1.0)")
    ap.add_argument("--descent-pwm", type=int, default=1700,
                    help="PWM para descender (default: 1700)")
    ap.add_argument("--spin-pwm", type=int, default=1550,
                    help="PWM para girar motores en profundidad (default: 1550)")
    ap.add_argument("--spin-time", type=float, default=7.5,
                    help="Segundos girando motores (default: 7.5)")
    ap.add_argument("--arm-seconds", type=float, default=4.0,
                    help="Tiempo de armado del ESC (default: 4.0)")
    ap.add_argument("--saltwater", action="store_true",
                    help="Usar densidad de agua salada (por defecto es agua dulce)")
    ap.add_argument("--stuck-timeout", type=float, default=45.0,
                    help="Segundos sin movimiento antes de failsafe (default: 45)")
    ap.add_argument("--max-descent-time", type=float, default=120.0,
                    help="Tiempo máximo de descenso en segundos (default: 120)")
    
    args = ap.parse_args()
    
    # Actualizar parámetros globales
    global DESCENT_PWM, SPIN_PWM, SPIN_DURATION, STUCK_TIMEOUT, MAX_DESCENT_TIME
    DESCENT_PWM = args.descent_pwm
    SPIN_PWM = args.spin_pwm
    SPIN_DURATION = args.spin_time
    STUCK_TIMEOUT = args.stuck_timeout
    MAX_DESCENT_TIME = args.max_descent_time
    
    # Conectar a pigpiod
    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: No se pudo conectar a pigpiod")
        print("Ejecuta: sudo systemctl start pigpiod")
        sys.exit(1)
    
    # Crear controlador
    device = DepthDevice(pi, args.gpio, args.target, args.arm_seconds)
    
    # Configurar densidad del fluido si se especifica agua salada
    if args.saltwater:
        device.sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)
        print("⚠ Modo cambiado a AGUA SALADA\n")
    
    # Manejar señales de interrupción
    def signal_handler(signum, frame):
        print(f"\n\n⚠ Señal {signum} recibida - Deteniendo...")
        device.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Armar ESC
    device.arm_esc()
    
    # Ejecutar misión
    device.run_mission()

if __name__ == "__main__":
    main()