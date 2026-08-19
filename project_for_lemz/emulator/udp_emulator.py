#!/usr/bin/env python3
import socket
import json
import time
import math
import random

def main():
    target_ip = "127.0.0.1"
    target_port = 5005
    frequency = 50
    dt = 1.0 / frequency

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet_id = 0
    t = 0.0

    print(f"Эмулятор запущен. Отправка на {target_ip}:{target_port} с частотой {frequency} Гц")
    print("Нажмите Ctrl+C для остановки")

    try:
        while True:
            radius = 2.0
            speed = 0.5
            accel_x = -radius * speed**2 * math.sin(speed * t) + random.gauss(0, 0.1)
            accel_y = radius * speed**2 * math.cos(speed * t) + random.gauss(0, 0.1)
            accel_z = -9.81 + random.gauss(0, 0.05)
            gyro_x = 0.01 * math.sin(0.5 * t) + random.gauss(0, 0.001)
            gyro_y = 0.01 * math.cos(0.7 * t) + random.gauss(0, 0.001)
            gyro_z = 0.1 + random.gauss(0, 0.002)

            packet = {
                "packet_id": packet_id,
                "timestamp": time.time(),
                "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
                "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
                "quat": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
            }

            sock.sendto(json.dumps(packet).encode(), (target_ip, target_port))
            packet_id += 1
            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nЭмулятор остановлен.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()