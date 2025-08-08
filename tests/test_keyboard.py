#!/usr/bin/env python3
# keyboard_test.py

import time
import keyboard

def test_hold(key='i', duration=3):
    print(f"请按住 [{key}] 约 {duration}s，按 [q] 退出测试")
    cnt = 0
    t0 = None
    while True:
        ev = keyboard.read_event()
        if ev.name == 'q' and ev.event_type == keyboard.KEY_DOWN:
            print("退出测试")
            return
        if ev.name == key and ev.event_type == keyboard.KEY_DOWN:
            if t0 is None:
                t0 = time.time()
                print(f"检测到第一次 {key} KEY_DOWN…")
            cnt += 1
        if t0 and time.time() - t0 >= duration:
            break
    dt = time.time() - t0 if t0 else 0
    print(f"{key} KEY_DOWN 捕获 {cnt} 次，持续 {dt:.3f}s，≈ {cnt/dt:.1f} Hz" if dt else "没有捕获到按键")

if __name__=="__main__":
    print("⚠ 请确保 root 权限且 /dev/input/event* 可读")
    test_hold('i', duration=3)
