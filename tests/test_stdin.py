#!/usr/bin/env python3
# stdin_test.py

import sys, termios, tty, select, time

def test_hold(key='i', duration=3):
    print(f"请按住 [{key}] 约 {duration}s 测试 stdin 自动重复频率")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    cnt = 0
    t0 = time.time()
    try:
        tty.setraw(fd)
        while time.time() - t0 < duration:
            r, _, _ = select.select([fd], [], [], 0.01)
            if r and sys.stdin.read(1) == key:
                cnt += 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(f"stdin 捕获 {cnt} 字符，≈ {cnt/duration:.1f} Hz")

if __name__=="__main__":
    test_hold('i', duration=3)
