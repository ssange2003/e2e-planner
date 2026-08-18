#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import time
import threading
import numpy as np
from numba import jit

ANGLE_MIN_NUMBERS = 365
ANGLE_MAX_NUMBERS = 16000
EAI_HEAD_LEN = 10
EAI_FRAME_LEN_MIN = 12
EAI_Si_iMAX = 100
PI = 3.1415926

# Numba가 좋아하는 고정 크기 NumPy 배열 기반의 고속 파싱 및 누적 함수 (기존 연산 효율 100% 유지)
@jit(nopython=True, cache=True)
def parse_and_accumulate(buffer_bytes, temp_scan, ranges_buf, angle_buf, count_ref, N, dth):
    buffer_len = len(buffer_bytes)
    idx_offset = 0
    completed_scan = None
    count = count_ref[0]

    while idx_offset + EAI_FRAME_LEN_MIN <= buffer_len:
        if buffer_bytes[idx_offset] == 0xAA and buffer_bytes[idx_offset+1] == 0x55:
            ct = buffer_bytes[idx_offset+2]
            lsn = buffer_bytes[idx_offset+3]

            if ct == 0x00 and 0 < lsn <= EAI_Si_iMAX:
                frame_len = 2 * lsn + EAI_HEAD_LEN
                if idx_offset + frame_len > buffer_len:
                    break

                fsal = buffer_bytes[idx_offset+4]
                fsah = buffer_bytes[idx_offset+5]
                lsal = buffer_bytes[idx_offset+6]
                lsah = buffer_bytes[idx_offset+7]

                if (fsal & 0x01) and (lsal & 0x01):
                    f_start_angle = (fsal + fsah * 256.0) / 128.0
                    f_stop_angle = (lsal + lsah * 256.0) / 128.0

                    if f_start_angle < 10.0 and count > 300:
                        # 한 바퀴가 다 돌았을 때 360도 배열로 매핑
                        for i in range(count):
                            th = 2.0 * PI - (angle_buf[i] * PI / 180.0)
                            th = th % (2.0 * PI)
                            idx = int(th / dth + 0.5)
                            if idx >= N: idx = 0
                            if idx < 0: idx = 0
                            temp_scan[idx] = ranges_buf[i]

                        completed_scan = temp_scan.copy()
                        count = 0
                        temp_scan.fill(0.0)

                    if f_start_angle > f_stop_angle:
                        f_inc_angle = (360.0 - f_start_angle + f_stop_angle) / (lsn - 1) if lsn > 1 else 0.0
                    else:
                        f_inc_angle = (f_stop_angle - f_start_angle) / (lsn - 1) if lsn > 1 else 0.0

                    for j in range(lsn):
                        if count < len(ranges_buf):
                            si_idx = idx_offset + EAI_HEAD_LEN + (2 * j)
                            temp_depth = (buffer_bytes[si_idx] | (buffer_bytes[si_idx+1] << 8)) / 4.0
                            range_val = temp_depth / 1000.0

                            f_temp_angle = f_start_angle + (j * f_inc_angle)
                            if f_temp_angle > 360.0:
                                f_temp_angle -= 360.0

                            ranges_buf[count] = range_val
                            angle_buf[count] = f_temp_angle
                            count += 1

                idx_offset += frame_len
                continue

        idx_offset += 1

    count_ref[0] = count
    return idx_offset, completed_scan


# 💡 메인 제어루프와 통신할 인터페이스 구축
class LidarSensor:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        self.port = serial.Serial(port, baud_rate, timeout=0.0)
        self.buffer = bytearray()
        
        self.N = 1000
        self.temp_scan = np.zeros(self.N, dtype=np.float32)
        self.dth = 2.0 * PI / self.N
        
        self.max_buf_size = 2000
        self.ranges_buf = np.zeros(self.max_buf_size, dtype=np.float32)
        self.angle_buf = np.zeros(self.max_buf_size, dtype=np.float32)
        self.count_ref = np.zeros(1, dtype=np.int64)

        # 💡 [수정] 5구역 배열(_sectors)을 삭제하고, 1000개의 원본을 저장할 _raw_scan 배열 생성
        self._raw_scan = np.zeros(self.N, dtype=np.float32)
        self._lock = threading.Lock()

        # 백그라운드 데몬 스레드 가동 (메인 루프 절대 방해 안 함)
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _poll(self):
        """기존 poll 함수 (내부 호출용)"""
        if not self.port.is_open:
            return None

        in_waiting = self.port.in_waiting
        if in_waiting > 0:
            self.buffer.extend(self.port.read(in_waiting))

        if len(self.buffer) > 4096:
            self.buffer = self.buffer[-4096:]

        idx_offset, completed_scan = parse_and_accumulate(
            self.buffer, self.temp_scan, self.ranges_buf, self.angle_buf, self.count_ref, self.N, self.dth
        )

        if idx_offset > 0:
            self.buffer = self.buffer[idx_offset:]

        return completed_scan

    def _update_loop(self):
        """무한 루프를 돌며 라이다 원본 데이터를 캐싱"""
        while True:
            scan_data = self._poll()
            if scan_data is not None:
                # 💡 [수정] _process_sectors 함수 호출을 지우고 원본을 바로 덮어씌움
                with self._lock:
                    self._raw_scan = scan_data.copy()
            time.sleep(0.001)

    def get_raw_scan(self) -> np.ndarray:
        """💡 [수정] 외부 모듈(콜렉트, 알고리즘 등)에서 1000개 배열 전체를 가져갈 때 호출하는 함수"""
        with self._lock:
            return self._raw_scan.copy()