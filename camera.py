"""
RealSense Camera wrapper for v11 planner scripts.

Interface
---------
    cam = Camera(width=768, height=384, enable_depth=True)
    color_bgr, depth_uint16 = cam.read_frames()
    dist_m = depth_uint16[row, col] * cam.depth_scale

color_bgr    : np.ndarray (H, W, 3) uint8, BGR
depth_uint16 : np.ndarray (H, W)    uint16, raw sensor units
               None when enable_depth=False
cam.depth_scale : float  — multiply raw uint16 by this to get metres
                           (typically 0.001 for D435 / D455)
"""

import math
import statistics
from collections import deque

import numpy as np
import pyrealsense2 as rs


# ─────────────────────────────────────────────────────────────────────────────
# 💡 [추가된 부분: D435i IMU 축 정의]
# ─────────────────────────────────────────────────────────────────────────────
# D435i 의 IMU 좌표계는 depth 센서와 정렬되어 있습니다(librealsense 기준):
#     X = 오른쪽,  Y = 아래,  Z = 전방(광축)
# 따라서 카메라를 정립(正立)으로 전방을 향해 장착했다면
#     중력은 +Y 에 약 +9.81 로 실리고,
#     yaw(좌우 회전) 각속도는 Y 축 자이로에 나타납니다.
#
# [주의] 이 값은 마운트 자세에 따라 달라집니다. 차를 세워둔 채
#     python camera.py --imu-check
# 를 실행하면 어느 축에 중력이 실리는지 즉시 확인할 수 있습니다.
# 결과가 아래 가정과 다르면 이 상수만 고치면 됩니다.
IMU_AXIS_UP   = 1      # 중력이 실리는 축 인덱스 (0=X, 1=Y, 2=Z)
IMU_AXIS_FWD  = 2      # 차량 전진 방향 축 인덱스
IMU_SIGN_FWD  = 1.0    # 전진이 음수로 읽히면 -1.0 으로 바꿉니다

# imu_motion 을 계산할 윈도우 길이(샘플 수).
# accel 은 약 250Hz 로 들어오므로 50샘플 ≈ 0.2초 — 수집 주기(10fps=0.1초)보다
# 약간 길게 잡아 프레임마다 충분한 표본이 쌓이도록 합니다.
IMU_WINDOW = 50


def _dump_device_info(want_w=None, want_h=None, want_fps=None):
    """💡 [추가됨] 연결된 RealSense 장치와 지원 모드를 출력합니다.

    스트림 시작이 실패했을 때 원인을 눈으로 확인하기 위한 진단용입니다.
    """
    try:
        ctx = rs.context()
        devs = list(ctx.devices)
    except Exception as exc:
        print(f"  [진단] RealSense 컨텍스트 생성 실패: {exc}")
        return

    if not devs:
        print("  [진단] 연결된 RealSense 장치가 없습니다.")
        print("         USB 케이블/포트를 확인하세요 (USB3 권장).")
        return

    for d in devs:
        try:
            name = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            usb = d.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            name, serial, usb = "?", "?", "?"
        print(f"  [진단] 장치: {name}  serial={serial}  USB={usb}")
        has_imu = False
        for sensor in d.sensors:
            sname = sensor.get_info(rs.camera_info.name)
            print(f"         sensor: {sname}")
            if "Motion" in sname:
                has_imu = True
            if want_w and "RGB" in sname:
                modes = set()
                for prof in sensor.get_stream_profiles():
                    try:
                        vp = prof.as_video_stream_profile()
                        if prof.stream_type() == rs.stream.color:
                            modes.add((vp.width(), vp.height(), prof.fps()))
                    except Exception:
                        pass
                want = (want_w, want_h, want_fps)
                if want in modes:
                    print(f"         요청 모드 {want} 는 지원됩니다.")
                    print("         -> 다른 프로세스가 카메라를 점유 중일 가능성이 큽니다.")
                else:
                    print(f"         ★ 요청 모드 {want} 미지원.")
                    near = sorted(m for m in modes if m[2] == want_fps)[:6]
                    print(f"         가능한 모드 예: {near}")
        print(f"         IMU(Motion Module): {'있음' if has_imu else '★ 없음 (D435i 아님)'}")


class Camera:
    """RealSense D4xx camera with optional aligned depth stream."""

    def __init__(self, width: int = 640, height: int = 480,
                 enable_depth: bool = False, fps: int = 30,
                 enable_imu: bool = False, enable_color: bool = True):
        # 💡 [추가됨] enable_color=False 로 IMU 단독 기동이 가능합니다.
        # --imu-check 는 영상이 전혀 필요 없는데, 컬러 스트림을 함께 열면
        # 해상도 미지원/USB 대역폭/타 프로세스 점유 같은 영상 쪽 문제로
        # IMU 확인 자체가 막혀버립니다. 두 경로를 분리해 둡니다.
        self._enable_color = enable_color
        self._enable_depth = enable_depth and enable_color

        # 💡 [추가된 부분: IMU 상태 보관용 버퍼]
        # deque 는 GIL 하에서 append 가 원자적이라 콜백 스레드와 메인 루프가
        # 락 없이 안전하게 공유할 수 있습니다.
        self._enable_imu = enable_imu
        self._accel_mag  = deque(maxlen=IMU_WINDOW)   # |accel| 이력
        self._last_accel = (0.0, 0.0, 0.0)
        self._last_gyro  = (0.0, 0.0, 0.0)
        self._imu_pipeline = None

        # 💡 [수정됨] 컬러/depth 파이프라인은 enable_color 일 때만 기동
        self.pipeline = None
        profile = None
        if enable_color:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if self._enable_depth:
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = self.pipeline.start(cfg)
            except Exception as exc:
                # 💡 [추가됨] 실패 원인을 바로 알 수 있게 장치 상태를 덤프합니다.
                # 이 지점에서 죽는 원인은 대부분 셋 중 하나입니다:
                #   1) 다른 프로세스가 카메라를 이미 점유 중
                #   2) 요청한 해상도/포맷/fps 조합을 장치가 지원하지 않음
                #   3) USB 대역폭 부족 (USB2 포트에 연결된 경우)
                print(f"[Camera] 컬러 스트림 시작 실패: {exc}")
                _dump_device_info(width, height, fps)
                raise

        if self._enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale: float = depth_sensor.get_depth_scale()
            self._align = rs.align(rs.stream.color)
        else:
            self.depth_scale: float = 0.001
            self._align = None

        # 💡 [추가된 부분: IMU 전용 파이프라인 기동]
        # [설계 근거] color/depth 와 같은 파이프라인에 모션 스트림을 넣지 않습니다.
        # 30Hz 영상과 200~250Hz 모션을 하나의 wait_for_frames() 로 묶으면
        # 프레임 동기화기(syncer)가 영상 프레임 타이밍에 개입해 제어주기가
        # 흔들릴 수 있기 때문입니다. 별도 파이프라인 + 콜백으로 분리하면
        #   - 메인 루프는 기존과 완전히 동일하게 동작하고(블로킹 지점 불변),
        #   - IMU 가 실패해도 주행은 그대로 계속됩니다(우아한 성능 저하).
        # 메인 루프가 지불하는 비용은 공유 변수에서 float 를 읽는 것뿐입니다.
        if enable_imu:
            try:
                self._imu_pipeline = rs.pipeline()
                icfg = rs.config()
                icfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                icfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
                self._imu_pipeline.start(icfg, self._imu_callback)
                print("[Camera] IMU stream ON  (accel 250Hz / gyro 200Hz, 별도 파이프라인)")
            except Exception as exc:
                # D435(무印) 처럼 IMU 가 없는 모델이거나 펌웨어가 거부한 경우.
                # 여기서 죽으면 주행 자체가 불가능해지므로 반드시 흡수합니다.
                self._imu_pipeline = None
                self._enable_imu = False
                print(f"[Camera] IMU 사용 불가 — IMU 없이 계속합니다 ({exc})")

        # Warm up — discard the first few frames while exposure settles
        if self.pipeline is not None:
            for _ in range(15):
                self.pipeline.wait_for_frames()

        print(f"[Camera] RealSense ready  {width}x{height}  "
              f"depth={'ON  scale=' + str(self.depth_scale) if enable_depth else 'OFF'}")

    def read_frames(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return (color_bgr, depth_uint16).
        depth_uint16 is None when enable_depth=False.
        Blocks until a frame pair is available.
        """
        if self.pipeline is None:
            raise RuntimeError("컬러 스트림이 꺼진 상태입니다 (enable_color=False)")
        frames = self.pipeline.wait_for_frames()

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        color_bgr = np.asanyarray(color_frame.get_data())   # (H, W, 3) uint8 BGR

        depth_uint16 = None
        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_uint16 = np.asanyarray(depth_frame.get_data())  # (H, W) uint16

        return color_bgr, depth_uint16

    # 💡 [추가된 부분: IMU 콜백 — 별도 스레드에서 librealsense 가 호출]
    def _imu_callback(self, frame):
        """모션 프레임 1개를 받아 최신값과 |accel| 이력을 갱신합니다.

        여기서는 적분을 하지 않습니다. 가속도를 적분해 속도를 구하면
        바이어스 드리프트로 수 초 만에 값이 무의미해지기 때문입니다.
        대신 "노면 진동의 세기"를 봅니다 — 차가 구르면 |accel| 이 중력값
        주위에서 흔들리고, 완전히 멈추면 거의 상수가 됩니다. 이 방식은
        누적 오차가 원리적으로 존재하지 않습니다.
        """
        try:
            motion = frame.as_motion_frame()
            if not motion:
                return
            d = motion.get_motion_data()
            if frame.get_profile().stream_type() == rs.stream.accel:
                self._last_accel = (d.x, d.y, d.z)
                self._accel_mag.append(math.sqrt(d.x * d.x + d.y * d.y + d.z * d.z))
            else:
                self._last_gyro = (d.x, d.y, d.z)
        except Exception:
            # 콜백 안에서 예외가 새어나가면 librealsense 스레드가 죽습니다.
            # IMU 한 샘플을 잃는 것보다 스트림이 멈추는 쪽이 훨씬 나쁩니다.
            pass

    # 💡 [추가된 부분: 메인 루프가 매 프레임 호출하는 IMU 파생값 조회]
    def read_imu(self) -> tuple[float, float, float]:
        """(imu_motion, imu_yaw_rate, imu_accel_fwd) 반환.

        IMU 가 없거나 아직 표본이 모이지 않았으면 (0, 0, 0) 을 돌려줍니다.
        0 은 "정지 + 무회전 + 무가속" 과 같은 값이라, 이 경우 증강 단계에서
        IMU 증거를 쓰지 않도록 config 의 IMU_AVAILABLE 판정으로 걸러냅니다.

        연산량: 표준편차 1회(50샘플) + 튜플 인덱싱 2회. 메인 루프 예산
        (추론 약 54ms / 수집 100ms)에 비하면 무시할 수 있는 수준입니다.
        """
        if not self._enable_imu or len(self._accel_mag) < 2:
            return 0.0, 0.0, 0.0
        motion = statistics.pstdev(self._accel_mag)
        yaw    = self._last_gyro[IMU_AXIS_UP]
        fwd    = self._last_accel[IMU_AXIS_FWD] * IMU_SIGN_FWD
        return float(motion), float(yaw), float(fwd)

    def close(self):
        # 💡 [추가됨] IMU 파이프라인도 함께 정리
        if self._imu_pipeline is not None:
            try:
                self._imu_pipeline.stop()
            except Exception:
                pass
            self._imu_pipeline = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# 💡 [추가된 부분: IMU 축 확인 유틸리티]
# ─────────────────────────────────────────────────────────────────────────────
# 사용법:  python camera.py --imu-check
#
# 차를 평평한 곳에 세워둔 채 실행하세요. 중력(약 9.81)이 어느 축에
# 실리는지가 그대로 출력됩니다. 그 축 인덱스를 IMU_AXIS_UP 에 넣고,
# 남은 두 축 중 차량 전진 방향을 IMU_AXIS_FWD 에 넣으면 됩니다.
# 전진 방향은 차를 손으로 앞으로 살짝 밀어보며 어느 축이 양수로 튀는지
# 확인하면 확실합니다(음수로 튀면 IMU_SIGN_FWD = -1.0).
if __name__ == "__main__":
    import sys
    import time as _time

    if "--imu-check" not in sys.argv:
        print("사용법: python camera.py --imu-check")
        sys.exit(0)

    # 💡 [수정됨] 컬러 스트림 없이 IMU 만 연다.
    # 영상 쪽 문제(해상도 미지원/USB 대역폭/타 프로세스 점유)로 IMU 확인이
    # 막히지 않도록 두 경로를 분리했다.
    print("연결된 장치를 확인합니다...")
    _dump_device_info()
    print("")
    cam = Camera(enable_color=False, enable_imu=True)
    if not cam._enable_imu:
        print("")
        print("★ IMU 스트림을 열지 못했습니다. 위 진단 출력에서")
        print("  'IMU(Motion Module): 있음' 인지 먼저 확인하세요.")
        print("  없다면 D435i 가 아니라 D435 이며, IMU 기반 판정은 사용할 수 없습니다.")
        sys.exit(1)
    print("")
    print("5초간 IMU 를 읽습니다. 차를 움직이지 마세요...")
    t0 = _time.time()
    while _time.time() - t0 < 5.0:
        ax, ay, az = cam._last_accel
        gx, gy, gz = cam._last_gyro
        m, yaw, fwd = cam.read_imu()
        print(f"  accel=({ax:+6.2f},{ay:+6.2f},{az:+6.2f})  "
              f"gyro=({gx:+6.3f},{gy:+6.3f},{gz:+6.3f})  "
              f"motion={m:5.3f}", end="\r")
        _time.sleep(0.1)

    ax, ay, az = cam._last_accel
    mags = [abs(ax), abs(ay), abs(az)]
    up = mags.index(max(mags))
    print("")
    print("")
    print(f"  중력이 실린 축 = 인덱스 {up} ({chr(88+up) if up<3 else up}), 값 {[ax,ay,az][up]:+.2f}")
    print(f"  현재 설정 IMU_AXIS_UP = {IMU_AXIS_UP}  "
          f"-> {'일치' if up == IMU_AXIS_UP else '★ 불일치! camera.py 의 IMU_AXIS_UP 을 고치세요'}")
    print(f"  정지 상태 motion = {cam.read_imu()[0]:.4f}")
    print("  → 이 값보다 충분히 큰 값을 augment/config.py 의 IMU_MOTION_THRESH 로 잡으세요.")
    print("  → 이어서 차를 손으로 앞으로 밀어보며 어느 축 accel 이 양수로 튀는지 보고")
    print("     IMU_AXIS_FWD / IMU_SIGN_FWD 를 정하면 됩니다.")
    cam.close()
