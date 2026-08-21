"""
dataset_loader.py — CSV 로딩 + 파일명 기반 scenario metadata 추출
=================================================================

파일명 규약:
    {scenario_type}_course{N}.csv
    예) normal_course1.csv, s_curve_course2.csv, recovery_course1.csv

이 파일명이 곧 "사람이 어떤 의도로 수집했는가"에 대한 annotation이다.
센서만으로는 판별이 불가능한 케이스(예: 물리적 증거 없이 사람이 의도적으로
멈춘 경우)를 이 prior가 해결한다.

파일명이 규약에 맞지 않으면 조용히 넘어가지 않고 명확히 경고한다.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import KNOWN_SCENARIOS, SCENARIO_FALLBACK, IMU_COLS as IMU_FILL_COLS


# scenario_type은 s_curve처럼 밑줄을 포함할 수 있으므로 greedy하게 잡고,
# 뒤쪽의 _course{N}을 앵커로 사용한다.
_FILENAME_RE = re.compile(r"^(?P<scenario>.+)_course(?P<course_id>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ScenarioMetadata:
    """파일 하나에 대한 사람의 의도 annotation."""
    scenario_type: str
    course_id: int
    source_file: str

    def __str__(self) -> str:
        return f"{self.scenario_type}/course{self.course_id}"


def parse_filename(path) -> ScenarioMetadata:
    """
    파일명에서 scenario metadata를 추출한다.

    규약에 맞지 않거나 알 수 없는 scenario면 경고 후 SCENARIO_FALLBACK을
    사용한다 — 조용히 무시하면 사용자가 오타를 눈치채지 못한 채
    prior 없이 학습해버리기 때문이다.
    """
    path = Path(path)
    stem = path.stem

    match = _FILENAME_RE.match(stem)
    if not match:
        warnings.warn(
            f"[dataset_loader] 파일명이 규약('{{scenario}}_course{{N}}.csv')에 "
            f"맞지 않습니다: '{path.name}' → scenario='{SCENARIO_FALLBACK}'로 "
            f"처리합니다(파일명 prior 미적용).",
            stacklevel=2,
        )
        return ScenarioMetadata(SCENARIO_FALLBACK, 0, path.name)

    scenario = match.group("scenario").lower()
    course_id = int(match.group("course_id"))

    if scenario not in KNOWN_SCENARIOS:
        warnings.warn(
            f"[dataset_loader] 알 수 없는 scenario '{scenario}' "
            f"(파일: {path.name}). 지원 목록: {', '.join(KNOWN_SCENARIOS)} "
            f"→ '{SCENARIO_FALLBACK}'로 처리합니다.",
            stacklevel=2,
        )
        return ScenarioMetadata(SCENARIO_FALLBACK, course_id, path.name)

    return ScenarioMetadata(scenario, course_id, path.name)


def discover_files(input_dir) -> list:
    """디렉토리에서 CSV를 자동 탐색한다(--input-dir 모드).

    [중요] 규약에 맞는 파일만 받아들인다.

    이전에는 제외 목록(planner_data.csv, augmented_data.csv)에 없는 파일을
    전부 입력으로 삼았는데, 그 방식은 새로운 산출물이 하나 생길 때마다
    조용히 뚫린다. 실제로 augmented_data4.csv / augmented_test.csv /
    planner_data.bak*.csv 가 입력으로 딸려 들어가 이미 증강된 데이터가
    재증강되는 사고가 났다(전체 26799행 중 23128행이 오염).

    그래서 "제외 목록" 대신 "허용 규약"으로 뒤집는다. 자동 탐색은
    {scenario}_course{N}.csv 형식이면서 scenario 가 KNOWN_SCENARIOS 에
    있는 파일만 가져온다. 규약 밖 파일을 일부러 쓰고 싶으면 --inputs 로
    명시하면 되므로 유연성도 잃지 않는다.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")

    files, skipped = [], []
    for path in sorted(input_dir.glob("*.csv")):
        match = _FILENAME_RE.match(path.stem)
        if match and match.group("scenario").lower() in KNOWN_SCENARIOS:
            files.append(path)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"  [SKIP] 규약에 맞지 않아 건너뜀 ({len(skipped)}개): "
              f"{', '.join(skipped[:6])}" + (" ..." if len(skipped) > 6 else ""))
        print(f"         자동 탐색은 {{scenario}}_course{{N}}.csv 형식만 받습니다.")
        print(f"         꼭 포함하려면 --inputs 로 직접 지정하세요.")
    return files


class DatasetLoader:
    """CSV들을 읽어 파일별 그룹 정보와 scenario prior를 붙여 병합한다."""

    def load(self, input_files: list) -> pd.DataFrame:
        frames = []

        for src_idx, file_path in enumerate(input_files):
            path = Path(file_path)
            if not path.exists():
                warnings.warn(f"[dataset_loader] 파일 없음, 건너뜀: {path}")
                continue

            meta = parse_filename(path)

            df = pd.read_csv(path, on_bad_lines="warn")
            df = df.dropna().reset_index(drop=True)
            if df.empty:
                warnings.warn(f"[dataset_loader] 유효 행 없음, 건너뜀: {path}")
                continue

            # _src_idx: 모든 시계열 연산(rolling/diff/shift/interpolate)이
            # 이 그룹 안에서만 일어나야 파일 경계 오염이 없다.
            # (한 번에 assign — 개별 대입은 DataFrame을 단편화시킨다)
            # [중요] IMU 컬럼이 없는 파일은 0 으로 채운다.
            #
            # 채우지 않고 두면 IMU 가 있는 파일과 concat 될 때 NaN 이 되는데,
            # NaN 은 두 곳에서 조용히 사고를 낸다.
            #   1) pandas 에서 NaN == 0.0 은 False 라, sensor_evidence 의
            #      "전부 0이면 IMU 미탑재" 판정이 반대로 뒤집힌다. 그 결과
            #      IMU 가 없는 구 파일이 "완전히 정지해 있었다" 로 읽혀
            #      모든 zero-event 가 진짜 정지로 확정된다.
            #      (실측 사고: normal_course1 이 noise 185 -> stop 205 로 뒤집힘.
            #       정면이 4.45m 로 뚫려 있는데도 정지로 분류되었다)
            #   2) train_planner / saliency 의 dropna() 가 그 행 전체를 지운다.
            #      (실측: planner_data.csv 3671행 -> 1431행)
            #
            # 0 으로 채우면 IMU_ALLZERO_RATIO 판정이 정상 동작해
            # "이 파일에는 IMU 가 없다" 로 올바르게 인식되고, 행도 살아남는다.
            for col in IMU_FILL_COLS:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)

            df = df.assign(
                _src_idx=src_idx,
                _scenario_type=meta.scenario_type,
                _course_id=meta.course_id,
                _source_file=meta.source_file,
            )

            frames.append(df)
            print(f"  [LOAD] {path.name:28s} rows={len(df):5d}  scenario={meta}")

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)