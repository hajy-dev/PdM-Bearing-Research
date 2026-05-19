"""
═══════════════════════════════════════════════════════════════════
  기상청 허브 API 유틸리티
  AI-PASS 예지보전 | KAIST 데이터 환경 보정용
═══════════════════════════════════════════════════════════════════

용도:
  KAIST 베어링 데이터의 파일명에서 날짜/시간을 추출하고,
  해당 시점의 대전(KAIST 소재지) 기상 데이터를 기상청 허브 API에서 조회한다.

API 정보:
  플랫폼:     기상청 API 허브 (apihub.kma.go.kr)
  엔드포인트: https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php
  서비스:     종관기상관측(ASOS) 시간자료
  관측소:     133 (대전광역시 유성구 — KAIST와 같은 구)
  인증:       Query Parameter (authKey=키값)
  인증키:     환경변수 WEATHER_API_KEY에서 로드 (~22자)

응답 형식:
  고정폭 텍스트 (JSON이 아님)
  #으로 시작하는 줄 = 헤더/주석
  데이터 줄 = 공백 구분 컬럼

주요 컬럼:
  TA  → ambient_temp (기온, °C)     — 12번째 컬럼
  WS  → wind_speed   (풍속, m/s)    — 4번째 컬럼
  HM  → humidity     (상대습도, %)  — 14번째 컬럼

캐싱:
  API 호출 결과를 로컬 JSON 파일에 캐싱하여
  동일 날짜 재조회 시 API를 호출하지 않음

사용법:
  from weather_api import fetch_kaist_weather

  weather_df = fetch_kaist_weather(csv_file_paths)
  # 결과: DataFrame [filepath, datetime, ambient_temp, wind_speed, humidity, season]
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 0. 설정
# ══════════════════════════════════════════════
#
# 기상청 허브 API vs 공공데이터포털 차이:
#   - 허브: apihub.kma.go.kr / authKey 파라미터 / 텍스트 응답 / 짧은 키(~22자)
#   - 포털: apis.data.go.kr  / serviceKey 파라미터 / JSON 응답 / 긴 키(~100자)
#
API_URL    = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
STN_ID     = "133"           # 대전 유성구 ASOS 관측소
CACHE_DIR  = r"D:\project\예지보전_v2\weather_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_api_key() -> str:
    """환경변수에서 기상청 허브 API 인증키를 로드한다."""
    key = os.environ.get("WEATHER_API_KEY")
    if not key:
        raise EnvironmentError(
            "환경변수 WEATHER_API_KEY가 설정되지 않았습니다.\n"
            "  Windows:  set WEATHER_API_KEY=발급받은_인증키\n"
            "  PowerShell: $env:WEATHER_API_KEY='발급받은_인증키'\n"
            "  키 발급: https://apihub.kma.go.kr → 마이페이지 → 인증키"
        )
    return key


# ══════════════════════════════════════════════
# 1. 파일명에서 날짜/시간 추출
# ══════════════════════════════════════════════
#
# KAIST 파일명 형식: LogFile_YYYY-MM-DD-HH-mm-ss.csv
# 예: LogFile_2022-06-20-17-00-31.csv → 2022-06-20 17시
#
def parse_datetime_from_filename(filepath: str) -> datetime:
    """
    KAIST CSV 파일명에서 측정 날짜/시간을 추출한다.

    예시:
      LogFile_2022-06-20-17-00-31.csv
      → datetime(2022, 6, 20, 17, 0, 31)
    """
    basename = os.path.basename(filepath)
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', basename)
    if not match:
        raise ValueError(f"파일명에서 날짜를 추출할 수 없습니다: {basename}")
    year, month, day, hour, minute, second = [int(x) for x in match.groups()]
    return datetime(year, month, day, hour, minute, second)


# ══════════════════════════════════════════════
# 2. 기상청 허브 API 호출 (일 단위 캐싱)
# ══════════════════════════════════════════════
#
# 응답 텍스트 구조 (실제 확인됨):
#
#   #START7777
#   # YYMMDDHHMI STN  WD   WS ... TA ... HM ...
#   #        KST  ID  16  m/s ...  C ...  % ...
#   202206200000 133  32  2.0 ... 23.9 ... 80.0 ...
#   202206200100 133  32  1.4 ... 23.9 ... 80.0 ...
#   ...
#   #7777END
#
# 컬럼 인덱스 (공백 split 기준, 0-indexed):
#   0: TM (관측시각, YYYYMMDDHHmm)
#   1: STN (관측소)
#   2: WD (풍향)
#   3: WS (풍속, m/s)        ← wind_speed
#   11: TA (기온, °C)         ← ambient_temp
#   13: HM (상대습도, %)      ← humidity
#
# 참고: -9 또는 -9.0은 결측값을 의미
#
COL_IDX_TA = 11    # 기온 컬럼 인덱스
COL_IDX_WS = 3     # 풍속 컬럼 인덱스
COL_IDX_HM = 13    # 습도 컬럼 인덱스


def _fetch_daily_weather(date_str: str) -> dict:
    """
    특정 날짜의 24시간 기상 데이터를 기상청 허브 API에서 조회한다.

    Args:
        date_str: "20220620" 형식의 날짜 문자열 (YYYYMMDD)

    Returns:
        {"00": {"ta": 23.9, "ws": 2.0, "hm": 80.0},
         "01": {"ta": 23.9, "ws": 1.4, "hm": 80.0},
         ...}

    캐싱:
        D:/project/예지보전_v2/weather_cache/20220620.json
        → 이미 존재하면 API 호출 없이 캐시 반환
    """
    # ── 캐시 확인
    cache_path = os.path.join(CACHE_DIR, f"{date_str}.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        log.debug(f"  [캐시] {date_str} — {len(cached)}시간 데이터 로드")
        return cached

    # ── API 호출
    #    기상청 허브 API 요청 형식:
    #      tm1: 시작 시각 (YYYYMMDDHHmm, 12자리)
    #      tm2: 종료 시각
    #      stn: 관측소 번호
    #      authKey: 인증키
    api_key = _get_api_key()
    url = (
        f"{API_URL}"
        f"?tm1={date_str}0000"    # 해당 날짜 00시 00분
        f"&tm2={date_str}2300"    # 해당 날짜 23시 00분
        f"&stn={STN_ID}"          # 133 = 대전
        f"&authKey={api_key}"
    )

    log.info(f"  [API] 기상청 허브 조회: {date_str} (대전 {STN_ID})")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  [API] 요청 실패: {e}")
        return {}

    # ── 텍스트 응답 파싱
    #    기상청 허브 API는 JSON이 아닌 고정폭 텍스트를 반환
    #    # 으로 시작하는 줄 = 헤더/주석 → 건너뜀
    #    데이터 줄 = 공백으로 구분된 컬럼값
    text = response.text
    hourly = {}

    for line in text.strip().split('\n'):
        line = line.strip()
        # 헤더/주석 줄 건너뛰기
        if not line or line.startswith('#'):
            continue

        cols = line.split()
        if len(cols) < 15:
            continue

        # 첫 번째 컬럼: 관측 시각 (YYYYMMDDHHmm, 12자리)
        tm = cols[0]
        if len(tm) < 10:
            continue

        # 시각 추출 (8~10번째 자리 = HH)
        hour = tm[8:10]

        # 값 추출 및 결측(-9, -9.0) 처리
        ta = _safe_float(cols[COL_IDX_TA])
        ws = _safe_float(cols[COL_IDX_WS])
        hm = _safe_float(cols[COL_IDX_HM])

        hourly[hour] = {
            "ta": ta,     # 기온 (°C)
            "ws": ws,     # 풍속 (m/s)
            "hm": hm,     # 습도 (%)
        }

    # ── 캐시 저장
    if hourly:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(hourly, f, ensure_ascii=False, indent=2)
        log.info(f"  [캐시 저장] {cache_path} ({len(hourly)}시간)")

    # API 호출 간격 유지 (rate limit 대응)
    time.sleep(0.3)

    return hourly


def _safe_float(val) -> float:
    """
    문자열을 안전하게 float로 변환한다.
    기상청 결측값(-9, -9.0)은 NaN으로 처리한다.
    """
    if val is None or val == "":
        return np.nan
    try:
        v = float(val)
        # 기상청 허브 API에서 -9 또는 -9.0은 결측값을 의미
        if v <= -9.0:
            return np.nan
        return v
    except (ValueError, TypeError):
        return np.nan


# ══════════════════════════════════════════════
# 3. 계절 판별
# ══════════════════════════════════════════════
def get_season(month: int) -> str:
    """
    월(month)에서 계절을 반환한다.

    계절 구분 (한국 기준):
      봄: 3~5월,  여름: 6~8월,  가을: 9~11월,  겨울: 12~2월

    계절이 중요한 이유:
      - 여름: 외부 기온 높음 → 기기 냉각 효율 저하 → 기기 온도 상승
      - 겨울: 외부 기온 낮음 → 냉각 효율 양호 → 기기 온도 낮음
      - 같은 기기온도 75°C도 여름에는 정상, 겨울에는 이상일 수 있음
      → 모델이 계절 컨텍스트를 알아야 정확한 판단 가능
    """
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"


# ══════════════════════════════════════════════
# 4. 메인 함수 — KAIST 파일 목록에 기상 데이터 매핑
# ══════════════════════════════════════════════
def fetch_kaist_weather(csv_file_paths: list) -> pd.DataFrame:
    """
    KAIST CSV 파일 목록에서 날짜를 추출하고,
    기상청 허브 API에서 해당 시점의 기상 데이터를 조회하여 DataFrame으로 반환한다.

    Args:
        csv_file_paths: KAIST CSV 파일 경로 리스트 (정렬된 상태)

    Returns:
        DataFrame with columns:
          - filepath:      원본 파일 경로
          - datetime:      측정 시각
          - ambient_temp:  기상청 기온 (°C) — 실험실 col3 대신 사용
          - wind_speed:    풍속 (m/s)
          - humidity:      습도 (%)
          - season:        계절 (spring/summer/autumn/winter)

    동작 흐름:
      1) 파일명에서 날짜 추출
      2) 고유 날짜별로 API 1회 호출 (캐싱 → 재실행 시 즉시)
      3) 각 파일의 시각에 맞는 기상 데이터 매핑
      4) 매핑 실패 시 NaN → 전후 시간 보간(interpolation)
    """
    records = []
    # 날짜별 API 호출 결과 캐시 (메모리 — 같은 날짜 중복 호출 방지)
    daily_cache = {}

    for fp in csv_file_paths:
        try:
            dt = parse_datetime_from_filename(fp)
        except ValueError:
            continue

        # 날짜 문자열 (YYYYMMDD)
        date_str = dt.strftime("%Y%m%d")
        hour_str = f"{dt.hour:02d}"

        # 해당 날짜의 기상 데이터 조회 (날짜별 1회만 API 호출)
        if date_str not in daily_cache:
            daily_cache[date_str] = _fetch_daily_weather(date_str)

        hourly = daily_cache.get(date_str, {})
        weather = hourly.get(hour_str, {})

        records.append({
            "filepath":     fp,
            "datetime":     dt,
            "ambient_temp": weather.get("ta", np.nan),
            "wind_speed":   weather.get("ws", np.nan),
            "humidity":     weather.get("hm", np.nan),
            "season":       get_season(dt.month),
        })

    df = pd.DataFrame(records)

    if df.empty:
        log.warning("  [기상청] 매핑 결과 없음")
        return df

    # ── NaN 보간: 일부 시간대 데이터가 없을 수 있음
    #    → 전후 값으로 선형 보간 후, 양 끝은 ffill/bfill
    for col in ['ambient_temp', 'wind_speed', 'humidity']:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = (df[col].interpolate(method='linear')
                       .bfill().ffill())
            log.info(f"  [보간] {col}: {n_missing}건 NaN → 보간 완료")

    # ── 통계 출력
    log.info(f"  [기상청] 매핑 완료: {len(df)}건")
    log.info(f"    기온: {df['ambient_temp'].min():.1f} ~ {df['ambient_temp'].max():.1f}°C")
    log.info(f"    풍속: {df['wind_speed'].min():.1f} ~ {df['wind_speed'].max():.1f} m/s")
    log.info(f"    습도: {df['humidity'].min():.0f} ~ {df['humidity'].max():.0f}%")
    log.info(f"    계절: {df['season'].unique().tolist()}")

    return df


# ══════════════════════════════════════════════
# 5. 단독 실행 시 테스트
# ══════════════════════════════════════════════
if __name__ == "__main__":
    """
    단독 실행 시: KAIST 데이터 폴더의 파일에 대해 기상 매핑을 테스트한다.
    결과를 weather_mapped.csv로 저장한다.
    """
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    KAIST_DIR = os.path.join(r"D:\project\데이터셋", "Vibration_Bearing_RuntoFailure")
    csv_files = sorted(glob.glob(os.path.join(KAIST_DIR, "**", "*.csv"), recursive=True))

    if not csv_files:
        print("KAIST CSV 파일을 찾을 수 없습니다.")
    else:
        print(f"총 {len(csv_files)}개 파일 발견")
        weather_df = fetch_kaist_weather(csv_files)
        print(weather_df.head(10).to_string())
        print(f"\n총 {len(weather_df)}건 매핑 완료")

        # 저장
        out_path = os.path.join(r"D:\project\예지보전_v2", "weather_mapped.csv")
        weather_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"저장: {out_path}")
