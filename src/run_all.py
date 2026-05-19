# -*- coding: utf-8 -*-
"""
===================================================================
  AI-PASS 예지보전 -- 전체 비교 실험 순차 실행
===================================================================

실행 순서:
  1. RUL 위험도 4등급 분류 비교  (compare_rul_models.py)
  2. 고장모드 분류 비교           (compare_classification_models.py)
  3. 이상 탐지 비교               (compare_anomaly_models.py)

사용법:
  python run_all.py          <- 전체 실행
  python run_all.py rul      <- RUL만
  python run_all.py clf      <- 분류만
  python run_all.py ad       <- 이상탐지만
===================================================================
"""

import sys
import datetime
import importlib


def run_module(module_name: str, display_name: str):
    """모듈을 import하여 main()을 실행한다."""
    print(f"\n{'='*60}")
    print(f"  [{display_name}] 시작 - {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        # 이미 import된 경우 reload
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        # 로거 핸들러 초기화 (각 모듈이 로거를 재설정하므로)
        import logging
        root = logging.getLogger()
        root.handlers.clear()

        mod.main()
        print(f"\n  [{display_name}] 완료 [OK]")
    except Exception as e:
        print(f"\n  [{display_name}] 실패 [FAIL]: {e}")
        import traceback
        traceback.print_exc()


def main():
    start = datetime.datetime.now()
    print(f"AI-PASS 예지보전 전체 비교 실험")
    print(f"시작: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # 실행할 실험 결정
    args = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    target = args[0].lower()

    tasks = {
        'rul': ('compare_rul_models', 'RUL 위험도 4등급 분류'),
        'clf': ('compare_classification_models', '고장모드 분류'),
        'ad':  ('compare_anomaly_models', '이상 탐지'),
    }

    if target == 'all':
        run_list = ['rul', 'clf', 'ad']
    elif target in tasks:
        run_list = [target]
    else:
        print(f"알 수 없는 인자: {target}")
        print("사용법: python run_all.py [all|rul|clf|ad]")
        return

    for key in run_list:
        module_name, display_name = tasks[key]
        run_module(module_name, display_name)

    elapsed = (datetime.datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  전체 완료 -- 총 소요시간: {elapsed:.1f}분")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
