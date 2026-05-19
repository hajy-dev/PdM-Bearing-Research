from pathlib import Path

femto_root = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")

print("=== FEMTO 전체 폴더 구조 ===")
for item in sorted(femto_root.rglob("*")):
    # acc_ 파일은 개수만 카운트
    if item.is_dir():
        acc_files = list(item.glob("acc_*.csv"))
        if acc_files:
            print(f"  {item.relative_to(femto_root)} → acc 파일 {len(acc_files)}개")
        else:
            print(f"  {item.relative_to(femto_root)}/")