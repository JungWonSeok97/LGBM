# train_psf_all.py
# PSF 1~8 전체에 대해 PSF별 라벨 매핑을 적용하여 LGBM 학습/평가

import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier

# --------------------------------------------------------
# 1. PSF별 라벨 매핑
#    각 PSF에서 "가장 좋은 상태"를 0, 중간을 1, 가장 나쁜 상태를 2로 맞춤
# --------------------------------------------------------
label_map_by_psf = {
    1: {  # PSF1: 보통 / 나쁨 / 매우나쁨
        "보통": 0,       # 이 PSF에서 가장 양호한 상태
        "나쁨": 1,
        "매우나쁨": 2,
    },
    2: {  # PSF2: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    3: {  # PSF3: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    4: {  # PSF4: 교육/훈련 PSF (좋음 / 보통 / 나쁨)
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    5: {  # PSF5: 보통 / 나쁨 (2단계 구조)
        "보통": 0,
        "나쁨": 2,   # 중간(1)은 사용되지 않음
    },
    6: {  # PSF6: 보통 / 나쁨 (2단계 구조)
        "보통": 0,
        "나쁨": 2,
    },
    7: {  # PSF7: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    8: {  # PSF8: 보통 / 약간 높음 / 매우 높음 (높을수록 나쁨)
        "보통": 0,
        "약간 높음": 1,
        "매우 높음": 2,
    },
}

# 라벨 0/1/2를 보고서용 1/2/3 + 텍스트로 표현하고 싶을 때 참고용
severity_code_to_text = {
    0: "좋음(상)",
    1: "보통(중)",
    2: "나쁨(하)",
}

# --------------------------------------------------------
# 2. JSON 파일 읽기
# --------------------------------------------------------
with open("설문조사결과.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for resp in data:
    rnd = resp.get("round")
    for q in resp["questions"]:
        psf_id = q.get("id")
        answer_text = q.get("answer")

        # 1~8번 PSF만 사용
        if psf_id not in label_map_by_psf:
            continue

        # 해당 PSF에서 정의한 라벨 매핑에 없는 답변이면 건너뜀
        label_map = label_map_by_psf[psf_id]
        if answer_text not in label_map:
            continue

        row = {
            "psf_id": psf_id,
            "round": rnd,
            "answer_text": answer_text,
            "label": label_map[answer_text],  # 0/1/2
        }

        conds = q.get("conditions", {})
        for k, v in conds.items():
            row[k] = v

        rows.append(row)

df = pd.DataFrame(rows)
print("전체 PSF 질문 수:", len(df))
print("PSF별 데이터 개수:")
print(df["psf_id"].value_counts().sort_index())

# --------------------------------------------------------
# 3. PSF별로 반복 학습/평가
# --------------------------------------------------------
for psf_id in sorted(df["psf_id"].unique()):
    print("\n" + "=" * 70)
    print(f"=== PSF {psf_id} 학습/평가 ===")

    df_psf = df[df["psf_id"] == psf_id].copy()
    print("샘플 수:", len(df_psf))
    print("라벨 분포(0=좋음, 1=보통, 2=나쁨):")
    print(df_psf["label"].value_counts().sort_index())

    # 라벨이 1개뿐이면 학습할 수 없음
    if df_psf["label"].nunique() < 2:
        print("⚠️ 이 PSF는 라벨이 한 종류뿐이라 학습/평가를 건너뜁니다.")
        continue

    # feature 컬럼(조건들)만 추출
    drop_cols = ["psf_id", "round", "answer_text", "label"]
    feature_cols = [c for c in df_psf.columns if c not in drop_cols]

    if not feature_cols:
        print("⚠️ 사용 가능한 하위 PSF(조건) feature가 없습니다. 건너뜁니다.")
        continue

    # ----------------------------------------------------
    # 3-1. 조건값 문자열 -> 범주형 코드(0,1,2,...)로 자동 인코딩
    #      (PSF마다 조건 항목과 선택지가 달라서 자동 인코딩 사용)
    # ----------------------------------------------------
    for col in feature_cols:
        df_psf[col] = df_psf[col].astype("category").cat.codes

    # cat.codes는 결측값에 -1을 할당하므로, -1 포함된 행은 제거
    mask_valid = (df_psf[feature_cols] != -1).all(axis=1)
    df_psf = df_psf[mask_valid].copy()

    if len(df_psf) < 5:
        print("⚠️ 유효한 샘플이 너무 적어(5개 미만) 학습/평가를 건너뜁니다.")
        continue

    X = df_psf[feature_cols]
    y = df_psf["label"]

    # ----------------------------------------------------
    # 3-2. train / test 분리
    # ----------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        # 데이터가 적어서 stratify가 안 되는 경우 대비
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )
        print("⚠️ stratify 없이 train/test 분리 수행")

    # ----------------------------------------------------
    # 3-3. LGBM 모델 학습
    # ----------------------------------------------------
    # 일부 PSF는 실제로 2개 라벨만 쓰더라도, num_class=3으로 통일
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ----------------------------------------------------
    # 3-4. 예측 및 성능 평가
    # ----------------------------------------------------
    y_pred = model.predict(X_test)

    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"[정확도] {acc * 100:.2f}%")

    # ----------------------------------------------------
    # 3-5. Feature Importance (하위 PSF 중요도)
    # ----------------------------------------------------
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n[하위 PSF 중요도(Feature Importance)]")
    print(fi)

    # ----------------------------------------------------
    # 3-6. 예시: 테스트 샘플 1건에 대한 실제 vs 예측
    # ----------------------------------------------------
    if len(X_test) > 0:
        sample_idx = X_test.index[0]
        true_label012 = int(y_test.loc[sample_idx])
        pred_label012 = int(y_pred[0])

        print("\n[예시 예측 1건]")
        print(f"실제 라벨: {true_label012} ({severity_code_to_text.get(true_label012, 'N/A')})")
        print(f"예측 라벨: {pred_label012} ({severity_code_to_text.get(pred_label012, 'N/A')})")
    else:
        print("\n[예시 예측] 테스트 샘플이 없어 생략")
