# train_psf_all.py
# PSF 1~8 전체에 대해 PSF별 라벨/수준을 반영한 LGBM 학습/평가 스크립트

import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective="multiclass",
    n_estimators=200,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,     # 또는 verbose=-1
)

# --------------------------------------------------------
# 1. PSF별 라벨 매핑
#    - 내부 라벨: 0,1,2 (혹은 0,1)
#    - 의미:
#       * PSF 1,2,3,4,7,8 → 3단계 수준(1,2,3) → 0/1/2
#       * PSF 5,6         → 2단계 수준(1,2)   → 0/1
# --------------------------------------------------------
label_map_by_psf = {
    1: {  # PSF1: 보통 / 나쁨 / 매우나쁨
        "보통": 0,        # 수준 1 (상대적으로 양호)
        "나쁨": 1,        # 수준 2
        "매우나쁨": 2,    # 수준 3
    },
    2: {  # PSF2: 좋음 / 보통 / 나쁨
        "좋음": 0,        # 수준 1
        "보통": 1,        # 수준 2
        "나쁨": 2,        # 수준 3
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
    5: {  # PSF5: 수준 1,2 (2단계) → 내부 0,1
        "보통": 0,        # 수준 1
        "나쁨": 1,        # 수준 2
    },
    6: {  # PSF6: 수준 1,2 (2단계) → 내부 0,1
        "보통": 0,        # 수준 1
        "나쁨": 1,        # 수준 2
    },
    7: {  # PSF7: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    8: {  # PSF8: 보통 / 약간 높음 / 매우 높음 (높을수록 나쁨)
        "보통": 0,        # 수준 1
        "약간 높음": 1,   # 수준 2
        "매우 높음": 2,   # 수준 3
    },
}

# 3단계용 기본 텍스트 (PSF 1,2,3,4,7,8)
default_severity_text = {
    0: "좋음(수준 1)",
    1: "보통(수준 2)",
    2: "나쁨(수준 3)",
}

# PSF 5,6 전용 텍스트 (2단계)
severity_text_by_psf = {
    5: {
        0: "보통(수준 1)",
        1: "나쁨(수준 2)",
    },
    6: {
        0: "보통(수준 1)",
        1: "나쁨(수준 2)",
    },
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

        # 1~8번 PSF 중 아닌 건 무시
        if psf_id not in label_map_by_psf:
            continue

        # 이 PSF에서 정의된 라벨에 없는 답변이면 무시
        label_map = label_map_by_psf[psf_id]
        if answer_text not in label_map:
            continue

        row = {
            "psf_id": psf_id,
            "round": rnd,
            "answer_text": answer_text,
            "label": label_map[answer_text],  # 내부 라벨(0/1/2 or 0/1)
        }

        # 하위 PSF 조건들 추가
        conds = q.get("conditions", {})
        for k, v in conds.items():
            row[k] = v

        rows.append(row)

df = pd.DataFrame(rows)
print("전체 PSF 질문 수:", len(df))
print("PSF별 데이터 개수:")
print(df["psf_id"].value_counts().sort_index())

# --------------------------------------------------------
# 3. PSF별로 학습/평가
# --------------------------------------------------------
for psf_id in sorted(df["psf_id"].unique()):
    print("\n" + "=" * 70)
    print(f"=== PSF {psf_id} 학습/평가 ===")

    df_psf = df[df["psf_id"] == psf_id].copy()
    print("샘플 수:", len(df_psf))
    print("라벨 분포:")
    print(df_psf["label"].value_counts().sort_index())

    # 라벨이 1종류뿐이면 학습 불가
    if df_psf["label"].nunique() < 2:
        print("⚠️ 라벨이 한 종류뿐이라 학습/평가를 건너뜁니다.")
        continue

    # 이 PSF에서 실제 값이 있는 feature 컬럼만 사용
    drop_cols = ["psf_id", "round", "answer_text", "label"]
    candidate_cols = [c for c in df_psf.columns if c not in drop_cols]
    feature_cols = [c for c in candidate_cols if df_psf[c].notna().any()]

    if not feature_cols:
        print("⚠️ 사용 가능한 feature(조건)가 없습니다. 건너뜁니다.")
        continue

    # 조건값 문자열 -> 카테고리 코드(0,1,2,...)로 변환
    for col in feature_cols:
        df_psf[col] = df_psf[col].astype("category").cat.codes

    # -1 (결측) 포함된 행 제거
    mask_valid = (df_psf[feature_cols] != -1).all(axis=1)
    df_psf = df_psf[mask_valid].copy()

    if len(df_psf) < 5:
        print("⚠️ 유효한 샘플이 너무 적어(5개 미만) 학습/평가를 건너뜁니다.")
        continue

    X = df_psf[feature_cols]
    y = df_psf["label"]

    # ----------------------------------------------------
    # 3-1. train / test 분리
    # ----------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        # 데이터가 적어서 stratify 실패 시
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )
        print("⚠️ stratify 없이 train/test 분리 수행")

    # ----------------------------------------------------
    # 3-2. 모델 정의 (2클래스 vs 3클래스)
    # ----------------------------------------------------
    n_classes = y.nunique()

    if n_classes == 2:
        # 이 경우는 PSF 5,6 (수준 1,2만 있는 PSF)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        )
    else:
        # 나머지 PSF (3단계 수준)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        )

    # ----------------------------------------------------
    # 3-3. 모델 학습
    # ----------------------------------------------------
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
    # 3-6. 예시 예측 1건 (테스트셋 기준)
    # ----------------------------------------------------
    if len(X_test) > 0:
        sample_idx = X_test.index[0]
        true_label = int(y_test.loc[sample_idx])
        pred_label = int(y_pred[0])

        # PSF별 severity 텍스트 매핑 선택
        sev_map = severity_text_by_psf.get(psf_id, default_severity_text)

        print("\n[예시 예측 1건]")
        print(f"실제 라벨: {true_label} ({sev_map.get(true_label, 'N/A')})")
        print(f"예측 라벨: {pred_label} ({sev_map.get(pred_label, 'N/A')})")
    else:
        print("\n[예시 예측] 테스트 샘플이 없어 생략")
