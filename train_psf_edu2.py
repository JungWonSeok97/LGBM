# train_psf_edu.py
# PSF 1~8 전체에 대해 LGBM 학습 + 정확도 + Feature Importance + 예시 예측 출력

import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier

# --------------------------------------------------------
# 1. 라벨 매핑 (answer_text -> 0/1/2)
#    0: 좋음(1), 1: 보통(2), 2: 나쁨(3)
# --------------------------------------------------------
answer_to_label012 = {
    "좋음": 0,
    "약간 높음": 0,   # 필요하면 1로 바꿀 수 있음

    "보통": 1,

    "나쁨": 2,
    "매우나쁨": 2,
    "매우 높음": 2,
}

label123_to_text = {
    1: "좋음",
    2: "보통",
    3: "나쁨",
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
        row = {
            "psf_id": q.get("id"),
            "round": rnd,
            "answer_text": q.get("answer"),
        }
        conds = q.get("conditions", {})
        # conditions dict 안의 key들을 그대로 컬럼으로 사용
        for k, v in conds.items():
            row[k] = v
        rows.append(row)

df = pd.DataFrame(rows)
print("총 question 행 개수:", len(df))
print("발견된 PSF ID:", sorted(df["psf_id"].unique()))

# --------------------------------------------------------
# 3. 라벨 필터링 & 숫자 라벨 생성
# --------------------------------------------------------
df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
df["label"] = df["answer_text"].map(answer_to_label012)

print("라벨이 매핑된 총 행 개수:", len(df))

# --------------------------------------------------------
# 4. PSF별로 학습/평가 반복
# --------------------------------------------------------
for psf_id in sorted(df["psf_id"].unique()):
    print("\n" + "=" * 70)
    print(f"=== PSF {psf_id} 학습/평가 ===")

    df_psf = df[df["psf_id"] == psf_id].copy()
    print("샘플 수:", len(df_psf))
    print("라벨 분포(0:좋음, 1:보통, 2:나쁨):")
    print(df_psf["label"].value_counts())

    # 라벨이 한 종류뿐이면 학습/평가 불가
    if df_psf["label"].nunique() < 2:
        print("⚠️ 이 PSF는 라벨이 한 종류뿐이라 학습/평가를 건너뜁니다.")
        continue

    # feature 컬럼(조건들)만 추출
    drop_cols = ["psf_id", "round", "answer_text", "label"]
    feature_cols = [c for c in df_psf.columns if c not in drop_cols]

    if not feature_cols:
        print("⚠️ 사용 가능한 조건(feature)이 없습니다. 건너뜁니다.")
        continue

    # ----------------------------------------------------
    # 4-1. 조건값 문자열 -> 범주형 코드(0,1,2,...)로 자동 인코딩
    # ----------------------------------------------------
    for col in feature_cols:
        df_psf[col] = df_psf[col].astype("category").cat.codes

    # cat.codes 는 NaN에 -1을 할당하므로, -1 있는 행은 제거
    mask_valid = (df_psf[feature_cols] != -1).all(axis=1)
    df_psf = df_psf[mask_valid].copy()

    if len(df_psf) < 5:
        print("⚠️ 유효한 샘플이 너무 적어(5개 미만) 학습/평가를 건너뜁니다.")
        continue

    X = df_psf[feature_cols]
    y = df_psf["label"]

    # ----------------------------------------------------
    # 4-2. train / test 분리
    # ----------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    except ValueError:
        # stratify가 안 될 경우(데이터 너무 적거나 한 클래스 샘플 부족) 그냥 랜덤 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )
        print("⚠️ stratify 없이 train/test 분리 수행")

    # ----------------------------------------------------
    # 4-3. LGBM 모델 학습
    # ----------------------------------------------------
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ----------------------------------------------------
    # 4-4. 예측 및 성능 평가
    # ----------------------------------------------------
    y_pred = model.predict(X_test)

    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"[정확도] {acc * 100:.2f}%")

    # ----------------------------------------------------
    # 4-5. Feature Importance (하위 PSF 중요도)
    # ----------------------------------------------------
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n[하위 PSF 중요도(Feature Importance)]")
    print(fi)

    # ----------------------------------------------------
    # 4-6. 예시: 테스트 샘플 1개에 대한 실제 vs 예측
    # ----------------------------------------------------
    if len(X_test) > 0:
        sample_idx = X_test.index[0]
        true012 = int(y_test.loc[sample_idx])
        pred012 = int(y_pred[0])

        true123 = true012 + 1
        pred123 = pred012 + 1

        true_text = label123_to_text.get(true123, "알 수 없음")
        pred_text = label123_to_text.get(pred123, "알 수 없음")

        print("\n[예시 예측 1건]")
        print("실제 라벨:",
              f"{true123} ({true_text})")
        print("예측 라벨:",
              f"{pred123} ({pred_text})")
    else:
        print("\n[예시 예측] 테스트 샘플이 없어 생략")
