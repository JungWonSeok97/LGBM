import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier

# 1. PSF별 라벨 매핑
label_map_by_psf = {
    1: {"보통": 0, "나쁨": 1, "매우나쁨": 2},
    2: {"좋음": 0, "보통": 1, "나쁨": 2},
    3: {"좋음": 0, "보통": 1, "나쁨": 2},
    4: {"좋음": 0, "보통": 1, "나쁨": 2},
    5: {"보통": 0, "나쁨": 2},
    6: {"보통": 0, "나쁨": 2},
    7: {"좋음": 0, "보통": 1, "나쁨": 2},
    8: {"보통": 0, "약간 높음": 1, "매우 높음": 2},
}

severity_code_to_text = {
    0: "좋음(상)",
    1: "보통(중)",
    2: "나쁨(하)",
}

# 2. JSON 읽기
with open("설문조사결과.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for resp in data:
    rnd = resp.get("round")
    for q in resp["questions"]:
        psf_id = q.get("id")
        answer_text = q.get("answer")

        if psf_id not in label_map_by_psf:
            continue

        label_map = label_map_by_psf[psf_id]
        if answer_text not in label_map:
            continue

        row = {
            "psf_id": psf_id,
            "round": rnd,
            "answer_text": answer_text,
            "label": label_map[answer_text],
        }

        conds = q.get("conditions", {})
        for k, v in conds.items():
            row[k] = v

        rows.append(row)

df = pd.DataFrame(rows)
print("전체 PSF 질문 수:", len(df))
print("PSF별 데이터 개수:")
print(df["psf_id"].value_counts().sort_index())

# 3. PSF별 학습/평가
for psf_id in sorted(df["psf_id"].unique()):
    print("\n" + "=" * 70)
    print(f"=== PSF {psf_id} 학습/평가 ===")

    df_psf = df[df["psf_id"] == psf_id].copy()
    print("샘플 수:", len(df_psf))
    print("라벨 분포(0=좋음,1=보통,2=나쁨):")
    print(df_psf["label"].value_counts().sort_index())

    if df_psf["label"].nunique() < 2:
        print("⚠️ 라벨이 한 종류뿐이라 건너뜁니다.")
        continue

    drop_cols = ["psf_id", "round", "answer_text", "label"]
    candidate_cols = [c for c in df_psf.columns if c not in drop_cols]
    # 이 PSF에서 실제 값이 있는 컬럼만 feature로
    feature_cols = [c for c in candidate_cols if df_psf[c].notna().any()]

    if not feature_cols:
        print("⚠️ 사용 가능한 feature 없음. 건너뜁니다.")
        continue

    # 조건값 문자열 -> 카테고리 코드
    for col in feature_cols:
        df_psf[col] = df_psf[col].astype("category").cat.codes

    # -1(결측) 행 제거
    mask_valid = (df_psf[feature_cols] != -1).all(axis=1)
    df_psf = df_psf[mask_valid].copy()

    if len(df_psf) < 5:
        print("⚠️ 유효한 샘플이 너무 적어(5개 미만) 학습/평가를 건너뜁니다.")
        continue

    X = df_psf[feature_cols]
    y = df_psf["label"]

    # train / test 분리
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )
        print("⚠️ stratify 없이 분리")

    # 모델 학습
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 성능 평가
    y_pred = model.predict(X_test)
    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"[정확도] {acc * 100:.2f}%")

    # Feature Importance
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n[하위 PSF 중요도]")
    print(fi)

    # 예시 예측 1건
    if len(X_test) > 0:
        sample_idx = X_test.index[0]
        true012 = int(y_test.loc[sample_idx])
        pred012 = int(y_pred[0])
        print("\n[예시 예측 1건]")
        print(f"실제 라벨: {true012} ({severity_code_to_text.get(true012, 'N/A')})")
        print(f"예측 라벨: {pred012} ({severity_code_to_text.get(pred012, 'N/A')})")
