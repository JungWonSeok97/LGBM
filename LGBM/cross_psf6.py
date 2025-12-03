# psf6_cv.py  - PSF6(환경) 수준 예측 모델 (교차검증 버전)

import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np


def run_cross_psf6():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    # -----------------------------
    # 0: 좋음, 1: 보통, 2: 나쁨/매우나쁨 계열
    answer_to_label012 = {
        "좋음": 0,
        "약간 높음": 0,   # 다른 PSF에서 쓸 수도 있으니 그냥 유지

        "보통": 1,

        "나쁨": 2,
        "매우나쁨": 2,
        "매우 높음": 2,
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    #    0: 좋은 환경 / 1: 중간 / 2: 나쁜 환경
    # -----------------------------

    rain_map = {
        "비가 10mm/h(눈의 경우 1cm/h) 미만": 0,
        "비가 10mm/s(눈의 경우 1cm/s) 이상": 2,
    }

    noise_map = {
        "소음이 85dB 이하인 경우": 0,
        "소음이 85~115dB인 경우": 1,
        "소음이 115dB 이상인 경우": 2,
    }

    fog_map = {
        "안개가 발생할 경우": 2,
    }

    vent_map = {
        "통풍이 잘 될 경우": 0,
        "통풍이 잘 되지 않을 경우": 2,
    }

    wind_map = {
        "풍속이 10m/s 미만인 경우": 0,
        "풍속 10m/s 이상인 경우": 2,
    }

    dust_map = {
        "황사로 인해 작업자가 숨 쉬기가 불편할 경우": 1,
        "황사로 인해 작업자가 확실히 사물을 볼 수 없는 경우": 2,
    }

    friction_map = {
        "바닥 (정지)마찰계수(SCOF)가 ≥ 0.63인 경우": 0,
        "바닥 (정지)마찰계수(SCOF)가 < 0.63인 경우": 2,
    }

    indoor_hum_map = {
        "실내 습도가 40~75%인 경우": 0,
        "실내 습도가 40% 이하인 경우": 2,
        "실내 습도가 75% 이상인 경우": 2,
    }

    indoor_temp_map = {
        "실내 온도가 17~28℃인 경우": 0,
        "실내 온도가 17℃ 이하인 경우": 2,
        "실내 온도가 28℃ 이상인 경우": 2,
    }

    outdoor_hum_map = {
        "실외 습도가 90% 이하인 경우": 0,
        "실외 습도가 90% 이상인 경우": 2,
    }

    outdoor_temp_map = {
        "실외 온도가 35℃(체감 온도) 이하인 경우": 0,
        "실외 온도가 35℃(체감 온도) 이상인 경우": 2,
    }

    workspace_size_map = {
        "작업공간의 크기가 10m^3 이상인 경우": 0,
        "작업공간의 크기가 10m^3 이하인 경우": 2,
    }

    visibility_map = {
        "정밀작업을 수행할 정도의 조도가 형성된 경우": 0,
        "초정밀작업을 수행할 정도의 조도가 형성된 경우": 0,
        "보통작업을 수행할 정도의 조도가 형성된 경우": 1,
    }

    thunder_map = {
        "낙뢰가 발생할 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 6:   # 환경 PSF(id=6)만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]   # 좋음/보통/나쁨 등

            conds = q["conditions"]

            # 하위 PSF(환경) 항목들
            row["비"] = conds["비"]
            row["소음"] = conds["소음"]
            row["안개"] = conds["안개"]
            row["통풍"] = conds["통풍"]
            row["풍속"] = conds["풍속"]
            row["황사"] = conds["황사"]
            row["마찰계수"] = conds["마찰계수"]
            row["실내 습도"] = conds["실내 습도"]
            row["실내 온도"] = conds["실내 온도"]
            row["실외 습도"] = conds["실외 습도"]
            row["실외 온도"] = conds["실외 온도"]
            row["작업공간 크기"] = conds["작업공간 크기"]
            row["업무공간 가시성"] = conds["업무공간 가시성"]
            row["낙뢰"] = conds["낙뢰"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF6] 원본 행 개수:", len(df), "===")

    # -----------------------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------------------

    # (1) 라벨 문자열 필터링 & 숫자 변환
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    print("\n[PSF6] 라벨 분포:")
    print(df["label"].value_counts())

    # (2) 하위 PSF 조건 숫자화
    df["비"] = df["비"].map(rain_map)
    df["소음"] = df["소음"].map(noise_map)
    df["안개"] = df["안개"].map(fog_map)
    df["통풍"] = df["통풍"].map(vent_map)
    df["풍속"] = df["풍속"].map(wind_map)
    df["황사"] = df["황사"].map(dust_map)
    df["마찰계수"] = df["마찰계수"].map(friction_map)
    df["실내 습도"] = df["실내 습도"].map(indoor_hum_map)
    df["실내 온도"] = df["실내 온도"].map(indoor_temp_map)
    df["실외 습도"] = df["실외 습도"].map(outdoor_hum_map)
    df["실외 온도"] = df["실외 온도"].map(outdoor_temp_map)
    df["작업공간 크기"] = df["작업공간 크기"].map(workspace_size_map)
    df["업무공간 가시성"] = df["업무공간 가시성"].map(visibility_map)
    df["낙뢰"] = df["낙뢰"].map(thunder_map)

    print("\n[PSF6] 인코딩 후 head:")
    print(df.head())

    # 매핑 안 된 값이 있으면 NaN이 되므로 제거
    df = df.dropna().copy()
    print("\n[PSF6] NaN 제거 후 행 개수:", len(df))

    feature_cols = [
        "비",
        "소음",
        "안개",
        "통풍",
        "풍속",
        "황사",
        "마찰계수",
        "실내 습도",
        "실내 온도",
        "실외 습도",
        "실외 온도",
        "작업공간 크기",
        "업무공간 가시성",
        "낙뢰",
    ]

    X = df[feature_cols]
    y = df["label"]

    # -----------------------------
    # 5. 교차검증 설정
    # -----------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,          # 좋음/보통/나쁨 3클래스 기준
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    # -----------------------------
    # 6. 교차검증 Accuracy
    # -----------------------------
    cv_scores = cross_val_score(
        base_model, X, y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n=== [PSF6] 5-Fold 교차검증 Accuracy ===")
    for i, s in enumerate(cv_scores, start=1):
        print(f"Fold {i}: {s * 100:.2f}%")
    print(f"평균 Accuracy: {cv_scores.mean() * 100:.2f}%")
    print()

    # -----------------------------
    # 7. 교차검증 기반 분류 리포트 (out-of-fold 예측)
    # -----------------------------
    y_pred_cv = cross_val_predict(
        base_model, X, y, cv=cv
    )

    print("=== [PSF6] 교차검증 분류 리포트 (out-of-fold) ===")
    acc_cv = accuracy_score(y, y_pred_cv)
    print(f"Accuracy: {acc_cv*100:.2f}%")
    print(classification_report(y, y_pred_cv, digits=3))

    # -----------------------------
    # 8. 최종 모델: 전체 데이터(X, y)로 재학습
    # -----------------------------
    final_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    final_model.fit(X, y)

    # 피처 중요도
    importances = final_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n=== [PSF6] 하위 PSF 중요도 (전체 데이터 학습 기준) ===")
    print(fi_df)

    # -----------------------------
    # 9. 예시: 새 설문 응답 예측 (최종 모델 사용)
    # -----------------------------
    example_conds = {
        "비": "비가 10mm/h(눈의 경우 1cm/h) 미만",
        "소음": "소음이 85~115dB인 경우",
        "안개": "안개가 발생할 경우",
        "통풍": "통풍이 잘 될 경우",
        "풍속": "풍속이 10m/s 미만인 경우",
        "황사": "황사로 인해 작업자가 숨 쉬기가 불편할 경우",
        "마찰계수": "바닥 (정지)마찰계수(SCOF)가 ≥ 0.63인 경우",
        "실내 습도": "실내 습도가 40~75%인 경우",
        "실내 온도": "실내 온도가 17~28℃인 경우",
        "실외 습도": "실외 습도가 90% 이하인 경우",
        "실외 온도": "실외 온도가 35℃(체감 온도) 이하인 경우",
        "작업공간 크기": "작업공간의 크기가 10m^3 이상인 경우",
        "업무공간 가시성": "정밀작업을 수행할 정도의 조도가 형성된 경우",
        "낙뢰": "낙뢰가 발생할 경우",
    }

    x_new = [
        rain_map[example_conds["비"]],
        noise_map[example_conds["소음"]],
        fog_map[example_conds["안개"]],
        vent_map[example_conds["통풍"]],
        wind_map[example_conds["풍속"]],
        dust_map[example_conds["황사"]],
        friction_map[example_conds["마찰계수"]],
        indoor_hum_map[example_conds["실내 습도"]],
        indoor_temp_map[example_conds["실내 온도"]],
        outdoor_hum_map[example_conds["실외 습도"]],
        outdoor_temp_map[example_conds["실외 온도"]],
        workspace_size_map[example_conds["작업공간 크기"]],
        visibility_map[example_conds["업무공간 가시성"]],
        thunder_map[example_conds["낙뢰"]],
    ]

    x_new = np.array(x_new).reshape(1, -1)

    pred_012 = int(final_model.predict(x_new)[0])   # 0,1,2
    pred_123 = pred_012 + 1                         # 1,2,3 으로 변경

    label123_to_text = {
        1: "좋음",
        2: "보통",
        3: "나쁨",
    }

    print("\n=== [PSF6] 예시 설문 응답에 대한 예측 (최종 모델) ===")
    print("예측 결과 (숫자 라벨 0/1/2):", pred_012)
    print("예측 결과 (1~3 변환):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_cross_psf6()
