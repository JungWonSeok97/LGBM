# psf1_cv.py  (PSF1: 작업부하, 교차검증 버전)

import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np

def run_cross_psf1():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    # -----------------------------
    answer_to_label012 = {
        "보통": 0,
        "나쁨": 1,
        "매우나쁨": 2,
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    # -----------------------------
    work_time_map = {
        "8시간 미만 근무할 경우": 0,
        "8~12시간 사이에 근무할 경우": 1,
        "12시간 이상 근무할 경우": 2,
    }
    sleep_time_map = {
        "8시간 이상인 경우": 0,
        "6~8시간 사이인 경우": 1,
        "6시간 이내인 경우": 2,
    }
    task_time_map = {
        "근무시간의 50% 미만일 경우": 0,
        "근무시간의 50~80% 사이일 경우": 1,
        "근무시간의 80% 이상일 경우": 2,
    }
    shift_map = {
        "9시~18시인 경우": 0,
        "야간근무 비심야조": 1,
        "야간근무 심야조": 2,
    }
    allowed_time_map = {
        "평상 시와 동일 하거나 느리게 해도 될 만큼의 여유시간이 주어진 상황일 경우": 0,
        "평상 시보다 25%이내로 빠르게 해야 하는 상황일 경우": 1,
        "평상 시보다 25% 이상 빠르게 해야 하는 상황일 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:  # 설문결과 한 라운드를 반복
        for q in resp["questions"]:
            if q["id"] != 1:  # PSF1만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]

            conds = q["conditions"]

            row["근무시간"] = conds["근무시간"]
            row["수면시간"] = conds["수면시간"]
            row["작업시간"] = conds["작업시간"]
            row["근무시간대"] = conds["근무시간대"]
            row["작업 허용 시간"] = conds["작업 허용 시간"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF1] 원본 행 개수:", len(df), "===\n")

    # 4. 인코딩
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    df["근무시간"] = df["근무시간"].map(work_time_map)
    df["수면시간"] = df["수면시간"].map(sleep_time_map)
    df["작업시간"] = df["작업시간"].map(task_time_map)
    df["근무시간대"] = df["근무시간대"].map(shift_map)
    df["작업 허용 시간"] = df["작업 허용 시간"].map(allowed_time_map)

    df = df.dropna().copy()

    feature_cols = ["근무시간", "수면시간", "작업시간", "근무시간대", "작업 허용 시간"]
    X = df[feature_cols]
    y = df["label"]

    # -----------------------------
    # 5. 교차검증 설정
    # -----------------------------
    cv = StratifiedKFold(
        n_splits=5,      # 5-fold 교차검증
        shuffle=True,    # 섞어서 나누기
        random_state=42,
    )

    # 교차검증용 모델 (fold마다 새로 clone됨)
    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    # -----------------------------
    # 6. 교차검증 Accuracy 출력
    # -----------------------------
    cv_scores = cross_val_score(
        base_model, X, y,
        cv=cv,
        scoring="accuracy"
    )

    print("=== [PSF1] 5-Fold 교차검증 Accuracy ===")
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

    print("=== [PSF1] 교차검증 분류 성능 (out-of-fold) ===")
    acc_cv = accuracy_score(y, y_pred_cv) * 100
    print(f"Accuracy: {acc_cv:.2f}%")
    print(classification_report(y, y_pred_cv))

    # -----------------------------
    # 8. 최종 모델: 전체 데이터(X, y)로 재학습
    #    → Feature importance, 새 설문 예측에 사용
    # -----------------------------
    final_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    final_model.fit(X, y)

    importances = final_model.feature_importances_
    print("\n=== [PSF1] Feature Importance (전체 데이터로 학습한 최종 모델) ===")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp}")

    # -----------------------------
    # 9. 예시 예측 (새 설문 응답)
    # -----------------------------
    example_conds = {
        "근무시간": "12시간 이상 근무할 경우",
        "수면시간": "6시간 이내인 경우",
        "작업시간": "근무시간의 80% 이상일 경우",
        "근무시간대": "야간근무 심야조",
        "작업 허용 시간": "평상 시보다 25% 이상 빠르게 해야 하는 상황일 경우",
    }

    x_new = [
        work_time_map[example_conds["근무시간"]],
        sleep_time_map[example_conds["수면시간"]],
        task_time_map[example_conds["작업시간"]],
        shift_map[example_conds["근무시간대"]],
        allowed_time_map[example_conds["작업 허용 시간"]],
    ]
    x_new = np.array(x_new).reshape(1, -1)

    pred_label = int(final_model.predict(x_new)[0])
    label012_to_text = {0: "보통", 1: "나쁨", 2: "매우나쁨"}

    print("\n=== [PSF1] 새 설문 응답 예측 (최종 모델) ===")
    print("입력 조건:", example_conds)
    print("예측 결과 (숫자 라벨):", pred_label)
    print("예측 결과 (텍스트):", label012_to_text[pred_label])


if __name__ == "__main__":
    run_psf1()
