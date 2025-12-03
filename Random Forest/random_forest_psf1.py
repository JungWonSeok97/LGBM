# psf1_rf.py  (PSF1: 작업부하, RandomForest 버전)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier  # ✅ 랜덤포레스트
import numpy as np


def run_psf1():
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
        for q in resp["questions"]:  # 각 응답 내에 포함된 질문 목록(questions)을 반복
            if q["id"] != 1:  # PSF1만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]  # PSF ID 저장
            row["round"] = resp["round"]  # 라운드 정보 저장
            row["answer_text"] = q["answer"]  # 응답 텍스트 저장

            conds = q["conditions"]  # 질문의 조건들(conditions) 가져오기

            row["근무시간"] = conds["근무시간"]
            row["수면시간"] = conds["수면시간"]
            row["작업시간"] = conds["작업시간"]
            row["근무시간대"] = conds["근무시간대"]
            row["작업 허용 시간"] = conds["작업 허용 시간"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF1] 원본 행 개수:", len(df), "===")

    # -----------------------------
    # 4. 인코딩
    # -----------------------------
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
    # 5. train/test 분리
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 6. RandomForest 모델 학습
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=300,   # 트리 개수 (적당히 넉넉하게)
        max_depth=None,    # None이면 자동으로 충분히 분할
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,         # CPU 여러 코어 사용
        class_weight=None, # 라벨 불균형 심하면 'balanced'도 가능
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # 7. 성능 평가
    # -----------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print("\n=== [PSF1][RandomForest] 작업부하 분류 성능 ===")
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 8. Feature Importance
    # -----------------------------
    importances = model.feature_importances_
    print("\n=== [PSF1][RandomForest] Feature Importance ===")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp:.4f}")

    # -----------------------------
    # 9. 예시 예측
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

    pred_label = int(model.predict(x_new)[0])
    label012_to_text = {0: "보통", 1: "나쁨", 2: "매우나쁨"}

    print("\n=== [PSF1][RandomForest] 새 설문 응답 예측 ===")
    print("예측 결과 (숫자 라벨):", pred_label)
    print("예측 결과 (텍스트):", label012_to_text[pred_label])


if __name__ == "__main__":
    run_psf1()
