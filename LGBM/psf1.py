# psf1.py  (PSF1: 작업부하)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
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
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:  # 설문조사결과.txt를 UTF-8 인코딩으로 열고 파일 객체 f를 만든다. with문 사용하면 파일이 자동으로 닫힘.
        data = json.load(f) # 파일 객체 f에서 JSON 데이터를 읽어와서 Python 객체로 변환. 여기서는 data 변수에 설문조사 결과가 리스트 형태로 저장된다.

    rows = []

    for resp in data:  # 설문결과 한 라운드를 반복
        for q in resp["questions"]:  # 각 응답 내에 포함된 질문 목록(questions)을 반복
            if q["id"] != 1:  # PSF1만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]

            conds = q["conditions"]  # 질문의 조건들(conditions) 가져오기

            row["근무시간"] = conds["근무시간"]
            row["수면시간"] = conds["수면시간"]
            row["작업시간"] = conds["작업시간"]
            row["근무시간대"] = conds["근무시간대"]
            row["작업 허용 시간"] = conds["작업 허용 시간"]

            rows.append(row)  # 완성된 행(row)을 rows 리스트에 추가
  
    df = pd.DataFrame(rows)
    print("=== [PSF1] 원본 행 개수:", len(df), "===")

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

    # 5. train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 모델 학습
    model = LGBMClassifier(
        objective="multiclass", # 모델이 해결해야 할 목표를 정의함. 이 값은 예측하고자 하는 라벨이 3개 이상일 때 사용하는 다중 클래스 분류 문제임을 지정
        num_class=3, # label: 0,1,2 분류할 클래스의 개수를 지정합니다. 이 설문 데이터에서는 '보통(0)', '나쁨(1)', '매우나쁨(2)'의 세 가지 라벨이 있으므로 3으로 설정되었습니다.
        n_estimators=200, # 학습에 사용할 결정 트리(Decision Tree)의 개수를 지정합니다. 트리가 200개 생성되어 순차적으로 오차를 보정하며 학습을 진행합니다. 이 값이 높으면 성능이 좋아질 수 있지만, 학습 시간이 길어지고 과적합(Overfitting) 위험이 커집니다.
        learning_rate=0.05, # 학습률을 의미하며, 모델이 가중치를 얼마나 큰 폭으로 업데이트할지를 결정하는 가장 중요한 하이퍼파라미터입니다.
        random_state=42, # 랜덤 시드(Seed) 값입니다. 이 값을 고정하면 모델을 여러 번 실행해도 항상 동일한 결과가 나오도록 보장하여, 실험의 재현성(Reproducibility)을 높이는 데 사용됩니다.
    )
    model.fit(X_train, y_train)

    # 7. 성능 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print("\n=== [PSF1] 작업부하 분류 성능 ===")
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred))

    # 8. 중요도
    importances = model.feature_importances_
    print("\n=== [PSF1] Feature Importance ===")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp}")

    # 9. 예시 예측
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

    print("\n=== [PSF1] 새 설문 응답 예측 ===")
    print("예측 결과 (숫자 라벨):", pred_label)
    print("예측 결과 (텍스트):", label012_to_text[pred_label])


if __name__ == "__main__":
    run_psf1()
