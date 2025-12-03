# psf8.py  (PSF 8: 인지적 복잡성 / 의사결정)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np


def run_psf8():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    #    0 = 좋음(또는 부담 낮음)
    #    1 = 보통
    #    2 = 나쁨(또는 부담 높음)
    # -----------------------------
    answer_to_label012 = {
        "좋음": 0,
        "약간 높음": 1,   # 인지적 복잡성이 약간 높은 수준은 '아직 괜찮다' 쪽으로 묶음

        "보통": 0,

        "나쁨": 2,
        "매우나쁨": 2,
        "매우 높음": 2,   # 인지적 복잡성이 매우 높음 = 나쁨 쪽으로 묶음
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    #    숫자가 커질수록 '더 복잡/더 나쁜' 방향으로 설정
    # -----------------------------

    # 정보의 불확실성
    info_uncertainty_map = {
        "업무수행 및 의사결정을 위해 필요한 정보의 불확실성이 높지 않은 경우": 0,
        "업무수행 및 의사결정을 위해 필요한 정보의 불확실성이 높은 경우": 2,
        # 중간값(1)은 실제로는 등장하지 않지만, 스케일상 0 < 2 구조를 유지
    }

    # 관측되는 정보의 양
    observed_info_amount_map = {
        "업무수행 및 의사결정을 위해 관측 되어야 하는 정보의 양이 3개 이하일 경우": 0,
        "업무수행 및 의사결정을 위해 관측 되어야 하는 정보의 양이 4~6개일 경우": 1,
        "업무수행 및 의사결정을 위해 관측 되어야 하는 정보의 양이 7개 이상일 경우": 2,
    }

    # 의사결정 대안의 수
    decision_alternatives_map = {
        "의사결정 과정에서 선택할 수 있는 대안의 수가 2건 이하인 경우": 0,
        "의사결정 과정에서 선택할 수 있는 대안의 수가 3~4건인 경우": 1,
        "의사결정 과정에서 선택할 수 있는 대안의 수가 5건 이상인 경우": 2,
    }

    # 의사결정 목표의 명확성
    goal_clarity_map = {
        "의사결정이 필요한 상황에서 의사결정 목표가 명확할 경우": 0,
        "의사결정이 필요한 상황에서 의사결정 목표가 명확하지 않을 경우": 2,
    }

    # 의사결정 판단 기준의 수
    criteria_count_map = {
        "올바른 의사결정을 위해 고려해야 하는 판단기준(예: 시간, 비용, 인력 등)의 수가 1개일 경우": 0,
        "올바른 의사결정을 위해 고려해야 하는 판단기준(예: 시간, 비용, 인력 등)의 수가 2~3개일 경우": 1,
        "올바른 의사결정을 위해 고려해야 하는 판단기준(예: 시간, 비용, 인력 등)의 수가 4개 이상일 경우": 2,
    }

    # (동시에 처리되어야 하는) 의사결정 수
    simul_decisions_map = {
        "동시에 처리되어야 하는 의사결정의 수가 1개일 경우": 0,
        "동시에 처리되어야 하는 의사결정의 수가 2개일 경우": 1,
        "동시에 처리되어야 하는 의사결정의 수가 3개 이상일 경우": 2,
    }

    # (관측이 불가능하여) 추론해야 하는 정보의 양
    inferred_info_amount_map = {
        "업무수행 및 의사결정을 위해 파악하고 있어야 하나 직접적으로 관측되지 않아 관측된 정보를 이용해 추론해야 하는 정보의 양이 1개인 경우": 0,
        "업무수행 및 의사결정을 위해 파악하고 있어야 하나 직접적으로 관측되지 않아 관측된 정보를 이용해 추론해야 하는 정보의 양이 2~4개인 경우": 1,
        "업무수행 및 의사결정을 위해 파악하고 있어야 하나 직접적으로 관측되지 않아 관측된 정보를 이용해 추론해야 하는 정보의 양이 5개 이상인 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 8:   # PSF 8만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]

            conds = q["conditions"]

            # 하위 PSF 항목들
            row["정보의 불확실성"] = conds["정보의 불확실성"]
            row["관측되는 정보의 양"] = conds["관측되는 정보의 양"]
            row["의사결정 대안의 수"] = conds["의사결정 대안의 수"]
            row["의사결정 목표의 명확성"] = conds["의사결정 목표의 명확성"]
            row["의사결정 판단 기준의 수"] = conds["의사결정 판단 기준의 수"]
            row["(동시에 처리 되어야 하는)의사결정 수"] = conds["(동시에 처리 되어야 하는)의사결정 수"]
            row["(관측이 불가능하여)추론해야하는 정보의 양"] = conds["(관측이 불가능하여)추론해야하는 정보의 양"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("원본 행 개수:", len(df))
    print("answer 분포:\n", df["answer_text"].value_counts(), "\n")

    # -----------------------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------------------
    # (1) 라벨 인코딩
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    print("매핑 후 라벨 분포(0/1/2):\n", df["label"].value_counts(), "\n")

    # (2) 하위 PSF 조건 숫자화
    df["정보의 불확실성"] = df["정보의 불확실성"].map(info_uncertainty_map)
    df["관측되는 정보의 양"] = df["관측되는 정보의 양"].map(observed_info_amount_map)
    df["의사결정 대안의 수"] = df["의사결정 대안의 수"].map(decision_alternatives_map)
    df["의사결정 목표의 명확성"] = df["의사결정 목표의 명확성"].map(goal_clarity_map)
    df["의사결정 판단 기준의 수"] = df["의사결정 판단 기준의 수"].map(criteria_count_map)
    df["(동시에 처리 되어야 하는)의사결정 수"] = df["(동시에 처리 되어야 하는)의사결정 수"].map(simul_decisions_map)
    df["(관측이 불가능하여)추론해야하는 정보의 양"] = df["(관측이 불가능하여)추론해야하는 정보의 양"].map(inferred_info_amount_map)

    print(df.head())

    # NaN이 있으면 제거 (혹시 매핑 안 된 값이 있을 경우)
    df = df.dropna().copy()
    print("NaN 제거 후 행 개수:", len(df))

    feature_cols = [
        "정보의 불확실성",
        "관측되는 정보의 양",
        "의사결정 대안의 수",
        "의사결정 목표의 명확성",
        "의사결정 판단 기준의 수",
        "(동시에 처리 되어야 하는)의사결정 수",
        "(관측이 불가능하여)추론해야하는 정보의 양",
    ]

    X = df[feature_cols]
    y = df["label"]

    # -----------------------------
    # 5. train / test 분리
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("train 크기:", X_train.shape, " / test 크기:", X_test.shape)

    # -----------------------------
    # 6. LGBM 모델 학습
    # -----------------------------
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # 7. 성능 평가
    # -----------------------------
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"정확도(accuracy): {acc * 100:.2f}%")

    # -----------------------------
    # 8. 특성 중요도 출력
    # -----------------------------
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 중요도 내림차순

    print("\n=== Feature Importances (중요도 순) ===")
    for idx in indices:
        print(f"{feature_cols[idx]}: {importances[idx]}")

    # -----------------------------
    # 9. 예시: 새 설문 응답 예측
    # -----------------------------
    example_conds = {
        "정보의 불확실성": "업무수행 및 의사결정을 위해 필요한 정보의 불확실성이 높은 경우",
        "관측되는 정보의 양": "업무수행 및 의사결정을 위해 관측 되어야 하는 정보의 양이 7개 이상일 경우",
        "의사결정 대안의 수": "의사결정 과정에서 선택할 수 있는 대안의 수가 5건 이상인 경우",
        "의사결정 목표의 명확성": "의사결정이 필요한 상황에서 의사결정 목표가 명확하지 않을 경우",
        "의사결정 판단 기준의 수": "올바른 의사결정을 위해 고려해야 하는 판단기준(예: 시간, 비용, 인력 등)의 수가 4개 이상일 경우",
        "(동시에 처리 되어야 하는)의사결정 수": "동시에 처리되어야 하는 의사결정의 수가 3개 이상일 경우",
        "(관측이 불가능하여)추론해야하는 정보의 양": "업무수행 및 의사결정을 위해 파악하고 있어야 하나 직접적으로 관측되지 않아 관측된 정보를 이용해 추론해야 하는 정보의 양이 5개 이상인 경우",
    }

    x_new = [
        info_uncertainty_map[example_conds["정보의 불확실성"]],
        observed_info_amount_map[example_conds["관측되는 정보의 양"]],
        decision_alternatives_map[example_conds["의사결정 대안의 수"]],
        goal_clarity_map[example_conds["의사결정 목표의 명확성"]],
        criteria_count_map[example_conds["의사결정 판단 기준의 수"]],
        simul_decisions_map[example_conds["(동시에 처리 되어야 하는)의사결정 수"]],
        inferred_info_amount_map[example_conds["(관측이 불가능하여)추론해야하는 정보의 양"]],
    ]

    x_new = np.array(x_new).reshape(1, -1)

    pred_012 = model.predict(x_new)[0]   # 0,1,2
    pred_123 = int(pred_012) + 1         # 1,2,3 으로 변경

    label123_to_text = {
        1: "좋음",   # 복잡도 낮음 (또는 상태 좋음)
        2: "보통",
        3: "나쁨",   # 복잡도 높음 (또는 상태 나쁨)
    }

    print("\n=== 새 설문 예측 ===")
    print("입력 조건:", example_conds)
    print("예측 결과 (클래스: 1/2/3):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_psf8()
