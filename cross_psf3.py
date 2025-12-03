# psf3_cv.py  (PSF3: 직무 절차, 교차검증 버전)

import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np


def run_cross_psf3():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    # -----------------------------
    answer_to_label012 = {
        "좋음": 0,
        "약간 높음": 0,   # 필요하면 1로 바꿔도 됨

        "보통": 1,

        "나쁨": 2,
        "매우나쁨": 2,
        "매우 높음": 2,
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    #    PSF3: 직무 절차 관련
    # -----------------------------
    consistency_map = {
        "직무 절차서 내용이 매우 일관되게 표현되어 있는 경우": 0,
        "직무 절차서 내용이 전반적으로 일관되게 표현되어 있는 경우": 1,
        "직무 절차서 내용의 일관성이 부족한 경우": 2,
    }

    traceability_map = {
        "직무 절차서 추적성이 높은 경우": 0,
        "직무 절차서 추적성이 보통인 경우": 1,
        "직무 절차서 추적성이 낮은 경우": 2,
    }

    availability_map = {
        "직무를 수행하는 데 필요한 절차의 가용성(접근성)이 높은 경우": 0,
        # 보통 값은 현재 데이터에 없음
        "직무를 수행하는 데 필요한 절차의 가용성(접근성)이 낮은 경우": 2,
    }

    clarity_map = {
        "직무 절차서 내용의 명료성이 높은 경우": 0,
        "직무 절차서 내용의 명료성이 보통인 경우": 1,
        "직무 절차서 내용의 명료성이 낮은 경우": 2,
    }

    adequacy_map = {
        "직무 절차서 내용이 매우 적절하게 업무에 도움이 되는 경우": 0,
        "직무 절차서 내용이 전반적으로 적절하게 업무에 도움이 되는 경우": 1,
        "직무 절차서 내용이 부적절하거나 미흡하게 제공되는 경우": 2,
    }

    complexity_map = {
        "직무를 수행하는 데 필요한 절차의 복잡성이 낮은 경우": 0,
        "직무를 수행하는 데 필요한 절차의 복잡성이 보통인 경우": 1,
        "직무를 수행하는 데 필요한 절차의 복잡성이 높은 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 3:   # PSF3 (직무 절차)만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp.get("round")
            row["answer_text"] = q["answer"]

            conds = q["conditions"]

            # 하위 PSF 항목들
            row["(직무)절차의 일관성"] = conds["(직무)절차의 일관성"]
            row["(직무)절차의 추적성"] = conds["(직무)절차의 추적성"]
            row["(직무)절차의 유무(가용성)"] = conds["(직무)절차의 유무(가용성)"]
            row["(직무)절차의 명료성(구체성)"] = conds["(직무)절차의 명료성(구체성)"]
            row["(직무)절차의 적절성(완전성 포함)"] = conds["(직무)절차의 적절성(완전성 포함)"]
            row["(직무)절차 기술(description)의 복잡성"] = conds["(직무)절차 기술(description)의 복잡성"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF3] 원본 행 개수:", len(df), "===")

    # -----------------------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------------------
    # (1) 라벨
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    # (2) 하위 PSF 조건 숫자화
    df["(직무)절차의 일관성"] = df["(직무)절차의 일관성"].map(consistency_map)
    df["(직무)절차의 추적성"] = df["(직무)절차의 추적성"].map(traceability_map)
    df["(직무)절차의 유무(가용성)"] = df["(직무)절차의 유무(가용성)"].map(availability_map)
    df["(직무)절차의 명료성(구체성)"] = df["(직무)절차의 명료성(구체성)"].map(clarity_map)
    df["(직무)절차의 적절성(완전성 포함)"] = df["(직무)절차의 적절성(완전성 포함)"].map(adequacy_map)
    df["(직무)절차 기술(description)의 복잡성"] = df["(직무)절차 기술(description)의 복잡성"].map(complexity_map)

    print(df.head())

    # NaN 제거 (매핑 안된 값이 있으면)
    df = df.dropna().copy()

    feature_cols = [
        "(직무)절차의 일관성",
        "(직무)절차의 추적성",
        "(직무)절차의 유무(가용성)",
        "(직무)절차의 명료성(구체성)",
        "(직무)절차의 적절성(완전성 포함)",
        "(직무)절차 기술(description)의 복잡성",
    ]

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

    # fold마다 새로 clone되는 베이스 모델
    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.01,   # 기존 코드 값 유지
        random_state=42,
    )

    # -----------------------------
    # 6. 교차검증 Accuracy (fold별 & 평균)
    # -----------------------------
    cv_scores = cross_val_score(
        base_model, X, y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n=== [PSF3] 5-Fold 교차검증 Accuracy ===")
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

    print("=== [PSF3] 교차검증 분류 성능 (out-of-fold) ===")
    acc_cv = accuracy_score(y, y_pred_cv) * 100
    print(f"Accuracy: {acc_cv:.2f}%")
    print(classification_report(y, y_pred_cv, digits=3))

    # -----------------------------
    # 8. 최종 모델: 전체 데이터(X, y)로 재학습
    #    → Feature importance, 새 설문 예측에 사용
    # -----------------------------
    final_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.01,
        random_state=42,
    )
    final_model.fit(X, y)

    print("\n=== [PSF3] Feature Importance (전체 데이터로 학습한 최종 모델) ===")
    importances = final_model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    for name, imp in fi.items():
        print(f"{name}: {imp}")

    # -----------------------------
    # 9. 예시: 새 설문 응답 예측 (최종 모델 사용)
    # -----------------------------
    example_conds = {
        "(직무)절차의 일관성": "직무 절차서 내용이 전반적으로 일관되게 표현되어 있는 경우",
        "(직무)절차의 추적성": "직무 절차서 추적성이 보통인 경우",
        "(직무)절차의 유무(가용성)": "직무를 수행하는 데 필요한 절차의 가용성(접근성)이 높은 경우",
        "(직무)절차의 명료성(구체성)": "직무 절차서 내용의 명료성이 낮은 경우",
        "(직무)절차의 적절성(완전성 포함)": "직무 절차서 내용이 전반적으로 적절하게 업무에 도움이 되는 경우",
        "(직무)절차 기술(description)의 복잡성": "직무를 수행하는 데 필요한 절차의 복잡성이 높은 경우",
    }

    x_new_list = [
        consistency_map[example_conds["(직무)절차의 일관성"]],
        traceability_map[example_conds["(직무)절차의 추적성"]],
        availability_map[example_conds["(직무)절차의 유무(가용성)"]],
        clarity_map[example_conds["(직무)절차의 명료성(구체성)"]],
        adequacy_map[example_conds["(직무)절차의 적절성(완전성 포함)"]],
        complexity_map[example_conds["(직무)절차 기술(description)의 복잡성"]],
    ]

    x_new = np.array(x_new_list).reshape(1, -1)

    pred_012 = int(final_model.predict(x_new)[0])   # 0,1,2
    pred_123 = pred_012 + 1                         # 1,2,3으로 변경

    label123_to_text = {
        1: "좋음",
        2: "보통",
        3: "나쁨",
    }

    print("\n=== [PSF3] 새 설문 응답 예측 (최종 모델) ===")
    print("입력 조건:", example_conds)
    print("예측 결과 (숫자 라벨 1~3):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_cross_psf3()
