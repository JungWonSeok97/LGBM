# psf7.py
# PSF 7 (역할 분담/작업 계획/협업/감독) RandomForest 학습 + 정확도 + 특성 중요도 + 예시 예측

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier   # ✅ LGBM 대신 RandomForest 사용


def run_psf7():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    # -----------------------------
    # 0: 좋음(1), 1: 보통(2), 2: 나쁨(3)
    answer_to_label012 = {
        "좋음": 0,

        "보통": 1,

        "나쁨": 2,
        "매우나쁨": 2,
        "약간 높음": 0,   # 혹시 들어와도 대비
        "매우 높음": 2,   # 혹시 들어와도 대비
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    #    PSF7 하위 PSF들
    # -----------------------------
    role_division_map = {
        "역할 분담이 명확한 경우": 0,          # 좋음
        "역할 분담이 모호한 경우": 1,          # 보통
        "역할 분담이 명확하지 않은 경우": 2,   # 나쁨
    }

    plan_adequacy_map = {
        "작업 계획이 적절하게 이루어진 경우": 0,
        "작업 계획이 불완전하게 이루어진 경우": 1,
        "작업 계획이 부적절하게 이루어진 경우": 2,
    }

    procedure_rule_map = {
        "작업절차 및 규정이 좋은 경우": 0,
        "작업절차 및 규정이 보통인 경우": 1,
        "작업절차 및 규정이 좋지 않은 경우": 2,
    }

    collaboration_map = {
        "작업자간 협업이 원활한 경우": 0,
        "작업자간 협업이 제한적인 경우": 1,
        "작업자간 협업이 원활하지 않은 경우": 2,
    }

    supervision_map = {
        "작업 관리 및 감독이 적절하게 이루어진 경우": 0,
        "작업 관리 및 감독이 불완전하게 이루어진 경우": 1,
        "작업 관리 및 감독이 부적절하게 이루어진 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp.get("questions", []):
            # PSF7(id=7)만 사용
            if q.get("id") != 7:
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp.get("round")
            row["answer_text"] = q.get("answer")

            conds = q.get("conditions", {})

            # 하위 PSF 항목들
            row["역할 분담 명료화"] = conds.get("역할 분담 명료화")
            row["작업 계획 적절성"] = conds.get("작업 계획 적절성")
            row["작업절차 및 규정"] = conds.get("작업절차 및 규정")
            row["작업자간 협업 정도"] = conds.get("작업자간 협업 정도")
            row["작업 관리 및 감독 적절성"] = conds.get("작업 관리 및 감독 적절성")

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF7] 원본 행 개수:", len(df), "===")

    # -----------------------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------------------
    # (1) 라벨 필터링 및 숫자화
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    # (2) 하위 PSF 조건 숫자화
    df["역할 분담 명료화"] = df["역할 분담 명료화"].map(role_division_map)
    df["작업 계획 적절성"] = df["작업 계획 적절성"].map(plan_adequacy_map)
    df["작업절차 및 규정"] = df["작업절차 및 규정"].map(procedure_rule_map)
    df["작업자간 협업 정도"] = df["작업자간 협업 정도"].map(collaboration_map)
    df["작업 관리 및 감독 적절성"] = df["작업 관리 및 감독 적절성"].map(supervision_map)

    print("\n[PSF7] 인코딩 후 head")
    print(df.head())

    # 혹시 매핑 안 된 값이 있으면 NaN 이 되므로 제거
    df = df.dropna().copy()

    feature_cols = [
        "역할 분담 명료화",
        "작업 계획 적절성",
        "작업절차 및 규정",
        "작업자간 협업 정도",
        "작업 관리 및 감독 적절성",
    ]

    X = df[feature_cols]
    y = df["label"]

    print("\n[PSF7] 사용할 샘플 수:", len(df))
    print("[PSF7] 라벨 분포:\n", y.value_counts())

    # -----------------------------
    # 5. train / test 분리
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # 각 클래스 비율 유지
    )

    # -----------------------------
    # 6. RandomForest 모델 학습 (LGBM → RF)
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=300,   # 트리 개수
        max_depth=None,    # None이면 자동으로 충분히 분할
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,         # CPU 병렬 처리
        class_weight=None, # 필요하면 'balanced'로 변경 가능
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # 7. 성능 평가 (정확도 + 리포트)
    # -----------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== [PSF7][RandomForest] 정확도(Accuracy) ===")
    print(f"{acc * 100:.2f}%")

    print("\n=== [PSF7][RandomForest] 분류 리포트 (class 0=좋음, 1=보통, 2=나쁨) ===")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 8. 특성 중요도 (어떤 하위 PSF가 영향력 큰지)
    # -----------------------------
    importances = model.feature_importances_
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print("\n=== [PSF7][RandomForest] Feature Importances (높을수록 영향력 큼) ===")
    for name, score in fi:
        print(f"{name:>20s} : {score:.4f}")

    # -----------------------------
    # 9. 예시: 새 설문 응답 예측
    # -----------------------------
    example_conds = {
        "역할 분담 명료화": "역할 분담이 명확하지 않은 경우",
        "작업 계획 적절성": "작업 계획이 부적절하게 이루어진 경우",
        "작업절차 및 규정": "작업절차 및 규정이 보통인 경우",
        "작업자간 협업 정도": "작업자간 협업이 제한적인 경우",
        "작업 관리 및 감독 적절성": "작업 관리 및 감독이 불완전하게 이루어진 경우",
    }

    x_new = [
        role_division_map[example_conds["역할 분담 명료화"]],
        plan_adequacy_map[example_conds["작업 계획 적절성"]],
        procedure_rule_map[example_conds["작업절차 및 규정"]],
        collaboration_map[example_conds["작업자간 협업 정도"]],
        supervision_map[example_conds["작업 관리 및 감독 적절성"]],
    ]

    x_new = np.array(x_new).reshape(1, -1)

    pred_012 = int(model.predict(x_new)[0])    # 0,1,2
    pred_123 = pred_012 + 1                   # 1,2,3 으로 변경

    label123_to_text = {
        1: "좋음",
        2: "보통",
        3: "나쁨",
    }

    print("\n=== [PSF7][RandomForest] 예시 설문 응답에 대한 예측 결과 ===")
    print("예측 결과 (클래스 번호 1/2/3):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_psf7()
