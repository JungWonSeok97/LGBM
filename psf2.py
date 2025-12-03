# psf2.py  (PSF2: 장비 및 기기)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np

def run_psf2():
    # 1. 라벨 매핑
    answer_to_label012 = {
        "좋음": 0,
        "약간 높음": 0,
        "보통": 1,
        "나쁨": 2,
        "매우나쁨": 2,
        "매우 높음": 2,
    }

    # 2. 조건 매핑
    equip_use_map = {
        "업무에 필요한 장비의 사용성이 우수한 경우": 0,
        "업무에 필요한 장비의 사용성이 보통인 경우": 1,
        "업무에 필요한 장비의 사용성이 미흡한 경우": 2,
    }
    maint_map = {
        "사용하는 장비가 매우 적절하게 유지보수 되는 경우": 0,
        "사용하는 장비가 일반적으로 유지보수 되는 경우": 1,
        "사용하는 장비가 미흡하게 유지보수 되는 경우": 2,
    }
    tool_map = {
        "사용하는 작업도구의 적절성이 우수한 경우": 0,
        "사용하는 작업도구의 적절성이 보통한 경우": 1,
        "사용하는 작업도구의 적절성이 미흡한 경우": 2,
    }
    link_map = {
        "사용하는 장비가 타 기기와의 연계성이 우수한 경우": 0,
        "사용하는 장비가 타 기기와의 연계성이 보통인 경우": 1,
        "사용하는 장비가 타 기기와의 연계성이 미흡한 경우": 2,
    }
    avail_map = {
        "업무에 필요한 장비의 가용성이 높은 경우": 0,
        "업무에 필요한 장비의 가용성이 낮은 경우": 2,
    }
    manual_map = {
        "장비 사용 매뉴얼의 질이 우수한 경우": 0,
        "장비 사용 매뉴얼의 질이 보통인 경우": 1,
        "장비 사용 매뉴얼의 질이 미흡한 경우": 2,
    }
    itsys_map = {
        "업무에 사용되는 컴퓨터지원시스템(운행정보시스템, 운행관리시스템, 업무용 소프트웨어 등)이 우수한 경우": 0,
        "업무에 사용되는 컴퓨터지원시스템(운행정보시스템, 운행관리시스템, 업무용 소프트웨어 등)이 보통인 경우": 1,
        "업무에 사용되는 컴퓨터지원시스템(운행정보시스템, 운행관리시스템, 업무용 소프트웨어 등)이 미흡한 경우": 2,
    }

    # 3. JSON 읽기
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 2:  # PSF2만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]

            conds = q["conditions"]
            row["장비의 사용성"] = conds["장비의 사용성"]
            row["유지보수의 적절성"] = conds["유지보수의 적절성"]
            row["작업 도구의 적절성"] = conds["작업 도구의 적절성"]
            row["타 기기와의 연계성"] = conds["타 기기와의 연계성"]
            row["장비 가용성(접근성)"] = conds["장비 가용성(접근성)"]
            row["장비 사용 매뉴얼의 질"] = conds["장비 사용 매뉴얼의 질"]
            row["컴퓨터 지원 시스템의 적절성"] = conds["컴퓨터 지원 시스템의 적절성"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF2] 원본 행 개수:", len(df), "===")

    # 4. 인코딩
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    df["장비의 사용성"] = df["장비의 사용성"].map(equip_use_map)
    df["유지보수의 적절성"] = df["유지보수의 적절성"].map(maint_map)
    df["작업 도구의 적절성"] = df["작업 도구의 적절성"].map(tool_map)
    df["타 기기와의 연계성"] = df["타 기기와의 연계성"].map(link_map)
    df["장비 가용성(접근성)"] = df["장비 가용성(접근성)"].map(avail_map)
    df["장비 사용 매뉴얼의 질"] = df["장비 사용 매뉴얼의 질"].map(manual_map)
    df["컴퓨터 지원 시스템의 적절성"] = df["컴퓨터 지원 시스템의 적절성"].map(itsys_map)

    df = df.dropna().copy()

    feature_cols = [
        "장비의 사용성",
        "유지보수의 적절성",
        "작업 도구의 적절성",
        "타 기기와의 연계성",
        "장비 가용성(접근성)",
        "장비 사용 매뉴얼의 질",
        "컴퓨터 지원 시스템의 적절성",
    ]
    X = df[feature_cols]
    y = df["label"]

    # 5. train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 모델 학습
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 7. 성능 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print("\n=== [PSF2] 장비 및 기기 분류 성능 ===")
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred))

    # 8. 중요도
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    print("\n=== [PSF2] Feature Importance ===")
    for name, val in fi.items():
        print(f"{name}: {val}")

    # 9. 예시 예측
    example_conds = {
        "장비의 사용성": "업무에 필요한 장비의 사용성이 미흡한 경우",
        "유지보수의 적절성": "사용하는 장비가 일반적으로 유지보수 되는 경우",
        "작업 도구의 적절성": "사용하는 작업도구의 적절성이 보통한 경우",
        "타 기기와의 연계성": "사용하는 장비가 타 기기와의 연계성이 보통인 경우",
        "장비 가용성(접근성)": "업무에 필요한 장비의 가용성이 낮은 경우",
        "장비 사용 매뉴얼의 질": "장비 사용 매뉴얼의 질이 보통인 경우",
        "컴퓨터 지원 시스템의 적절성": "업무에 사용되는 컴퓨터지원시스템(운행정보시스템, 운행관리시스템, 업무용 소프트웨어 등)이 미흡한 경우",
    }

    x_new = [
        equip_use_map[example_conds["장비의 사용성"]],
        maint_map[example_conds["유지보수의 적절성"]],
        tool_map[example_conds["작업 도구의 적절성"]],
        link_map[example_conds["타 기기와의 연계성"]],
        avail_map[example_conds["장비 가용성(접근성)"]],
        manual_map[example_conds["장비 사용 매뉴얼의 질"]],
        itsys_map[example_conds["컴퓨터 지원 시스템의 적절성"]],
    ]
    x_new = np.array(x_new).reshape(1, -1)

    pred_012 = int(model.predict(x_new)[0])
    pred_123 = pred_012 + 1
    label123_to_text = {1: "좋음", 2: "보통", 3: "나쁨"}

    print("\n=== [PSF2] 새 설문 응답 예측 ===")
    print("예측 결과 (숫자):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_psf2()
