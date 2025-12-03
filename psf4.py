# psf4_edu.py  (PSF4: 교육/훈련)

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier


def run_psf4():
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
    # -----------------------------
    edu_period_map = {
        "교육/훈련 주기(빈도)가 1년 미만일 경우": 0,
        "교육/훈련 주기(빈도)가 1~3년 사이일 경우": 1,
        "교육/훈련 주기(빈도)가 3년 이상일 경우": 2,
    }

    eval_freq_map = {
        "직무능력 평가 빈도가 6개월일 경우": 0,
        "직무능력 평가 빈도가 1년일 경우": 1,
        "직무능력 평가 빈도가 3년일 경우": 2,
    }

    proc_skill_map = {
        "절차적 지식의 숙련도가 높을 경우": 0,
        "절차적 지식의 숙련도가 보통인 경우": 1,
        "절차적 지식의 숙련도가 낮은 경우": 2,
    }

    equip_skill_map = {
        "장비 사용 지식의 숙련도가 높을 경우": 0,
        "장비 사용 지식의 숙련도가 보통인 경우": 1,
        "장비 사용 지식의 숙련도가 낮은 경우": 2,
    }

    program_map = {
        "교육/훈련 프로그램이 적절할 경우": 0,
        "교육/훈련 프로그램에 개선할 필요가 있을 경우": 1,
        "교육/훈련 프로그램이 적절하지 않을 경우": 2,
    }

    system_skill_map = {
        "시스템 지식(원리적 지식)의 숙련도가 높은 경우": 0,
        "시스템 지식(원리적 지식)의 숙련도가 보통인 경우": 1,
        "시스템 지식(원리적 지식)의 숙련도가 낮은 경우": 2,
    }

    experience_map = {
        "해야 하는 일의 90% 이상을 경험한 경우": 0,
        "해야 하는 일의 70~90%를 경험한 경우": 1,
        "해야 하는 일의 30% 이상을 경험하지 못한 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 4:   # 교육/훈련 PSF(id=4)만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp["round"]
            row["answer_text"] = q["answer"]

            conds = q["conditions"]

            # 하위 PSF 항목들
            row["교육 주기(빈도)"] = conds["교육 주기(빈도)"]
            row["직무능력 평가 빈도"] = conds["직무능력 평가 빈도"]
            row["절차적 지식의 숙련도"] = conds["절차적 지식의 숙련도"]
            row["장비 사용 지식의 숙련도"] = conds["장비 사용 지식의 숙련도"]
            row["교육/훈련 프로그램 적절성"] = conds["교육/훈련 프로그램 적절성"]
            row["시스템 지식(원리적 지식)의 숙련도"] = conds["시스템 지식(원리적 지식)의 숙련도"]
            row["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"] = conds["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF4] 원본 행 개수:", len(df), "===")

    # -----------------------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------------------
    # (1) 라벨
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    # (2) 하위 PSF 조건 숫자화
    df["교육 주기(빈도)"] = df["교육 주기(빈도)"].map(edu_period_map)
    df["직무능력 평가 빈도"] = df["직무능력 평가 빈도"].map(eval_freq_map)
    df["절차적 지식의 숙련도"] = df["절차적 지식의 숙련도"].map(proc_skill_map)
    df["장비 사용 지식의 숙련도"] = df["장비 사용 지식의 숙련도"].map(equip_skill_map)
    df["교육/훈련 프로그램 적절성"] = df["교육/훈련 프로그램 적절성"].map(program_map)
    df["시스템 지식(원리적 지식)의 숙련도"] = df["시스템 지식(원리적 지식)의 숙련도"].map(system_skill_map)
    df["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"] = df["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"].map(experience_map)

    print(df.head())

    # NaN이 있으면 제거 (혹시 매핑 안 된 값이 있을 경우)
    df = df.dropna().copy()

    feature_cols = [
        "교육 주기(빈도)",
        "직무능력 평가 빈도",
        "절차적 지식의 숙련도",
        "장비 사용 지식의 숙련도",
        "교육/훈련 프로그램 적절성",
        "시스템 지식(원리적 지식)의 숙련도",
        "상황에 대한 경험 정도(자주 또는 드물게 겪는지)",
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
    # 6-1. Feature Importance 확인
    # -----------------------------
    importances = model.feature_importances_

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n=== [PSF4] 하위 PSF 중요도(Feature Importance) ===")
    print(fi)

    # -----------------------------
    # 7. 성능 평가
    # -----------------------------
    y_pred = model.predict(X_test)

    # 클래스별 정밀도/재현율/F1
    print("\n=== [PSF4] 교육/훈련 분류 성능 ===")
    print(classification_report(y_test, y_pred))

    # 전체 정확도
    acc = accuracy_score(y_test, y_pred)
    print(f"정확도: {acc * 100:.2f}%")

    # -----------------------------
    # 8. 예시: 새 설문 응답 예측
    # -----------------------------
    example_conds = {
        "교육 주기(빈도)": "교육/훈련 주기(빈도)가 1년 미만일 경우",
        "직무능력 평가 빈도": "직무능력 평가 빈도가 6개월일 경우",
        "절차적 지식의 숙련도": "절차적 지식의 숙련도가 높을 경우",
        "장비 사용 지식의 숙련도": "장비 사용 지식의 숙련도가 높을 경우",
        "교육/훈련 프로그램 적절성": "교육/훈련 프로그램이 적절할 경우",
        "시스템 지식(원리적 지식)의 숙련도": "시스템 지식(원리적 지식)의 숙련도가 높은 경우",
        "상황에 대한 경험 정도(자주 또는 드물게 겪는지)": "해야 하는 일의 90% 이상을 경험한 경우",
    }

    x_new = [
        edu_period_map[example_conds["교육 주기(빈도)"]],
        eval_freq_map[example_conds["직무능력 평가 빈도"]],
        proc_skill_map[example_conds["절차적 지식의 숙련도"]],
        equip_skill_map[example_conds["장비 사용 지식의 숙련도"]],
        program_map[example_conds["교육/훈련 프로그램 적절성"]],
        system_skill_map[example_conds["시스템 지식(원리적 지식)의 숙련도"]],
        experience_map[example_conds["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"]],
    ]

    x_new = np.array(x_new).reshape(1, -1)

    pred_012 = int(model.predict(x_new)[0])   # 0,1,2
    pred_123 = pred_012 + 1                   # 1,2,3 으로 변경

    label123_to_text = {
        1: "좋음",
        2: "보통",
        3: "나쁨",
    }

    print("\n=== [PSF4] 새 설문 응답 예측 ===")
    print("예측 결과 (숫자 라벨 0/1/2):", pred_012)
    print("예측 결과 (1~3 변환):", pred_123)
    print("예측 결과 (텍스트):", label123_to_text[pred_123])


if __name__ == "__main__":
    run_psf4_edu()
