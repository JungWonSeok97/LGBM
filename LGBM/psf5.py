# psf5.py  (PSF5: 의사소통)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np


def run_psf5():
    # -----------------------------
    # 1. 라벨 매핑 (answer -> 0/1/2)
    #    0: 좋음/약간 높음, 1: 보통, 2: 나쁨/매우나쁨/매우 높음
    # -----------------------------
    answer_to_label012 = {
        "좋음": 0,
        "약간 높음": 0,

        "보통": 1,

        "나쁨": 2,
        "매우나쁨": 2,
        "매우 높음": 2,
    }

    # -----------------------------
    # 2. 조건값 매핑 (문자열 -> 0/1/2)
    #    0: 좋은 상태, 1: 중간, 2: 나쁜 상태
    # -----------------------------

    # 의사소통 메시지의 길이
    msg_len_map = {
        "의사소통 메시지 길이가 짧을(단문) 경우": 0,
        "의사소통 메시지 길이가 보통일(중문) 경우": 1,
        "의사소통 메시지 길이가 길(복문) 경우 경우": 2,  # 설문 텍스트 그대로 사용
    }

    # 팀/조직 의사소통 자율성
    comm_autonomy_map = {
        "팀/조직 의사소통 자율성이 높을 경우": 0,
        "팀/조직 의사소통 자율성이 보통일 경우": 1,
        "팀/조직 의사소통 자율성이 낮을 경우": 2,
    }

    # 의사소통 프로토콜의 구체성
    protocol_specificity_map = {
        "의사소통 프로토콜이 매우 구체적일 경우": 0,
        "의사소통 프로토콜이 일부만 구체적일 경우": 1,
        "의사소통 프로토콜이 구체적이지 않을 경우": 2,
    }

    # 의사소통자 간 상황인식 공유 정도
    situation_share_map = {
        "의사소통자 간의 상황인식 공유가 잘 되어 있을 경우": 0,
        "의사소통자 간의 상황인식 공유가 부분적으로 되어 있을 경우": 1,
        "의사소통자 간의 상황인식 공유가 되지 않을 경우": 2,
    }

    # 의사소통을 위한 장비/기기의 적절성
    device_adequacy_map = {
        "의사소통을 하기 위한 장비/기기가 적절한 상태일 경우": 0,
        "의사소통을 하기 위한 장비/기기가 보통 수준인 상태일 경우": 1,
        "의사소통을 하기 위한 장비/기기가 부적절한 상태일 경우": 2,
    }

    # 의사소통을 위한 전문용어 숙지 필요성
    jargon_need_map = {
        "의사소통을 하기 위해 전문용어를 숙지할 필요가 없는 경우": 0,
        "의사소통을 하기 위해 전문용어를 부분적으로 숙지할 필요가 있는 경우": 1,
        "의사소통을 하기 위해 전문용어를 항상 숙지할 필요가 있는 경우": 2,
    }

    # -----------------------------
    # 3. JSON 파일 읽기
    # -----------------------------
    with open("설문조사결과.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for resp in data:
        for q in resp["questions"]:
            if q["id"] != 5:  # PSF5 (의사소통)만 사용
                continue

            row = {}
            row["psf_id"] = q["id"]
            row["round"] = resp.get("round")
            row["answer_text"] = q["answer"]

            conds = q["conditions"]

            row["의사소통 메시지의 길이"] = conds["의사소통 메시지의 길이"]
            row["팀/조직 의사소통 자율성"] = conds["팀/조직 의사소통 자율성"]
            row["의사소통 프로토콜의 구체성"] = conds["의사소통 프로토콜의 구체성"]
            row["의사소통자 간 상황인식 공유 정도"] = conds["의사소통자 간 상황인식 공유 정도"]
            row["의사소통을 위한 장비/기기의 적절성"] = conds["의사소통을 위한 장비/기기의 적절성"]
            row["의사소통을 위한 전문용어 숙지 필요성"] = conds["의사소통을 위한 전문용어 숙지 필요성"]

            rows.append(row)

    df = pd.DataFrame(rows)
    print("=== [PSF5] 원본 행 개수:", len(df), "===")

    # -----------------------------
    # 4. 라벨/조건 문자열 -> 숫자로 인코딩
    # -----------------------------
    # (1) 라벨
    df = df[df["answer_text"].isin(answer_to_label012.keys())].copy()
    df["label"] = df["answer_text"].map(answer_to_label012)

    # (2) 하위 PSF 조건 인코딩
    df["의사소통 메시지의 길이"] = df["의사소통 메시지의 길이"].map(msg_len_map)
    df["팀/조직 의사소통 자율성"] = df["팀/조직 의사소통 자율성"].map(comm_autonomy_map)
    df["의사소통 프로토콜의 구체성"] = df["의사소통 프로토콜의 구체성"].map(protocol_specificity_map)
    df["의사소통자 간 상황인식 공유 정도"] = df["의사소통자 간 상황인식 공유 정도"].map(situation_share_map)
    df["의사소통을 위한 장비/기기의 적절성"] = df["의사소통을 위한 장비/기기의 적절성"].map(device_adequacy_map)
    df["의사소통을 위한 전문용어 숙지 필요성"] = df["의사소통을 위한 전문용어 숙지 필요성"].map(jargon_need_map)

    print("\n[PSF5] 인코딩 후 데이터 일부")
    print(df.head())

    df = df.dropna().copy()

    feature_cols = [
        "의사소통 메시지의 길이",
        "팀/조직 의사소통 자율성",
        "의사소통 프로토콜의 구체성",
        "의사소통자 간 상황인식 공유 정도",
        "의사소통을 위한 장비/기기의 적절성",
        "의사소통을 위한 전문용어 숙지 필요성",
    ]

    X = df[feature_cols]
    y = df["label"]

    # -----------------------------
    # 5. train/test 분리
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 6. 모델 학습
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
    acc = accuracy_score(y_test, y_pred) * 100

    print("\n=== [PSF5] 의사소통 분류 성능 ===")
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 8. Feature Importance
    # -----------------------------
    importances = model.feature_importances_
    print("\n=== [PSF5] Feature Importance ===")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp}")

    # -----------------------------
    # 9. 예시 예측
    # -----------------------------
    example_conds = {
        "의사소통 메시지의 길이": "의사소통 메시지 길이가 보통일(중문) 경우",
        "팀/조직 의사소통 자율성": "팀/조직 의사소통 자율성이 보통일 경우",
        "의사소통 프로토콜의 구체성": "의사소통 프로토콜이 일부만 구체적일 경우",
        "의사소통자 간 상황인식 공유 정도": "의사소통자 간의 상황인식 공유가 부분적으로 되어 있을 경우",
        "의사소통을 위한 장비/기기의 적절성": "의사소통을 하기 위한 장비/기기가 보통 수준인 상태일 경우",
        "의사소통을 위한 전문용어 숙지 필요성": "의사소통을 하기 위해 전문용어를 부분적으로 숙지할 필요가 있는 경우",
    }

    x_new = [
        msg_len_map[example_conds["의사소통 메시지의 길이"]],
        comm_autonomy_map[example_conds["팀/조직 의사소통 자율성"]],
        protocol_specificity_map[example_conds["의사소통 프로토콜의 구체성"]],
        situation_share_map[example_conds["의사소통자 간 상황인식 공유 정도"]],
        device_adequacy_map[example_conds["의사소통을 위한 장비/기기의 적절성"]],
        jargon_need_map[example_conds["의사소통을 위한 전문용어 숙지 필요성"]],
    ]
    x_new = np.array(x_new).reshape(1, -1)

    pred_label = int(model.predict(x_new)[0])
    label012_to_text = {0: "좋음/약간 높음", 1: "보통", 2: "나쁨 이상"}

    print("\n=== [PSF5] 새 설문 응답 예측 ===")
    print("입력 조건:", example_conds)
    print("예측 결과 (숫자 라벨):", pred_label)
    print("예측 결과 (텍스트):", label012_to_text[pred_label])


if __name__ == "__main__":
    run_psf5()
