# psf1_cnn.py  (PSF1: 작업부하, CNN 버전)

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# ✅ 새로 추가된 부분 (딥러닝용)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 재현성 확보 (선택)
tf.random.set_seed(42)
np.random.seed(42)


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

    # ✅ CNN을 위해 numpy 배열로 변환 (float32)
    X = df[feature_cols].values.astype("float32")
    y = df["label"].values

    # -----------------------------
    # 5. train/test 분리
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ CNN 입력 형태로 reshape: (샘플 수, 시퀀스 길이=특징 개수, 채널=1)
    X_train_cnn = X_train.reshape(-1, len(feature_cols), 1)
    X_test_cnn = X_test.reshape(-1, len(feature_cols), 1)

    # ✅ 라벨을 원-핫 인코딩 (다중 클래스 분류용)
    num_classes = 3
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # -----------------------------
    # 6. CNN 모델 정의 & 학습
    # -----------------------------
    model = Sequential(
        [
            Conv1D(
                filters=32,
                kernel_size=2,
                activation="relu",
                input_shape=(len(feature_cols), 1),
            ),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=2, activation="relu"),
            BatchNormalization(),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train_cnn,
        y_train_cat,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1,
    )

    # -----------------------------
    # 7. 성능 평가
    # -----------------------------
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred) * 100

    print("\n=== [PSF1][CNN] 작업부하 분류 성능 ===")
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 8. (간이) 특성 중요도 - Permutation Importance
    # -----------------------------
    def compute_permutation_importance(model, X_test_cnn, y_test, feature_cols):
        base_pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
        base_acc = accuracy_score(y_test, base_pred)

        importances = []
        for i, col in enumerate(feature_cols):
            X_perm = X_test_cnn.copy()
            idx = np.random.permutation(X_perm.shape[0])
            # i번째 feature만 섞기
            X_perm[:, i, 0] = X_perm[idx, i, 0]

            perm_pred = np.argmax(model.predict(X_perm, verbose=0), axis=1)
            perm_acc = accuracy_score(y_test, perm_pred)
            importance = base_acc - perm_acc  # 정확도 감소량
            importances.append((col, importance))
        return importances

    importances = compute_permutation_importance(
        model, X_test_cnn, y_test, feature_cols
    )

    print("\n=== [PSF1][CNN] Feature Importance (Permutation) ===")
    for col, imp in sorted(importances, key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp:.4f} (accuracy drop)")

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
    x_new = np.array(x_new, dtype="float32").reshape(1, len(feature_cols), 1)

    pred_proba_new = model.predict(x_new)
    probs = pred_proba_new[0]  # shape: (3,)  → [p(class0), p(class1), p(class2)]
    pred_label = int(np.argmax(probs))

    label012_to_text = {0: "보통", 1: "나쁨", 2: "매우나쁨"}

    print("\n=== [PSF1][CNN] 새 설문 응답 예측 ===")
    print("입력 조건:", example_conds)
    print("각 클래스별 확률:")
    for cls in [0, 1, 2]:
        print(f"  {cls} ({label012_to_text[cls]}): {probs[cls]:.4f}")

    print("\n가장 높은 확률을 가진 클래스:")
    print("예측 결과 (숫자 라벨):", pred_label)
    print("예측 결과 (텍스트):", label012_to_text[pred_label])


if __name__ == "__main__":
    run_psf1()