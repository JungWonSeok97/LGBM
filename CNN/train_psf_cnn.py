# train_psf_all_cnn.py
# PSF 1~8 전체에 대해 PSF별 라벨/수준을 반영한 1D-CNN 학습/평가 스크립트

import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

# 재현성(랜덤 고정)
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------------
# 0. CNN 모델 생성 함수
# --------------------------------------------------------
def build_cnn_model(n_features: int, n_classes: int):
    """
    n_features: feature 개수 (시퀀스 길이)
    n_classes: 라벨 개수 (2 or 3)
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(n_features, 1)))     # (길이, 채널=1)

    # 1번째 Conv 블록
    model.add(layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 2번째 Conv 블록
    model.add(layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 출력부
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.5))

    if n_classes == 2:
        # 이진 분류 (PSF 5, 6)
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        # 다중 분류 (PSF 1,2,3,4,7,8)
        model.add(layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------
# 1. PSF별 라벨 매핑 (기존 코드 그대로)
# --------------------------------------------------------
label_map_by_psf = {
    1: {  # PSF1: 보통 / 나쁨 / 매우나쁨
        "보통": 0,
        "나쁨": 1,
        "매우나쁨": 2,
    },
    2: {  # PSF2: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    3: {  # PSF3: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    4: {  # PSF4: 교육/훈련 (좋음 / 보통 / 나쁨)
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    5: {  # PSF5: 2단계
        "보통": 0,
        "나쁨": 1,
    },
    6: {  # PSF6: 2단계
        "보통": 0,
        "나쁨": 1,
    },
    7: {  # PSF7: 좋음 / 보통 / 나쁨
        "좋음": 0,
        "보통": 1,
        "나쁨": 2,
    },
    8: {  # PSF8: 보통 / 약간 높음 / 매우 높음
        "보통": 0,
        "약간 높음": 1,
        "매우 높음": 2,
    },
}

# 3단계용 기본 텍스트 (PSF 1,2,3,4,7,8)
default_severity_text = {
    0: "좋음(수준 1)",
    1: "보통(수준 2)",
    2: "나쁨(수준 3)",
}

# PSF 5,6 전용 텍스트 (2단계)
severity_text_by_psf = {
    5: {
        0: "보통(수준 1)",
        1: "나쁨(수준 2)",
    },
    6: {
        0: "보통(수준 1)",
        1: "나쁨(수준 2)",
    },
}

# --------------------------------------------------------
# 2. JSON 파일 읽기 (기존과 동일)
# --------------------------------------------------------
with open("설문조사결과.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for resp in data:
    rnd = resp.get("round")
    for q in resp["questions"]:
        psf_id = q.get("id")
        answer_text = q.get("answer")

        # 1~8번 PSF 중 아닌 건 무시
        if psf_id not in label_map_by_psf:
            continue

        label_map = label_map_by_psf[psf_id]
        if answer_text not in label_map:
            continue

        row = {
            "psf_id": psf_id,
            "round": rnd,
            "answer_text": answer_text,
            "label": label_map[answer_text],
        }

        # 하위 PSF 조건들 추가
        conds = q.get("conditions", {})
        for k, v in conds.items():
            row[k] = v

        rows.append(row)

df = pd.DataFrame(rows)
print("전체 PSF 질문 수:", len(df))
print("PSF별 데이터 개수:")
print(df["psf_id"].value_counts().sort_index())

# --------------------------------------------------------
# 3. PSF별로 CNN 학습/평가
# --------------------------------------------------------
for psf_id in sorted(df["psf_id"].unique()):
    print("\n" + "=" * 70)
    print(f"=== PSF {psf_id} 학습/평가 (CNN) ===")

    df_psf = df[df["psf_id"] == psf_id].copy()
    print("샘플 수:", len(df_psf))
    print("라벨 분포:")
    print(df_psf["label"].value_counts().sort_index())

    # 라벨이 1종류뿐이면 학습 불가
    if df_psf["label"].nunique() < 2:
        print("⚠️ 라벨이 한 종류뿐이라 학습/평가를 건너뜁니다.")
        continue

    # 이 PSF에서 실제 값이 있는 feature 컬럼만 사용
    drop_cols = ["psf_id", "round", "answer_text", "label"]
    candidate_cols = [c for c in df_psf.columns if c not in drop_cols]
    feature_cols = [c for c in candidate_cols if df_psf[c].notna().any()]

    if not feature_cols:
        print("⚠️ 사용 가능한 feature(조건)가 없습니다. 건너뜁니다.")
        continue

    # 조건값 문자열 -> 카테고리 코드(0,1,2,...)로 변환 (기존과 동일)
    for col in feature_cols:
        df_psf[col] = df_psf[col].astype("category").cat.codes

    # -1 (결측) 포함된 행 제거
    mask_valid = (df_psf[feature_cols] != -1).all(axis=1)
    df_psf = df_psf[mask_valid].copy()

    if len(df_psf) < 5:
        print("⚠️ 유효한 샘플이 너무 적어(5개 미만) 학습/평가를 건너뜁니다.")
        continue

    # 입력/라벨 분리
    X = df_psf[feature_cols].values.astype(np.float32)
    y = df_psf["label"].values.astype(np.int64)

    # ----------------------------------------------------
    # 3-1. train / test 분리 (기존과 동일)
    # ----------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        # 데이터가 적어서 stratify 실패 시
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        print("⚠️ stratify 없이 train/test 분리 수행")

    # ----------------------------------------------------
    # 3-2. 스케일링 + CNN 입력 형태로 reshape
    # ----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # (samples, features) → (samples, features, 1)
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]

    n_features = X_train_cnn.shape[1]
    n_classes = len(np.unique(y))

    print(f"feature 개수: {n_features}, 클래스 개수: {n_classes}")

    # ----------------------------------------------------
    # 3-3. CNN 모델 정의 및 학습
    # ----------------------------------------------------
    model = build_cnn_model(n_features, n_classes)
    model.summary(print_fn=lambda x: None)  # 콘솔에 summary 쏟아지는 게 싫으면 이렇게

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train_cnn,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # ----------------------------------------------------
    # 3-4. 예측 및 성능 평가
    # ----------------------------------------------------
    if n_classes == 2:
        # 이진 분류 → sigmoid 출력
        y_pred_proba = model.predict(X_test_cnn).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        # 다중 분류 → softmax 출력
        y_pred_proba = model.predict(X_test_cnn)
        y_pred = np.argmax(y_pred_proba, axis=1)

    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"[정확도] {acc * 100:.2f}%")

    # ----------------------------------------------------
    # 3-5. 예시 예측 1건 (테스트셋 기준)
    # ----------------------------------------------------
    if len(X_test) > 0:
        sample_idx = 0
        true_label = int(y_test[sample_idx])
        pred_label = int(y_pred[sample_idx])

        sev_map = severity_text_by_psf.get(psf_id, default_severity_text)

        print("\n[예시 예측 1건]")
        print(f"실제 라벨: {true_label} ({sev_map.get(true_label, 'N/A')})")
        print(f"예측 라벨: {pred_label} ({sev_map.get(pred_label, 'N/A')})")
    else:
        print("\n[예시 예측] 테스트 샘플이 없어 생략")

    # 🔹 참고: LGBM에서 하던 Feature Importance는
    # CNN에는 기본 제공되지 않기 때문에 여기서는 생략했습니다.
    # 나중에 permutation importance 같은 걸로 따로 만들 수 있어요.
