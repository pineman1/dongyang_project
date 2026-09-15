import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# 파일 경로 설정 (app.py에서 부를 때 에러 방지)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "customer_data.csv")

# ==========================================
# 🧠 모델 초기화 및 학습 (파일이 불려올 때 1번만 실행됨)
# ==========================================
df = pd.read_csv(CSV_PATH)
encoders = {}
categorical_columns = ["성별", "결혼여부", "흡연여부", "만성질환", "가족력", "가입상품", "가입특약"]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# 1단계: 주계약 학습
X_main = df.drop(["가입상품", "가입특약"], axis=1)
y_main = df["가입상품"]
rf_main = RandomForestClassifier(n_estimators=100, random_state=42)
rf_main.fit(X_main, y_main)

# 2단계: 상품별 특약 학습
rider_models = {}
for product_encoded in df["가입상품"].unique():
    df_sub = df[df["가입상품"] == product_encoded]
    X_sub = df_sub.drop(["가입상품", "가입특약"], axis=1)
    y_sub = df_sub["가입특약"]
    rf_rider = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_rider.fit(X_sub, y_sub)
    rider_models[product_encoded] = rf_rider

# ==========================================
# 🎯 외부(app.py)에서 호출할 예측 함수
# ==========================================
def get_recommendation(customer_info):
    """
    고객 딕셔너리를 받아 (추천상품, 상품확률, 추천특약, 특약확률)을 리턴합니다.
    """
    test_df = pd.DataFrame([customer_info])
    ordered_columns = ["나이", "성별", "연소득_만원", "직업위험등급", "결혼여부", "자녀수", "흡연여부", "만성질환", "가족력"]
    test_df = test_df[ordered_columns]
    
    # 만약 AI가 직업을 '사무직' 같은 글자로 잘못 뽑았을 경우를 대비해 숫자로 강제 변환
    if test_df["직업위험등급"].dtype == object:
        test_df["직업위험등급"] = pd.to_numeric(test_df["직업위험등급"], errors='coerce').fillna(1).astype(int)
    
    for col in ["성별", "결혼여부", "흡연여부", "만성질환", "가족력"]:
        # 만약 입력값이 없거나 이상하면 기본값(최빈값 등) 처리도 가능하지만, 
        # 프로토타입이므로 정확하게 들어온다고 가정합니다.
        test_df[col] = encoders[col].transform(test_df[col])
        
    main_probs = rf_main.predict_proba(test_df)[0]
    best_main_encoded = rf_main.classes_[np.argmax(main_probs)]
    best_main_name = encoders["가입상품"].inverse_transform([best_main_encoded])[0]
    best_main_prob = np.max(main_probs)
    
    rf_rider = rider_models[best_main_encoded]
    rider_probs = rf_rider.predict_proba(test_df)[0]
    best_rider_encoded = rf_rider.classes_[np.argmax(rider_probs)]
    best_rider_name = encoders["가입특약"].inverse_transform([best_rider_encoded])[0]
    best_rider_prob = np.max(rider_probs)
    
    return {
        "주계약": best_main_name,
        "주계약_확률": round(best_main_prob * 100, 1),
        "추천특약": best_rider_name,
        "특약_확률": round(best_rider_prob * 100, 1)
    }