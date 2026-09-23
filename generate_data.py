import numpy as np
import pandas as pd

# 재현성을 위한 시드 설정
np.random.seed(42)
NUM_SAMPLES = 5000

# 1. 기본 인적 데이터 생성
ages = np.random.randint(20, 66, size=NUM_SAMPLES)
genders = np.random.choice(["남성", "여성"], size=NUM_SAMPLES, p=[0.5, 0.5])
incomes = np.random.choice(
    [3000, 4000, 5000, 6000, 7500, 9000, 12000],
    size=NUM_SAMPLES,
    p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.04],
)
job_risk_levels = np.random.choice([1, 2, 3], size=NUM_SAMPLES, p=[0.7, 0.2, 0.1])

# 2. 가족 관계 (나이에 따른 자연스러운 분기)
marital_statuses = []
children_counts = []

for age in ages:
    if age < 30:
        is_married = np.random.choice(["미혼", "기혼"], p=[0.85, 0.15])
        child = 0 if is_married == "미혼" else np.random.choice([0, 1], p=[0.8, 0.2])
    elif age < 45:
        is_married = np.random.choice(["미혼", "기혼"], p=[0.3, 0.7])
        child = 0 if is_married == "미혼" else np.random.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.35, 0.05])
    else:
        is_married = np.random.choice(["미혼", "기혼"], p=[0.1, 0.9])
        child = 0 if is_married == "미혼" else np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])

    marital_statuses.append(is_married)
    children_counts.append(child)

# 3. 건강 데이터
smokings = np.random.choice(["비흡연", "흡연"], size=NUM_SAMPLES, p=[0.75, 0.25])
family_histories = np.random.choice(["없음", "암", "심혈관"], size=NUM_SAMPLES, p=[0.6, 0.25, 0.15])

chronic_diseases = []
for age in ages:
    if age < 40:
        disease = np.random.choice(["없음", "고혈압", "당뇨"], p=[0.90, 0.07, 0.03])
    elif age < 55:
        disease = np.random.choice(["없음", "고혈압", "당뇨"], p=[0.65, 0.25, 0.10])
    else:
        disease = np.random.choice(["없음", "고혈압", "당뇨"], p=[0.40, 0.40, 0.20])
    chronic_diseases.append(disease)

# 4. 정답(최종 가입 상품 및 특약) 라벨링 규칙 부여
target_products = []
target_riders = []  # 🌟 새로 추가된 특약 리스트

for i in range(NUM_SAMPLES):
    age = ages[i]
    gender = genders[i]
    child = children_counts[i]
    disease = chronic_diseases[i]
    family = family_histories[i]
    income = incomes[i]
    married = marital_statuses[i]

    # [1단계] 주계약(보험 상품) 결정 로직 ---------------------------
    scores = {
        "수호천사 암/건강보험": 10,
        "수호천사 간편심사(유병자)보험": 5,
        "수호천사 우리가족 종신보험": 5,
        "수호천사 행복 연금보험": 5,
    }

    if disease in ["고혈압", "당뇨"]:
        scores["수호천사 간편심사(유병자)보험"] += 35
        scores["수호천사 암/건강보험"] -= 10
    if family in ["암", "심혈관"]:
        scores["수호천사 암/건강보험"] += 25
    if smokings[i] == "흡연":
        scores["수호천사 암/건강보험"] += 10
    if married == "기혼" and child >= 1 and (30 <= age <= 50):
        scores["수호천사 우리가족 종신보험"] += 30
    if (married == "미혼" and income >= 5000) or (age >= 50 and child == 0):
        scores["수호천사 행복 연금보험"] += 25

    products = list(scores.keys())
    probs = np.array(list(scores.values()), dtype=float)
    probs = np.maximum(probs, 1)
    probs /= probs.sum()
    selected_product = np.random.choice(products, p=probs)
    
    # [2단계] 선택된 주계약 안에서 조건별 특약 결정 (질문자님 아이디어 적용) 🌟
    selected_rider = "선택 안함"

    if selected_product == "수호천사 암/건강보험":
        # 규칙 1: 20~30대 남성은 수술비 특약 선호 (80%)
        if gender == "남성" and age < 40:
            selected_rider = np.random.choice(["수술비 보장 특약", "표적항암약물허가치료 특약", "선택 안함"], p=[0.8, 0.1, 0.1])
        # 규칙 2: 50~60대 여성은 표적항암 특약 선호 (85%)
        elif gender == "여성" and age >= 50:
            selected_rider = np.random.choice(["표적항암약물허가치료 특약", "수술비 보장 특약", "선택 안함"], p=[0.85, 0.05, 0.10])
        # 규칙 3: 암 가족력이 있으면 무조건 표적항암 선호
        elif family in ["암", "심혈관"]:
            selected_rider = np.random.choice(["표적항암약물허가치료 특약", "수술비 보장 특약", "선택 안함"], p=[0.75, 0.15, 0.10])
        else:
            selected_rider = np.random.choice(["표적항암약물허가치료 특약", "수술비 보장 특약", "선택 안함"], p=[0.4, 0.4, 0.2])

    elif selected_product == "수호천사 간편심사(유병자)보험":
        # 유병자는 대부분 산정특례 특약을 꼭 챙김 (80%)
        selected_rider = np.random.choice(["중증질환 산정특례 보장 특약", "선택 안함"], p=[0.8, 0.2])

    elif selected_product == "수호천사 우리가족 종신보험":
        # 자녀가 있는 가장은 가족 수입 보장 특약을 압도적으로 선호 (90%)
        if child >= 1:
            selected_rider = np.random.choice(["가족 수입 보장 특약", "선택 안함"], p=[0.9, 0.1])
        else:
            selected_rider = np.random.choice(["가족 수입 보장 특약", "선택 안함"], p=[0.3, 0.7])

    elif selected_product == "수호천사 행복 연금보험":
        # 저축/연금보험은 일반적으로 보장성 특약을 잘 넣지 않음
        selected_rider = "특약 없음"

    target_products.append(selected_product)
    target_riders.append(selected_rider)

# 5. 데이터프레임 조립 및 CSV 저장 (특약 열 추가)
df = pd.DataFrame(
    {
        "나이": ages,
        "성별": genders,
        "연소득_만원": incomes,
        "직업위험등급": job_risk_levels,
        "결혼여부": marital_statuses,
        "자녀수": children_counts,
        "흡연여부": smokings,
        "만성질환": chronic_diseases,
        "가족력": family_histories,
        "가입상품": target_products,
        "가입특약": target_riders,  # 🌟 새로 추가된 열
    }
)

df.to_csv("customer_data.csv", index=False, encoding="utf-8-sig")
print(f"특약 정보가 포함된 총 {len(df)}건의 데이터가 'customer_data.csv'로 저장되었습니다.")
