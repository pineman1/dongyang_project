# 고객 더미데이터 상품·특약 모델

사용자가 제공한 `customer_data.csv` 1,000명을 학습합니다. 실제 고객 계약이나 공개 보험료 표는 이 모델의 학습에 사용하지 않습니다. CSV에는 보험료가 없어 **가입상품과 가입특약을 분류**합니다.

## 기존 venv로 실행

저장소 폴더에서 실행합니다. Python 가상환경 자체에 소스나 데이터를 설치하지 않습니다.

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements-premium.txt
..\venv\Scripts\python.exe -m streamlit run premium_app.py
```

`premium_app.py`는 기존 실행 명령 호환용입니다. `customer_app.py`도 같은 화면을 실행합니다. API 키 없이 동작합니다.

- **상품·특약 추천:** 나이·성별·연소득·직업위험등급·결혼·자녀·흡연·만성질환·가족력을 입력합니다. 4개 상품의 모델 점수와 해당 상품 내 특약 점수를 표시합니다.
- **고객 분류·정렬:** 연령대, 흡연 여부, 결혼 여부, 자녀구간, 상품, 특약을 조합해 조회하고 필터된 CSV를 내려받습니다. 조회 필터는 학습 표본을 바꾸지 않습니다.
- **모델 검증:** 학습과 검증을 분리한 정확도, 최빈 상품 기준선, 중복 처리 방식을 표시합니다.
- 사이드바에서 기본 CSV 또는 제공된 고객 CSV·정렬 ZIP을 선택합니다. 업로드한 파일은 현재 앱의 학습에만 사용하며 저장소 원본을 덮어쓰지 않습니다.

### 학습 자료가 없는 조건

입력한 **9개 고객 조건이 모두 일치하는 학습 행**이 없으면 `자료가 없음`을 표시합니다. 각 값이 따로 존재하더라도 그 조합이 없으면 추천하지 않습니다. 허용 범위 밖 값이나 누락된 예측 입력도 같은 안내를 반환하며, 상품·특약 점수를 만들지 않습니다.

특약은 **선택한 상품 안에서 같은 9개 조건으로 학습한 행**이 있어야 표시합니다. 해당 자료가 없으면 특약 결과·점수 대신 `자료가 없음`을 표시합니다. `선택 안함`·`특약 없음`은 학습된 정답이며, 자료 부족 안내와 구분합니다. 고객 조회 필터 결과가 0명일 때도 같은 안내를 표시합니다.

`recommend()`와 `predict_batch()`는 `available`, `rider_available`로 결과 유무를 알립니다. 자료가 없는 결과와 점수는 `None`, 순위는 빈 목록입니다. 기존 세일즈 앱에서도 자료가 없는 경우 이전 추천·설계 상태를 지우고 후속 LLM 설명과 설계 생성을 중단합니다. 선호상품 선택으로 이 조건을 우회할 수 없습니다.

## CSV와 정렬 ZIP 처리

원본 11개 항목 중 입력 9개와 정답 2개만 사용합니다.

| 구분 | 항목 |
| --- | --- |
| 입력 | 나이, 성별, 연소득_만원, 직업위험등급, 결혼여부, 자녀수, 흡연여부, 만성질환, 가족력 |
| 정답 | 가입상품, 가입특약 |
| 입력에서 제외 | 분석ID, 상품분류, 특약상태, 특약가입여부 등 추가 열 |

추가된 분류 열을 모델에 그대로 넣으면 정답을 미리 알려주는 문제가 생길 수 있으므로 제외합니다. 원본 나이·자녀 수는 숫자로 학습하고 연령대·자녀구간은 조회용으로 계산합니다.

`동양생명_항목별정렬_CSV.zip` 안의 정렬 CSV 6개는 동일한 1,000행인지 검증한 뒤 하나만 사용합니다. 요약 CSV와 설명 파일은 학습하지 않습니다. 데이터가 서로 다르거나 정렬 파일이 빠진 ZIP은 오류로 알려줍니다. 압축 해제 파일을 디스크에 쓰지 않습니다.

원본에서 11개 값이 모두 같은 중복 15행은 보존합니다. 동일한 입력 9개를 가진 행은 정답이 다르더라도 같은 그룹으로 묶어 학습·검증 간 중복을 막습니다. CSV 순서가 바뀌어도 같은 학습 결과가 나오도록 정렬을 표준화합니다.

## 학습·평가

1단계 RandomForest가 가입상품을 예측합니다. 2단계는 선택된 상품으로 학습한 별도 RandomForest가 특약을 예측합니다. 따라서 다른 상품의 특약이 섞이지 않습니다. `선택 안함`과 `특약 없음`을 별도 정답으로 보존합니다.

- 전처리: 숫자 4개 + 범주 5개 원핫 인코딩. 학습 데이터에서만 인코더를 학습합니다.
- 모델별 설정: 트리 160개, 최대 깊이 9, 리프 최소 4행, random_state=42.
- 검증: 입력 조건 그룹의 20%를 GroupShuffleSplit으로 분리합니다. 검증 점수로 설정을 탐색하지 않았습니다.
- 최종 화면 모델: 평가 후 전체 1,000행으로 재학습합니다.

2026-09-19 기존 venv에서 확인한 **자료 유무 제한 적용 전 분류기 진단 결과**:

| 항목 | 값 |
| --- | ---: |
| 학습 / 검증 고객수 | 796 / 204 |
| 학습 / 검증 조건 그룹 | 776 / 194 |
| 양쪽에 겹친 조건 그룹 | 0 |
| 상품 예측 정확도 | 53.4% |
| 상품 macro F1 | 0.5012 |
| 상품·특약 동시 정확도 | 40.7% |
| 정답 상품이 주어졌을 때 특약 정확도 | 77.9% |
| 최빈 상품 기준선 정확도 | 38.7% |

특약 77.9%는 상품을 이미 안다는 조건의 평가입니다. 제한 전 2단계 분류기의 동시 정답률은 40.7%입니다. 학습·검증 조건 그룹이 겹치지 않으므로, 자료 일치 여부를 적용하면 이 검증 세트의 추천 표시율은 0%이며 모두 `자료가 없음`입니다. 위 정확도는 실제 표시 규칙의 성공률이 아닙니다. 분류기 진단만 내부 `_predict_for_evaluation()`을 사용하고 화면·일반 예측에는 자료 일치 규칙을 적용합니다.

생성 코드가 확률로 정답을 부여한 더미데이터의 결과이므로 실제 고객 선호나 보험 가입 적합성을 의미하지 않습니다. 화면 점수는 보정되지 않은 분류 모델 출력이며 실제 가입 확률이 아닙니다.

## 재학습과 테스트

```powershell
# 기본 customer_data.csv로 학습·평가·모델 저장
..\venv\Scripts\python.exe -m customer_ml.train

# 제공한 정렬 ZIP을 직접 학습 자료로 사용
..\venv\Scripts\python.exe -m customer_ml.train --data "reports/customer_segmentation/동양생명_항목별정렬_CSV.zip"

# 테스트
..\venv\Scripts\python.exe -m pytest tests -q
```

모델, 지표, 검증 행별 예측을 `artifacts/customer_ml/`에 저장합니다. 생성 모델 파일은 Git에 올리지 않습니다. `reports/customer_ml_metrics.json`과 `reports/customer_ml_holdout_predictions.csv`에는 확인한 평가 결과를 보관합니다. 화면은 데이터 내용에 따라 캐시를 갱신하고 직접 학습하므로 joblib 파일이 없어도 실행됩니다.

```python
from customer_ml.data import load_dataset
from customer_ml.model import CustomerPredictor

dataset = load_dataset("customer_data.csv")
predictor = CustomerPredictor(dataset.data)
result = predictor.recommend({
    "나이": 40, "성별": "남성", "연소득_만원": 5000, "직업위험등급": 1,
    "결혼여부": "기혼", "자녀수": 1, "흡연여부": "비흡연",
    "만성질환": "없음", "가족력": "없음",
})
if not result["available"]:
    print(result["message"])  # 자료가 없음
else:
    print(result["product"], result["rider"] if result["rider_available"] else "자료가 없음")
```

기존 `app.py`가 호출하는 `ml_model.get_recommendation()`은 이전 반환 키를 유지하며 같은 새 모델을 사용합니다. 기존 세일즈 앱의 API 호출·가상 보험료 계산식은 별도 기능입니다. 이전 공개 보험료 회귀 모델은 `premium/`와 `premium_reference_app.py`에 보관되어 있으며, 현재 기본 화면에서 사용하지 않습니다.

입력 범위는 제공된 예시 데이터에 맞춘 나이 20~65세, 연소득 3,000~12,000만원, 직업위험등급 1~3, 자녀 0~3명입니다. 학습 파일의 결측·지원하지 않는 범주·잘못된 상품/특약 조합은 오류로 표시합니다. 예측 입력에 일치하는 학습 자료가 없으면 `자료가 없음`을 표시합니다.

평가 방식 참고: [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
