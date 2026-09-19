# 동양생명 공개 자료 기반 월 보험료 예측

> 이전 공개 자료 모델의 참고 문서입니다. 현재 기본 실행 화면(`premium_app.py`)은 고객 더미데이터 상품·특약 모델로 전환되었습니다. [현재 모델 설명](README_CUSTOMER_ML.md)을 참고하세요. 아래 회귀 모델을 별도로 실행하려면 `premium_reference_app.py`를 사용합니다.

나이·성별·보장 유형·납입기간·해약환급금 유형으로 월 보험료를 추정합니다. **2023년 9월 개정 「무배당 엔젤안심보험」 공개 예시표 72개**만 사용합니다. 실제 가입자 데이터나 현재 판매 견적을 학습한 모델이 아닙니다.

## 실행

Python 3.10 이상을 사용합니다. 개발 환경은 Windows / Python 3.12.8입니다.
소스는 `dongyang_project/`, 실행 환경은 상위 폴더의 기존 `venv/`로 분리했습니다. 가상환경을 삭제하거나 재생성해도 소스가 사라지지 않도록 한 구성입니다.

현재 컴퓨터에서는 저장소 폴더에서 다음 명령을 실행합니다. 가상환경 활성화나 PowerShell 실행 정책 변경은 필요 없습니다.

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements-premium.txt
..\venv\Scripts\python.exe -m streamlit run premium_reference_app.py
```

다른 컴퓨터에서 새로 시작하는 경우:

```powershell
git clone --branch namuking1/insurance-premium-model https://github.com/pineman1/dongyang_project.git
cd dongyang_project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-premium.txt
.\.venv\Scripts\python.exe -m streamlit run premium_reference_app.py
```

이하 `python`은 위 가상환경의 Python을 의미합니다. 기존 LLM 앱의 `requirements.txt` 대신 **`requirements-premium.txt`**를 사용하세요. 이 모델에는 OpenAI/Gemini API 키가 필요하지 않습니다. 기존 앱의 전체 의존성 호환성은 이번 검증 범위에 포함하지 않았습니다.

```powershell
# 학습 + 평가 + 모델 파일 저장
python -m premium.train

# 35세 남자 / 치매보장형 / 표준형 / 20년납 예측
python -m premium.predict --age 35 --sex male --coverage dementia --refund standard --payment-years 20

# 테스트
python -m pip install -r requirements-premium-dev.txt
python -m pytest tests -q

# 원본 PDF를 다시 내려받아 CSV와 출처 메타데이터 재생성
python scripts/collect_premium_data.py
```

첫 예측 예시는 약 **229,475원/월**입니다. 가입금액 1,000만원, 110세 만기, 2023.9 개정 상품의 보장 구조에 한정한 값입니다. 암보험 등 다른 상품의 보험료로 해석하면 안 됩니다.

`artifacts/premium/`에 `model.joblib`, `metrics.json`, `holdout_predictions.csv`가 생성됩니다. 모델 파일과 가상환경은 Git에 올리지 않으며, 누구나 CSV로 다시 학습할 수 있습니다. CLI와 화면은 작은 데이터로 즉시 재학습하므로 별도의 모델 파일 배포 없이 실행됩니다. `model.joblib`는 직접 생성한 파일만 사용하세요.

## 지원 조건

| 항목 | 지원 범위 |
| --- | --- |
| 상품·버전 | 무배당엔젤안심보험 / 2023.9 개정 |
| 보험나이 | 30~50세 정수; 원문 관측 나이는 30·40·50세 |
| 성별 | 원문 표의 남자·여자 |
| 치매보장형 납입기간 | 5·7·20년납 |
| 생활비보장형(장해) 납입기간 | 7·10·15년납 |
| 환급 유형 | 표준형·해약환급금일부지급형 |
| 고정 조건 | 보험가입금액 1,000만원·110세 만기·월납·주계약 |

다른 가입금액을 비례 계산하거나, 공개되지 않은 납입기간·범위 밖 나이로 예측하지 않습니다. 흡연·질병·직업·특약에 임의 할증을 더하지 않습니다. 관측값이 있는 조건에서는 **원문 조회값과 모델 추정값을 따로 표시**합니다. 보험가입금액은 각 보장 항목의 지급 보험금과 동일한 개념이 아닙니다.

## 모델과 검증

`scikit-learn`의 `TransformedTargetRegressor` + `LinearRegression`을 사용합니다.

```text
log(월 보험료) = 절편 + 나이 계수 × (나이 - 40) / 10 + 가입 조건별 계수
```

가입 조건은 성별 × 보장 유형 × 환급 유형 × 납입기간의 **24개 조합**이며, 원핫 인코딩합니다. 나이에 따른 비율 변화와 각 조건의 보험료 수준을 학습합니다. 모델 구조를 먼저 고정한 후 평가했으며, 평가 점수로 하이퍼파라미터를 탐색하지 않았습니다.

무작위 행 분할 대신 **30·50세 48개로 학습하고, 40세 24개를 모두 검증에 남겼습니다.** 전처리도 학습 데이터에서만 적합합니다. 최종 서비스 모델은 평가 후 72개 전체로 다시 학습합니다.

2026-09-17 실행 결과:

| 방법 | 40세 검증 MAE(원) | MAPE(%) |
| --- | ---: | ---: |
| 로그선형 회귀 | 169.17 | 0.0363 |
| 중앙값 기준선 | 170,600.00 | 39.4994 |
| RandomForest | 139,669.12 | 21.5057 |
| 조건별 선형 표 보간 | 24,860.42 | 4.3488 |
| 조건별 로그 표 보간 | 169.17 | 0.0363 |

**로그 표 보간도 같은 수준의 오차를 보입니다.** 이 작은 표에 복잡한 ML이 반드시 필요한 것은 아닙니다. 회귀 모델은 향후 데이터 확장을 위한 재현 가능한 출발점이며, 현재 결과를 ML의 우월성으로 해석하지 않습니다.

30세 또는 50세를 통째로 제외한 추가 진단도 저장합니다. 이 두 경우는 외삽 민감도 점검이며, API가 30~50세 바깥의 예측을 허용한다는 의미는 아닙니다. 성별별 오차와 평가 행별 실제값·예측값도 보고서에 포함합니다.

- [평가 설정·원본 행 ID·라이브러리 버전·지표](reports/premium_metrics.json)
- [검증용 24개 행의 실제값과 예측값](reports/premium_holdout_predictions.csv)

같은 상품의 규칙적인 표에서 얻은 낮은 오차는 **현재 판매 가격, 실제 고객, 다른 상품에 대한 정확도를 보장하지 않습니다.** 31~39·41~49세는 정답을 확보하지 못했으며 신뢰구간도 산출하지 않습니다. 72행은 독립적인 72명 고객이 아닙니다.

## Python에서 사용

```python
from premium.model import PremiumPredictor, PremiumRequest

predictor = PremiumPredictor()  # 72개 공개 예시 학습
result = predictor.predict(PremiumRequest(
    age=35,
    sex="남자",
    coverage_type="치매보장형",
    refund_type="표준형",
    payment_years=20,
))
print(result["monthly_premium_krw"])
print(result["published_example_krw"])  # 35세 원문 예시 없음 -> None
```

잘못된 성별·유형·납입 조합·가입금액·연령은 `ValueError`로 거절합니다. `premium/data.py`의 검증 규칙은 이번에 확인한 상품 스냅샷 전용입니다. 새 자료를 추가할 때는 출처·버전을 분리하고 검증 규칙과 입력 범위도 함께 확장해야 합니다.

## 기존 프로젝트와의 관계

- 기존 `ml_model.py`: `customer_ml`의 합성 고객 데이터 모델을 호출하는 호환 함수입니다.
- 기존 `app.py`: Gemini/RAG 및 가상 보험료 계산식을 사용하는 세일즈 데모입니다.
- 신규 `premium/`: 출처가 있는 공개 보험료를 학습하는 별도 회귀 모델입니다.
- `premium_reference_app.py`: 공개 보험료 참고 모델을 별도로 실행하는 화면입니다.
- `premium_app.py` 및 `pages/1_Customer_Recommendation.py`: 현재는 고객 더미데이터 상품·특약 모델을 실행합니다.

기존 상품 이름과 새 데이터의 정확한 상품·버전이 일치하지 않으므로, 가상 계산식을 새 모델로 조용히 대체하지 않았습니다. 실제 견적 기능으로 확장하려면 해당 상품의 최신 조건별 보험료와 가입금액별 자료부터 확보해야 합니다.
