접속 url : http://43.201.62.208:8501/

# 💼 동양생명 AI FC 어시스턴트 (Dongyang Life AI FC Assistant)

> **고객 데이터 기반 ML 상품·특약 추천 × Gemini RAG 기반 세일즈 스크립트 & 실시간 가입설계 지원 툴**

---

## 📌 1. 프로젝트 소개 (Overview)

본 프로젝트는 **생명보험 설계사(FC)**의 현장 영업 경쟁력과 업무 효율을 극대화하기 위해 개발된 **'실전형 AI 세일즈 지원 솔루션'**입니다.

고객 특성 데이터를 분석하는 **머신러닝(ML) 모델**과 동양생명 약관 DB를 활용하는 **언어모델(LLM RAG)**을 결합하여, 고객 맞춤형 상품 추천, 대화형 세일즈 스크립트 및 거절 극복 화법 제공, 실시간 가입설계 튜닝 및 PDF 발급을 원스톱으로 지원합니다.

* **브랜드 정체성**: 동양생명 시그니처 컬러(Cyan / Sky Blue) 기반의 맞춤 UI/UX 적용
* **핵심 가치**: 상담 준비 시간 단축, 고객 맞춤형 스크립트 자동화, 현장 즉시 견적 조율을 통한 계약 전환율 제고

---

## 👥 2. 팀원 및 역할 (Team & Roles)

| 이름 | 담당 역할 (Role) | 주요 기획 및 개발 내용 |
| :--- | :--- | :--- |
| **강현석** | **PO (Product Owner)** | 기획 총괄, 프롬프트 엔지니어링 설계, 시나리오 QA 및 전체 프로젝트 일정 관리 |
| **강아라** | **AI Engineer** | 개인정보 보안 및 로그인 인증 기능 설계 |
| **김경탁** | **Data Scientist** | 고객 데이터셋(CSV 1,000명) 전처리, 분석 및 ML 추천 모델 고도화 |
| **강동훈** | **AI Engineer** | 클라우드 배포 인프라 구축, 서버 환경 세팅, RAG 아키텍처 및 추천 로직 설계 |
| **손민근** | **AI Software Architect** | 프론트엔드(UI) 모듈화 통합, API 연동 및 Git/GitHub 버전 관리 |

---

## 🎯 3. 핵심 타겟 & 고객 페르소나 (Target Personas)

### 1. 이준호 (45세, 사무직) | *"방어형 고관여 직장인"*
* **Pain Point**: 대면 설계사의 불필요한 특약 끼워팔기 및 오프라인 영업에 대한 강한 불쾌감.
* **Service Value (AI 방패)**: 대면 협상 전, AI가 추출한 '객관적 원가(PDF)'를 무기로 주도권 확보.
* **Strategy & Copy**: *"내 보험료, 설계사 수당으로 얼마나 빠져나가는지 계산해 보셨나요? AI로 진짜 내 보험 원가를 확인하세요."*

### 2. 박미영 (58세, 식당 운영) | *"가입 거절 트라우마 유병자"*
* **Pain Point**: 고혈압/당뇨 등 병력으로 인한 오프라인 보험 심사 거절 불안감.
* **Service Value (심리적 안심)**: 대면 민망함 없이 모바일에서 즉시 가입 가능 여부 비대면 조회.
* **Strategy & Copy**: *"혈압약 5년째 드시고 계신다고요? 묻지도 따지지도 않고 내 폰에서 몰래 가입 승인 먼저 받아보세요."*

### 3. 최동만 (65세, 은퇴자) | *"전문가의 더블 체크가 필요한 시니어"*
* **Pain Point**: 복잡한 약관에 대한 이해 부담 및 신뢰할 수 있는 담당자 필요.
* **Service Value (AI + Human Touch)**: AI 1차 데이터 분석 + 동양생명 전문 설계사 2차 검증(O2O 연계).
* **Strategy & Copy**: *"첨단 AI가 1차로 찾고, 20년 차 동양생명 수석 설계사가 한 번 더 검증합니다."*

### 4. 정수진 (32세, 프리랜서) | *"스마트폰으로 끝내고 싶은 딩크족"*
* **Pain Point**: 설계사와의 대면/전화 상담(콜 포비아) 및 불필요한 스몰토크 거부감.
* **Service Value (완벽한 비대면 DIY)**: 설계사 주관을 배제한 데이터 추천 기반 가성비 상품 모바일 DIY 가입.
* **Strategy & Copy**: *"기 빨리는 설계사 스몰토크 극혐이시죠? 내 맘대로 안 쓰는 특약 싹 다 빼고 1분 만에 가입 끝내세요."*

---

## 🛠️ 4. 주요 기능 (Key Features)

* **의도 파악 및 고객 니즈 오버라이드**
  * 단순 질의응답을 넘어 고객의 입력 의도(추천, 설계, 질의)를 정확히 파악합니다.
  * 통계 모델 추천보다 고객이 명시한 '선호 상품'을 최우선 반영하는 유연한 추천 알고리즘을 적용했습니다.

* **💬 AI 상담 챗봇 & 거절 극복 방어 화법 (RAG & Objection Handling)**
  * 동양생명 약관 DB(Vector DB) 기반 질의응답 및 고객 맞춤 세일즈 대본을 자동 생성합니다.
  * 현장 거절 상황 발생 시 **'20년 차 지점장 페르소나'**가 작동하여 논리적인 반박 화법 및 대안을 제시합니다.

* **📊 실시간 가입설계 튜닝 패널 (Tuning Panel)**
  * 납입 기간(10년/20년/30년납), 특약 여부 등을 실시간 조율하여 월 예상 보험료를 즉시 계산합니다.
  * 확정된 조건으로 현장에서 바로 고객 전달용 표준 가입설계서 PDF를 자동 생성 및 다운로드합니다.

* **고객 더미데이터 기반 ML 추천 (1,000명 학습 모델)**
  * `customer_data.csv` (1,000명 표본) 데이터셋을 학습하여 고객 조건(나이, 질환, 직업 등 9가지 조건)에 최적화된 주계약 및 특약을 도출합니다.

---

## 💻 5. 기술 스택 및 디렉토리 구조 (Tech Stack)

### Tech Stack

| 구분 | 기술 스택 |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit (Custom Theme, Modular UI) |
| **AI / ML / RAG**| Google Gemini-3.5-Flash, LangChain, FAISS, Scikit-learn (RandomForest) |
| **Data & Doc** | Pandas, PyMuPDF, FPDF2 |

### Directory Structure

```plaintext
dongyang_project/
├── app.py                     # 메인 비즈니스 로직 및 세션 관리
├── ui.py                      # Streamlit UI/UX 모듈화 (화면 렌더링, 테마, 탭 설정)
├── premium_app.py             # 1,000명 고객 모델 실행 전용 앱 (customer_app.py)
├── ml_model.py                # ML 기반 고객 맞춤 상품/특약 추천 엔진
├── .streamlit/
│   └── config.toml            # 동양생명 브랜드 시그니처 테마 설정
├── data/                      # 고객 데이터셋 (customer_data.csv 등)
└── requirements.txt           # 프로젝트 의존성 라이브러리 목록
```

---

## 🚀 6. 시작 가이드 및 실행 방법 (Getting Started)

### 1️⃣ 저장소 복제 (Clone)

```bash
git clone [https://github.com/pineman1/dongyang_project.git](https://github.com/pineman1/dongyang_project.git)
cd dongyang_project
```

### 2️⃣ 가상환경 구축 및 라이브러리 설치

```bash
python -m venv venv

# Windows PowerShell 환경 실행
.\venv\Scripts\Activate.ps1

# (스크립트 실행 권한 오류 발생 시)
Set-ExecutionPolicy RemoteSigned -Scope Process

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 3️⃣ 애플리케이션 실행

**메인 AI FC 어시스턴트 실행 (기본):**

```bash
streamlit run app.py
```

**1,000명 고객 학습 모델 추천 패널 실행:**

```bash
..\venv\Scripts\python.exe -m pip install -r requirements-premium.txt
..\venv\Scripts\python.exe -m streamlit run premium_app.py
```

---

## 🔄 7. 팀 협업 및 Git 관리 가이드 (Git Workflow)

1. **작업 시작 전 최신 코드 동기화:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **기능 개발 시 브랜치 생성 및 이동:**
   ```bash
   git checkout -b feature/본인이름_기능명
   ```

3. **작업 내용 커밋 및 GitHub 푸시:**
   ```bash
   git add .
   git commit -m "feat: 구현한 기능 요약"
   git push -u origin HEAD
   ```
