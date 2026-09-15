# dongyang_project

# 💼 동양생명 하이브리드 세일즈 어시스턴트 (AI FC Assistant)

## 📌 프로젝트 개요
본 프로젝트는 생명보험 설계사(FC)의 영업 경쟁력을 극대화하기 위해 개발된 **'실전형 AI 세일즈 지원 툴'**입니다. 고객 데이터를 기반으로 한 머신러닝(ML) 상품 추천과 언어모델(LLM) 기반의 RAG(검색 증강 생성) 기술을 결합하여, 실시간 가입설계 튜닝 및 거절 극복 화법을 제공합니다.

## 👥 팀원 및 역할 (5인)
* **PO ** 강현석 - 기획 총괄, 프롬프트 엔지니어링 설계, 시나리오 QA 및 전체 일정 관리
* **AI Engineer:** 강아라 - RAG 아키텍처 및 추천 알고리즘 로직 설계,클라우드 배포 인프라 구축 및 서버 환경 세팅
* **Data Scientist:** 김경탁 - 고객 데이터셋(CSV) 전처리 및 ML 추천 모델 고도화
* **AI Engineer:** 강동훈 - 클라우드 배포 인프라 구축 및 서버 환경 세팅,RAG 아키텍처 및 추천 알고리즘 로직 설계
* **AI Software Architect:** 손민근 - 프론트엔드(UI) 통합, 모듈 간 API 연동 및 깃허브 버전 관리

## 🛠️ 핵심 시나리오 및 주요 기능
1. **의도 파악 및 고객 니즈 오버라이드:** 단순 챗봇을 넘어 고객의 의도(추천, 설계, 질의)를 파악하며, 통계 모델보다 고객의 '선호 상품'을 최우선으로 반영하는 유연한 추천 로직
2. **실시간 가입설계 튜닝 (Tuning Panel):** 예산 및 납입 기간을 고객과 실시간으로 조율하고, 확정된 조건으로 즉석에서 PDF 가입설계서 자동 발급
3. **거절 극복 방어 화법 (Objection Handling):** 현장에서 고객이 거절할 경우, 20년 차 지점장 페르소나가 논리적인 반박 화법 및 대안을 제시하여 계약 성사율 제고

## 💻 기술 스택
* **Language:** Python
* **Frontend:** Streamlit
* **AI / ML:** OpenAI API (gpt-4o-mini), LangChain, Scikit-learn (RandomForest)
* **Data / Doc:** Pandas, PyMuPDF, FPDF2

## 🚀 로컬 환경 실행 방법
```bash
# 1. 필수 라이브러리 설치
pip install -r requirements.txt

# 2. .env 파일 세팅 (루트 폴더에 생성 후 아래 내용 입력)
# OPENAI_API_KEY="sk-..." (본인의 API 키 입력)

# 3. 애플리케이션 실행
streamlit run app.py


# 1. 작업 시작 전: 항상 최신 코드 가져오기
git pull origin main

# 2. 작업 완료 후: 수정한 파일 스테이징
git add .

# 3. 커밋 생성: 변경 내용 요약 작성
git commit -m "feat: ML 모델 확률 계산 로직 수정"

# 4. 원격 저장소 반영: 깃허브에 푸시
git push origin main
