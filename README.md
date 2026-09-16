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
1️⃣ 저장소 최초 복제 (최초 1회만)
팀원들이 각자 본인 컴퓨터의 VS Code를 켜고 빈 폴더 터미널에서 깃허브 주소를 내려받습니다.
이 명령어를 치면 팀원 컴퓨터에 깃허브에 올린 파일들이 그대로 복사되어 다운로드됩니다.

Bash
git clone https://github.com/pineman1/dongyang_project
cd dongyang_project

2️⃣ 가상환경 및 라이브러리 세팅 (최초 1회)
각자 컴퓨터에 파이썬 가상환경을 켜고, requirements.txt로 동일한 라이브러리를 일괄 설치합니다.

Bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

오류발생시(입력 후 재실행)
Set-ExecutionPolicy RemoteSigned -Scope Process

+ 94줄 streamlit 관련 오류 발생으로 requirements.txt 수정하였습니다.

3️⃣ 매일 작업 시작: 최신 코드 내려받기 (필수)
다른 팀원이 올린 변경 사항을 내 컴퓨터 폴더로 덮어씌워 일치시킵니다. (작업 시작 전 항상 입력)

Bash
git checkout main
git pull origin main


4️⃣코드 수정 시에는 개인 브랜치 항상 만들어서 하기(main에 직접 하지 않도록 주의)

코드 수정 후 깃허브로 업로드
내 컴퓨터의 VS Code에서 코드를 수정한 뒤 터미널에 아래 3줄을 쳐서 온라인 깃허브에 반영합니다.

Bash
git add .
git commit -m "수정 내용 요약"
git push origin main

💻 애플리케이션 실행 방법
로컬에서 앱이 정상적으로 구동되는지 테스트하려면 터미널에 아래 명령어를 입력합니다.

Bash
streamlit run app.py
