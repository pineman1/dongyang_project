import streamlit as st
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from pdf_generator import generate_proposal_pdf
from datetime import datetime

# ==========================================
# 🔑 환경 변수 로드 (Google Gemini API 키)
# ==========================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

from ml_model import get_recommendation
# app.py 16~20라인 대체
from langchain_community.document_loaders import PyMuPDFLoader

try:
    # 최신 패키지 우선 시도
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    # 구버전 langchain 환경 호환
    from langchain.text_splitter import RecursiveCharacterTextSplitter
# 기존 코드 (삭제):
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🛡️ 추가된 모듈: privacy.py에서 mask_pii 함수를 가져옵니다. 🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
from privacy import mask_pii

# 변경 코드:
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
# Google Gemini 연동 라이브러리
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from login import check_login

import ui  #📌 분리된 UI 모듈 임포트 (ui.py에 작성될 함수들)
# 🛡️ 추가된 모듈: privacy.py에서 mask_pii 함수를 가져옵니다. 🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
from privacy import mask_pii

# 🛡️ 주민등록번호 마스킹 함수 추가 🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
def mask_rrn(text):
    if not isinstance(text, str):
        return text
    # 주민번호 패턴: 생년월일 6자리 + 하이픈(선택) + 성별(1~4) + 나머지 6자리
    pattern = r'(\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))[-?\s]*([1-4])\d{6}'
    return re.sub(pattern, r'\1-\2******', text)


st.set_page_config(page_title="동양생명 AI FC 어시스턴트", layout="wide")

# ==========================================
# 🔐 로그인 세션 상태 관리 및 로그인 화면
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 로그인되어 있지 않은 경우 로그인 폼을 보여주고 코드 실행을 중단(st.stop)
if not st.session_state.logged_in:
    st.title("🔐 동양생명 AI FC 어시스턴트 - 설계사 로그인")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            employee_no = st.text_input("사번", placeholder="사번을 입력하세요")
            password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
            submit_button = st.form_submit_button("로그인", use_container_width=True)
            
            if submit_button:
                if check_login(employee_no, password):
                    st.session_state.logged_in = True
                    st.session_state.employee_no = employee_no
                    st.success("로그인 성공!")
                    st.rerun()  # 로그인 성공 시 화면 즉시 갱신
                else:
                    st.error("사번 또는 비밀번호가 올바르지 않습니다.")
                    
    st.stop()  # 로그인 안 되었으면 아래 메인 앱 로직(AI 어시스턴트) 실행 금지

# ==========================================
# 💼 메인 서비스 (로그인 완료된 사용자만 접근 가능)
# ==========================================
# 사이드바 상단에 로그인 정보 표시 및 로그아웃 버튼 추가
with st.sidebar:
    st.write(f"👤 **{st.session_state.employee_no}** FC님 환영합니다.")
    if st.button("🚪 로그아웃", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

st.set_page_config(page_title="동양생명 AI FC 어시스턴트", layout="wide")
st.title("💼 동양생명 하이브리드 세일즈 어시스턴트")

PDF_FILE_PATH = "dongyang_products.pdf"
REQUIRED_KEYS = {
    "나이": "나이(숫자)", "성별": "성별(남/여)", "연소득_만원": "연소득", 
    "직업위험등급": "직업", "결혼여부": "결혼여부", "자녀수": "자녀 수", 
    "흡연여부": "흡연여부", "만성질환": "만성질환", "가족력": "가족력"
}

# ==========================================
# 🛠️ 다이내믹 Mock API (가상의 보험료 계산기)
# ==========================================
def mock_design_api_v2(customer_info, product_name, include_rider, payment_term):
    age = customer_info.get("나이", 40)
    
    # 1. 기본료 세팅
    base_premium = 50000 if "연금" in product_name else 30000
    premium = base_premium + (age * 1200) 
    
    # 2. 질환/흡연 가산금
    if customer_info.get("만성질환") in ["고혈압", "당뇨"]: 
        premium += 15000
    if customer_info.get("흡연여부") == "흡연": 
        premium += 8000
        
    # 3. 특약 추가 여부에 따른 금액 변동
    if include_rider:
        premium += 12500
        
    # 4. 납입 기간에 따른 변동
    if payment_term == "10년납": 
        premium = int(premium * 1.4)
    elif payment_term == "30년납": 
        premium = int(premium * 0.75)
    else: 
        premium = int(premium) # 20년납 기준
        
    return {
        "product_name": product_name,
        "monthly_premium": premium,
        "include_rider": include_rider,
        "payment_term": payment_term,
        "coverage_1": "사망 및 고도장해 보장" if "종신" in product_name else "암 진단 및 수술 보장",
        "coverage_2": "맞춤형 특약 혜택 포함" if include_rider else "특약 미포함 (기본 보장만 적용)"
    }

# ==========================================
# 🛠️ 실시간 반영 PDF 생성기
# ==========================================
#def generate_proposal_pdf(design_result):
    pdf = FPDF()
    pdf.add_page()
    
    font_path = "C:/Windows/Fonts/malgun.ttf"
#    if os.path.exists(font_path):
#        pdf.add_font("Malgun", "", font_path, uni=True)
#        pdf.set_font("Malgun", "", 18)
#    else:
#        pdf.set_font("Helvetica", "B", 18)
        
#    pdf.cell(0, 15, "[ 동양생명 AI 가입설계 제안서 ]", ln=True, align="C")
#    pdf.ln(10)
    
#    if os.path.exists(font_path): 
#        pdf.set_font("Malgun", "", 12)
        
#    pdf.cell(0, 10, f"▶ 추천 상품명 :  {design_result['product_name']}", ln=True)
#    pdf.cell(0, 10, f"▶ 납입 기간 :  {design_result['payment_term']}", ln=True)
#    pdf.cell(0, 10, f"▶ 최종 월 보험료 :  {design_result['monthly_premium']:,} 원", ln=True)
#   pdf.ln(5)
 #   pdf.cell(0, 10, f"✔ 핵심 보장 1 : {design_result['coverage_1']}", ln=True)
  #  pdf.cell(0, 10, f"✔ 핵심 보장 2 : {design_result['coverage_2']}", ln=True)
    
   # pdf.ln(20)
    #pdf.cell(0, 10, "* 본 제안서는 AI 추천 알고리즘에 의해 고객 맞춤형으로 산출된 가상의 결과물입니다.", ln=True)
    
    #return bytes(pdf.output())

# ==========================================
# 🧠 RAG 및 추천 로직 (FAISS 로컬 저장 적용)
# ==========================================
INDEX_SAVE_PATH = "faiss_index"  # 로컬에 Vector DB를 저장할 폴더명

@st.cache_resource
def create_rag_chain(file_path, google_api_key):
    # 1. Gemini 임베딩 모델 설정
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=google_api_key
    )
    
    # 2. 로컬에 저장된 FAISS 인덱스가 이미 존재하는지 확인 (캐싱)
    if os.path.exists(INDEX_SAVE_PATH):
        # 이미 한번 인덱싱한 적이 있다면 로컬 폴더에서 바로 불러옴 (속도 극대화, API 호출 0건)
        vectorstore = FAISS.load_local(
            INDEX_SAVE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        # 로컬 저장소가 없을 때만 최초 1회 PDF 로드 및 인덱싱 수행
        if not os.path.exists(file_path):
            st.error(f"❌ 약관 PDF 파일('{file_path}')을 찾을 수 없습니다. 프로젝트 루트에 배치해 주세요.")
            st.stop()
            
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100).split_documents(docs)
        
        # 임베딩 생성 후 faiss_index 폴더로 저장
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_SAVE_PATH)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. 메인 대화 및 스크립트 작성용 LLM (Gemini 3.6 Flash) 구현이 오래걸려서 1.5로 바꿔봄
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        google_api_key=google_api_key
    )
    
    # 4. 의도 분류기용 JSON 모드 LLM (Gemini 3.6 Flash)일단 1.5로 줄여봄
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        google_api_key=google_api_key,
        model_kwargs={"response_mime_type": "application/json"}
    )
    return retriever, llm, router_llm

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def process_recommendation(customer_info, retriever, llm):
    ml_result = get_recommendation(customer_info)
    override_msg = ""
    
    # 🌟 [고객 지시 우선권 발동]
    preferred = customer_info.get("선호상품")
    if preferred:
        if preferred == "암보험": 
            ml_result.update({'주계약': "무배당 수호천사 암/건강보험", '주계약_확률': 99.9, '추천특약': "표적항암약물허가치료 특약"})
        elif preferred == "연금보험": 
            ml_result.update({'주계약': "수호천사 행복 연금보험", '주계약_확률': 99.9, '추천특약': "특약 없음"})
        elif preferred == "종신보험": 
            ml_result.update({'주계약': "무배당 수호천사 우리가족 종신보험", '주계약_확률': 99.9})
        elif preferred == "유병자보험": 
            ml_result.update({'주계약': "무배당 수호천사 간편심사보험", '주계약_확률': 99.9})
        override_msg = f"\n\n🚨 **[고객 니즈 최우선 반영]** 통계 예측을 넘어, 고객님이 원하신 **{preferred}** 위주로 설계했습니다."

    st.session_state.last_recommended_product = ml_result['주계약']
    st.session_state.last_recommended_rider = ml_result['추천특약']
    st.session_state.last_customer_info = customer_info.copy()
    
    status_msg = f"📊 **데이터 분석 완료!**\n- 1순위 추천: **{ml_result['주계약']}**\n- 맞춤 특약: **{ml_result['추천특약']}**{override_msg}"
    
    with st.chat_message("assistant"): 
        st.info(status_msg)
    st.session_state.messages.append({"role": "assistant", "content": status_msg})
    
    # 1. 초경량 프롬프트로 교체 (구글 서버 부하 최소화)
    prompt = PromptTemplate.from_template(
        "당신은 동양생명 AI 보험 설계사입니다. [약관]을 바탕으로 추천된 [주계약]과 [특약]을 고객에게 권유하는 핵심 세일즈 화법을 3~4문장의 구어체로 짧게 작성하세요. 표나 장황한 설명은 절대 금지합니다.\n[정보]: {info}\n[주계약]: {product}\n[특약]: {rider}\n[약관]: {context}"
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "info": lambda x: str(customer_info), "product": lambda x: ml_result['주계약'], "rider": lambda x: ml_result['추천특약']} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    with st.chat_message("assistant"):
        with st.spinner("스크립트 생성 중..."):
            # 2. 방어막(try-except) 설치 (에러 발생 시 앱 다운 방지)
            try:
                response = rag_chain.invoke(f"{ml_result['주계약']} {ml_result['추천특약']}")
                st.markdown(response)
            except Exception as e:
                response = "⚠️ 현재 구글 AI 서버 트래픽이 많아 스크립트 생성이 지연되었습니다. 추천(ML)은 완료되었으니, 이어서 '설계서 뽑아줘'를 입력하시거나 잠시 후 다시 시도해 주세요."
                st.error(response)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.collected_info = {}

# API 키 검증
if not api_key:
    st.error("⚠️ `.env` 파일에 `GOOGLE_API_KEY`를 설정해주세요.")
    st.stop()

# 세션 상태 초기화
if "rag_components" not in st.session_state: 
    st.session_state.rag_components = create_rag_chain(PDF_FILE_PATH, api_key)
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "고객 정보를 입력하시고, 언제든 '설계서 뽑아줘'라고 말씀하세요!"}]
if "collected_info" not in st.session_state: 
    st.session_state.collected_info = {}

retriever, llm, router_llm = st.session_state.rag_components

# ==========================================
# ⚙️ 사이드바 (수동 입력 및 조건 선택)
# ==========================================
with st.sidebar:
    st.header("⚙️ 고객 수동 입력")
    age = st.number_input("나이", value=None)
    gender = st.selectbox("성별", ["선택 안함", "남성", "여성"])
    income = st.number_input("연소득 (만원)", value=None)
    job_risk = st.selectbox("직업 위험등급", ["선택 안함", 1, 2, 3])
    marital = st.selectbox("결혼 여부", ["선택 안함", "미혼", "기혼"])
    child = st.number_input("자녀 수", value=None)
    smoke = st.selectbox("흡연 여부", ["선택 안함", "비흡연", "흡연"])
    disease = st.selectbox("만성질환", ["선택 안함", "없음", "고혈압", "당뇨"])
    family = st.selectbox("가족력", ["선택 안함", "없음", "암", "심혈관"])
    
    st.markdown("---")
    st.subheader("💰 희망 조건 (선택)")
    budget = st.number_input("희망 예산 (월/만원)", value=None)
    pref_term = st.selectbox("희망 납입기간", ["선택 안함", "10년납", "20년납", "30년납"])
    
    if st.button("🚀 위 정보로 추천받기", use_container_width=True):
        manual_data = {"나이": age, "성별": gender, "연소득_만원": income, "직업위험등급": job_risk, "결혼여부": marital, "자녀수": child, "흡연여부": smoke, "만성질환": disease, "가족력": family}
        for k, v in manual_data.items():
            if v is not None and v != "선택 안함": 
                st.session_state.collected_info[k] = v
        missing_keys = [k for k in REQUIRED_KEYS.keys() if k not in st.session_state.collected_info]
        if missing_keys:
            st.error(f"누락된 정보: {', '.join([REQUIRED_KEYS[k] for k in missing_keys])}")
        else:
            process_recommendation(st.session_state.collected_info, retriever, llm)

# ==========================================
# 🎛️ 실시간 가입설계 튜닝 패널
# ==========================================
if st.session_state.get("show_tuning", False):
    st.markdown("---")
    st.success("🛠️ **가입설계 튜닝 모드**가 활성화되었습니다. 고객과 함께 옵션을 조율해 보세요!")
    
    with st.expander("📊 실시간 조율 패널 (Tuning Panel)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 옵션 선택")
            sel_term = st.selectbox("납입 기간 선택", ["10년납", "20년납", "30년납"], index=1)
            rider_name = st.session_state.last_recommended_rider
            has_rider = rider_name != "특약 없음"
            
            sel_rider = False
            if has_rider:
                sel_rider = st.checkbox(f"[{rider_name}] 보장 추가하기", value=True)
            else:
                st.info("이 상품은 기본 보장으로만 구성됩니다.")
                
        with col2:
            st.markdown("#### 실시간 산출 결과")
            calc_result = mock_design_api_v2(
                st.session_state.last_customer_info, 
                st.session_state.last_recommended_product, 
                sel_rider, 
                sel_term
            )
            
            st.metric(label="월 예상 납입 보험료", value=f"{calc_result['monthly_premium']:,} 원")
            st.write(f"✔ 기본 보장: {calc_result['coverage_1']}")
            st.write(f"✔ 추가 혜택: {calc_result['coverage_2']}")
        
        st.markdown("---")
        # customer_info를 함께 전달하여 제안서에 인적사항도 포함
        pdf_bytes = generate_proposal_pdf(
            calc_result, 
            customer_info=st.session_state.get("last_customer_info")
        )
        st.download_button(
            label="📥 이 조건으로 최종 가입설계서 발급하기 (PDF)",
            data=pdf_bytes,
            file_name=f"동양생명_맞춤설계_{calc_result['product_name']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
            
# 대화 내용 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

# ==========================================
# 💬 채팅 인터페이스 및 의도 라우팅
# ==========================================
raw_user_input = st.chat_input("채팅으로 대화하세요 (예: 30대 남성 고혈압인데 암보험 원해)")   #🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
if raw_user_input:
    # 1. 입력 마스킹 (privacy.py 의 mask_pii 통과)  
    user_input = mask_pii(raw_user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): 
        st.markdown(user_input)

    with st.spinner("의도를 파악하고 있습니다..."):
        current_year = datetime.now().year  # 시스템 현재 연도 자동 계산
        
        router_prompt = f"""
        당신은 동양생명의 최고 수준 데이터 정제 AI입니다.
        고객의 모호한 일상 자연어 입력을 분석하여, 시스템이 요구하는 정확한 형태의 JSON 데이터로만 변환하세요.
        현재 연도는 {current_year}년입니다. 나이 계산 시 반드시 이 연도를 기준점으로 삼으세요.

        [데이터 변환 핵심 규칙 (매우 중요)]
        1. 나이: "98년생", "올해 서른" 등은 {current_year}년을 기준으로 계산하여 순수 정수(예: 28)로 표기.
        2. 성별: "애엄마", "아저씨", "군필" 등 일상 단어에서 유추하여 "남성" 또는 "여성"으로 맵핑.
        3. 연소득: "월 300", "세후 250"은 12를 곱해 연소득 정수(3600, 3000)로, "3~4천"은 중간값(3500)으로 변환.
        4. 직업: "사무직/주부/백수"-> 1, "식당/영업/외근"-> 2, "노가다/배달/현장직"-> 3 (정수로 맵핑).
        5. 결혼/자녀: "이혼", "사별", "돌싱"은 무조건 "미혼"으로 처리. "딩크족", "애는 와이프가 키워요" 등 현재 부양하지 않으면 자녀수 0으로 처리.
        6. 건강/가족: "혈압약", "당 수치 높음" -> "고혈압", "당뇨" / "아버지가 위암", "할아버지 쓰러지심" -> "암", "심혈관" / "전담 펴요" -> "흡연".
        7. 상품/예산: "죽으면 나오는거"->"종신보험", "늙어서 타는거"->"연금보험", "아파도 되는거"->"유병자보험" / "10만원 안팎" -> 10.

        [예시 학습 (Few-Shot)]
        입력: "저 98년생 전담 피는 남자고요, 배달 투잡뜁니다. 월 250 벌고 딩크족이라 애는 없어요. 건강하고 가족 다 건강함. 달마다 5만원으로 암보험 알아봐요."
        출력: {{"intent": "recommend", "extracted_info": {{"나이": {current_year-1998}, "성별": "남성", "연소득_만원": 3000, "직업위험등급": 3, "결혼여부": "기혼", "자녀수": 0, "흡연여부": "흡연", "만성질환": "없음", "가족력": "없음", "선호상품": "암보험", "예산_만원": 5}}}}

        입력: "올해 쉰둘 애엄마인데 작년에 이혼해서 혼자 살아요. 애들은 남편이 키워서 자녀 없고요. 식당 주방 일하며 연봉 4천 받아요. 혈압약 먹고 있고 친정 아버지가 심근경색으로 돌아가셨어요. 비흡연자고 아파도 가입되는 유병자 찾아요."
        출력: {{"intent": "recommend", "extracted_info": {{"나이": 52, "성별": "여성", "연소득_만원": 4000, "직업위험등급": 2, "결혼여부": "미혼", "자녀수": 0, "흡연여부": "비흡연", "만성질환": "고혈압", "가족력": "심혈관", "선호상품": "유병자보험", "예산_만원": null}}}}

        [메시지]: {user_input}
        [반드시 아래 JSON 형식으로만 응답하세요]:
        {{
            "intent": "recommend" | "qa" | "design" | "objection",
            "extracted_info": {{
                "나이": (정수), "성별": ("남성"|"여성"), "연소득_만원": (정수),
                "직업위험등급": (1|2|3), "결혼여부": ("미혼"|"기혼"),
                "자녀수": (정수), "흡연여부": ("비흡연"|"흡연"),
                "만성질환": ("없음"|"고혈압"|"당뇨"), "가족력": ("없음"|"암"|"심혈관"),
                "선호상품": ("암보험"|"종신보험"|"연금보험"|"유병자보험"),
                "예산_만원": (정수)
            }}
        }}
        """
        raw_res = ""
        try:
            content_obj = router_llm.invoke(router_prompt).content
            
            # 리스트 타입으로 들어올 경우 텍스트만 안전하게 추출
            if isinstance(content_obj, list):
                raw_res = content_obj[0].get("text", "")
            else:
                raw_res = str(content_obj)
            
            # 마크다운 방어 로직
            raw_res = raw_res.replace("```json", "").replace("```", "").strip()
            parsed_intent = json.loads(raw_res)
            
        except Exception as e:
            print(f"🚨 라우팅 파싱 에러 발생: {e} \n[AI 원본 응답]: {raw_res}")
            parsed_intent = {"intent": "qa", "extracted_info": {}}
    intent = parsed_intent.get("intent")
    
    # [의도 1] 추천
    if intent == "recommend":
        st.session_state.show_tuning = False
        extracted = parsed_intent.get("extracted_info", {})
        for k, v in extracted.items():
            if v is not None and k in REQUIRED_KEYS: 
                st.session_state.collected_info[k] = v
            if k == "선호상품" and v is not None: 
                st.session_state.collected_info["선호상품"] = v
            
        missing_keys = [k for k in REQUIRED_KEYS.keys() if k not in st.session_state.collected_info]
        if missing_keys:
            ask_msg = f"정확한 보험 추천을 위해 다음 정보가 더 필요합니다.\n\n👉 **누락된 정보:** {', '.join([REQUIRED_KEYS[k] for k in missing_keys])}"
            with st.chat_message("assistant"): 
                st.markdown(ask_msg)
            st.session_state.messages.append({"role": "assistant", "content": ask_msg})
        else:
            process_recommendation(st.session_state.collected_info, retriever, llm)

    # [의도 2] 가입설계 튜닝
    elif intent == "design":
        if "last_recommended_product" not in st.session_state:
            reply = "⚠️ 먼저 고객 정보를 입력하여 상품 추천을 받은 뒤에 설계서를 요청해 주세요!"
            with st.chat_message("assistant"): 
                st.warning(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.session_state.show_tuning = True
            st.rerun()
            
    # [의도 3] 거절 극복 화법
    elif intent == "objection":
        with st.chat_message("assistant"):
            with st.spinner("최고의 영업 실장 모드로 거절 극복 화법을 작성 중입니다..."):
                try:
                    obj_prompt = PromptTemplate.from_template(...) # 기존 프롬프트
                    obj_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | obj_prompt | llm | StrOutputParser())
                    # 변경 후 (수정된 코드):
                    response = obj_chain.invoke(user_input)
                    # 💡 출력 마스킹: AI 응답에 개인정보가 포함될 가능성을 원천 차단 🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
                    response = mask_pii(response)
                    st.markdown(response)
                except Exception as e:
                    # 🚨 429 한도 초과 및 서버 에러 발생 시 방어 로직
                    response = "⚠️ 현재 이용자가 많아 AI 서버 통신이 지연되고 있습니다. 약 30초 후 다시 시도해 주세요."
                    st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # [의도 4] 일반 약관 Q&A
    else:
        with st.chat_message("assistant"):
            with st.spinner("약관을 확인 중입니다..."):
                try:
                    qa_prompt = PromptTemplate.from_template(...) # 기존 프롬프트
                    qa_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | qa_prompt | llm | StrOutputParser())
                    response = qa_chain.invoke(user_input)
                    response = mask_rrn(raw_response)  # 👈 AI 응답 마스킹
                    st.markdown(response)
                except Exception as e:
                    # 🚨 에러 방어 로직
                    response = "⚠️ 현재 이용자가 많아 AI 서버 통신이 지연되고 있습니다. 약 30초 후 다시 질문해 주세요."
                    st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})