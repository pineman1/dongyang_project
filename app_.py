import streamlit as st
import os
import re
import json
import time
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

from ml_model import get_recommendation
from langchain_community.document_loaders import PyMuPDFLoader
# [수정됨] 최신 버전 맞춰 패키지 분리 적용
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="동양생명 AI FC 어시스턴트", layout="wide")
st.title("💼 동양생명 하이브리드 세일즈 어시스턴트")

PDF_FILE_PATH = "dongyang_products.pdf"
FAISS_INDEX_PATH = "faiss_index"

REQUIRED_KEYS = {
    "나이": "나이(숫자)", "성별": "성별(남/여)", "연소득_만원": "연소득", 
    "직업위험등급": "직업", "결혼여부": "결혼여부", "자녀수": "자녀 수", 
    "흡연여부": "흡연여부", "만성질환": "만성질환", "가족력": "가족력"
}

# ==========================================
# 🛠️ 다이내믹 Mock API (보험료 계산기)
# ==========================================
def mock_design_api_v2(customer_info, product_name, include_rider, payment_term):
    age = customer_info.get("나이", 40)
    
    base_premium = 50000 if "연금" in product_name else 30000
    premium = base_premium + (age * 1200) 
    
    if customer_info.get("만성질환") in ["고혈압", "당뇨"]: 
        premium += 15000
    if customer_info.get("흡연여부") == "흡연": 
        premium += 8000
        
    if include_rider: 
        premium += 12500
        
    if payment_term == "10년납": 
        premium = int(premium * 1.4)
    elif payment_term == "30년납": 
        premium = int(premium * 0.75)
    else: 
        premium = int(premium)
        
    return {
        "product_name": product_name,
        "monthly_premium": premium,
        "include_rider": include_rider,
        "payment_term": payment_term,
        "coverage_1": "사망 및 고도장해 보장" if "종신" in product_name else "암 진단 및 수술 보장",
        "coverage_2": "맞춤형 특약 혜택 포함" if include_rider else "특약 미포함 (기본 보장만 적용)"
    }

# ==========================================
# 🛠️ PDF 생성기
# ==========================================
def generate_proposal_pdf(design_result):
    pdf = FPDF()
    pdf.add_page()
    
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        pdf.add_font("Malgun", "", font_path, uni=True)
        pdf.set_font("Malgun", "", 18)
    else:
        pdf.set_font("Helvetica", "B", 18)
        
    pdf.cell(0, 15, "[ 동양생명 AI 가입설계 제안서 ]", ln=True, align="C")
    pdf.ln(10)
    
    if os.path.exists(font_path): 
        pdf.set_font("Malgun", "", 12)
        
    pdf.cell(0, 10, f"▶ 추천 상품명 :  {design_result['product_name']}", ln=True)
    pdf.cell(0, 10, f"▶ 납입 기간 :  {design_result['payment_term']}", ln=True)
    pdf.cell(0, 10, f"▶ 최종 월 보험료 :  {design_result['monthly_premium']:,} 원", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, f"✔ 핵심 보장 1 : {design_result['coverage_1']}", ln=True)
    pdf.cell(0, 10, f"✔ 핵심 보장 2 : {design_result['coverage_2']}", ln=True)
    
    pdf.ln(20)
    pdf.cell(0, 10, "* 본 제안서는 AI 추천 알고리즘에 의해 고객 맞춤형으로 산출된 가상의 결과물입니다.", ln=True)
    
    return bytes(pdf.output())

# ==========================================
# 🛠️ RAG 파이프라인
# ==========================================
@st.cache_resource
def create_rag_chain(file_path, gemini_key):
    # [수정됨] 안정적인 구형 임베딩 모델로 롤백 (404 에러 원인 해결)
    # 수정된 코드 (현재 공식 지원되는 최신 임베딩 모델 적용)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=gemini_key
    )
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        
        batch_size = 15
        vectorstore = None
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            time.sleep(0.5) 
            
        vectorstore.save_local(FAISS_INDEX_PATH)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 기존 "gemini-1.5-flash"를 "gemini-pro"로 변경
    # 기존 "gemini-pro"를 "gemini-1.0-pro"로 변경
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.7, 
        google_api_key=gemini_key
    )
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.0, 
        google_api_key=gemini_key
    )
    return retriever, llm, router_llm

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def extract_clean_json(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(text)

def process_recommendation(customer_info, retriever, llm):
    ml_result = get_recommendation(customer_info)
    override_msg = ""
    
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
        override_msg = f"\n\n🚨 **[고객 니즈 최우선 반영]** 고객님이 희망하신 **{preferred}** 위주로 설계되었습니다."

    st.session_state.last_recommended_product = ml_result['주계약']
    st.session_state.last_recommended_rider = ml_result['추천특약']
    st.session_state.last_customer_info = customer_info.copy()
    
    status_msg = f"📊 **데이터 분석 완료!**\n- 1순위 추천: **{ml_result['주계약']}**\n- 맞춤 특약: **{ml_result['추천특약']}**{override_msg}"
    
    with st.chat_message("assistant"): 
        st.info(status_msg)
    st.session_state.messages.append({"role": "assistant", "content": status_msg})
    
    prompt = PromptTemplate.from_template(
        "당신은 동양생명 전문 FC입니다. 아래 정보를 종합하여 고객 맞춤형 세일즈 스크립트를 설득력 있게 작성하세요.\n\n"
        "[고객 정보]: {info}\n"
        "[주계약]: {product}\n"
        "[추천특약]: {rider}\n"
        "[상품 약관]:\n{context}"
    )
    rag_chain = (
        {"context": retriever | format_docs, "info": lambda x: str(customer_info), "product": lambda x: ml_result['주계약'], "rider": lambda x: ml_result['추천특약']} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    with st.chat_message("assistant"):
        with st.spinner("전문 세일즈 화법 구성 중..."):
            response = rag_chain.invoke(f"{ml_result['주계약']} {ml_result['추천특약']}")
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.collected_info = {}

# ==========================================
# 상태 초기화
# ==========================================
if not api_key:
    st.error("🔑 API 키 누락: `.env` 파일에 GEMINI_API_KEY를 입력하세요.")
    st.stop()

if "rag_components" not in st.session_state: 
    with st.spinner("약관 문서를 인덱싱하고 Gemini 엔진을 연결하는 중입니다... (최초 1회 약 20초 소요)"):
        st.session_state.rag_components = create_rag_chain(PDF_FILE_PATH, api_key)

if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 고객 정보를 입력하시거나 상품에 대해 문의해 주세요."}]
if "collected_info" not in st.session_state: 
    st.session_state.collected_info = {}

retriever, llm, router_llm = st.session_state.rag_components

# ==========================================
# 사이드바
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
        manual_data = {
            "나이": age, "성별": gender, "연소득_만원": income, 
            "직업위험등급": job_risk, "결혼여부": marital, "자녀수": child, 
            "흡연여부": smoke, "만성질환": disease, "가족력": family
        }
        for k, v in manual_data.items():
            if v is not None and v != "선택 안함": 
                st.session_state.collected_info[k] = v
        missing_keys = [k for k in REQUIRED_KEYS.keys() if k not in st.session_state.collected_info]
        if missing_keys:
            st.error(f"누락된 항목: {', '.join([REQUIRED_KEYS[k] for k in missing_keys])}")
        else:
            process_recommendation(st.session_state.collected_info, retriever, llm)

# ==========================================
# 🎛️ 실시간 가입설계 튜닝 패널
# ==========================================
if st.session_state.get("show_tuning", False):
    st.markdown("---")
    st.success("🛠️ **가입설계 튜닝 모드**가 열렸습니다.")
    
    with st.expander("📊 실시간 조율 패널 (Tuning Panel)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 옵션 선택")
            sel_term = st.selectbox("납입 기간 선택", ["10년납", "20년납", "30년납"], index=1)
            rider_name = st.session_state.get("last_recommended_rider", "특약 없음")
            has_rider = (rider_name != "특약 없음")
            
            sel_rider = False
            if has_rider:
                sel_rider = st.checkbox(f"[{rider_name}] 보장 추가하기", value=True)
            else:
                st.info("이 상품은 기본 보장으로만 구성됩니다.")
                
        with col2:
            st.markdown("#### 실시간 산출 결과")
            calc_result = mock_design_api_v2(
                st.session_state.get("last_customer_info", {}), 
                st.session_state.get("last_recommended_product", "동양생명 맞춤보험"), 
                sel_rider, 
                sel_term
            )
            
            st.metric(label="월 예상 납입 보험료", value=f"{calc_result['monthly_premium']:,} 원")
            st.write(f"✔ 기본 보장: {calc_result['coverage_1']}")
            st.write(f"✔ 추가 혜택: {calc_result['coverage_2']}")
            
        st.markdown("---")
        pdf_bytes = generate_proposal_pdf(calc_result)
        st.download_button(
            label="📥 이 조건으로 최종 가입설계서 발급하기 (PDF)",
            data=pdf_bytes,
            file_name=f"동양생명_맞춤설계_{calc_result['product_name']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ==========================================
# 대화 히스토리 출력
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

# ==========================================
# 대화 입력 & 라우팅 로직
# ==========================================
if user_input := st.chat_input("채팅으로 대화하세요 (예: 30대 남성 고혈압인데 암보험 원해 / 설계서 뽑아줘)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): 
        st.markdown(user_input)

    with st.spinner("입력 의도를 분석하고 있습니다..."):
        router_prompt = f"""
당신은 보험 상담 라우터입니다. 반드시 다른 설명 없이 유효한 JSON 형식으로만 응답하세요.

의도(intent) 분류:
- "recommend": 고객 정보를 입력하며 상품 추천을 원하는 경우
- "design": 설계서, 견적서, PDF 발급을 요청하는 경우
- "objection": 가격이 비싸다, 나중에 하겠다 등 거절 핑계를 대는 경우
- "qa": 일반 상품 질문, 약관 문의

[사용자 메시지]: {user_input}

[JSON 포맷 규격]:
{{
    "intent": "recommend" | "qa" | "design" | "objection",
    "extracted_info": {{
        "나이": 숫자 또는 null,
        "성별": "남성" | "여성" | null,
        "연소득_만원": 숫자 또는 null,
        "직업위험등급": 1 | 2 | 3 | null,
        "결혼여부": "미혼" | "기혼" | null,
        "자녀수": 숫자 또는 null,
        "흡연여부": "비흡연" | "흡연" | null,
        "만성질환": "없음" | "고혈압" | "당뇨" | null,
        "가족력": "없음" | "암" | "심혈관" | null,
        "선호상품": "암보험" | "종신보험" | "연금보험" | "유병자보험" | null
    }}
}}
"""
        try: 
            raw_res = router_llm.invoke(router_prompt).content
            parsed_intent = extract_clean_json(raw_res)
        except Exception: 
            parsed_intent = {"intent": "qa", "extracted_info": {}}

    intent = parsed_intent.get("intent")
    
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
            ask_msg = f"정확한 보험 설계를 위해 아래 정보를 추가로 알려주세요.\n\n👉 **필요 정보:** {', '.join([REQUIRED_KEYS[k] for k in missing_keys])}"
            with st.chat_message("assistant"): 
                st.markdown(ask_msg)
            st.session_state.messages.append({"role": "assistant", "content": ask_msg})
        else:
            process_recommendation(st.session_state.collected_info, retriever, llm)

    elif intent == "design":
        if "last_recommended_product" not in st.session_state:
            reply = "⚠️ 먼저 고객 정보를 입력하여 상품 추천을 받으신 후 설계서를 요청해 주세요."
            with st.chat_message("assistant"): 
                st.warning(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.session_state.show_tuning = True
            st.rerun()
            
    elif intent == "objection":
        with st.chat_message("assistant"):
            with st.spinner("거절 극복 세일즈 화법을 작성하고 있습니다..."):
                obj_prompt = PromptTemplate.from_template(
                    "당신은 동양생명 수석 영업 전문가입니다. 고객의 거절/망설임에 대해 [공감 -> 논리적 반박 -> 현실적 대안]의 3단계로 명확하고 강력한 설득 스크립트를 작성해 주세요.\n\n"
                    "[약관 참고자료]:\n{context}\n\n"
                    "[고객의 거절 내용]:\n{question}"
                )
                obj_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
                    | obj_prompt 
                    | llm 
                    | StrOutputParser()
                )
                response = obj_chain.invoke(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        with st.chat_message("assistant"):
            with st.spinner("동양생명 약관을 검색 중입니다..."):
                qa_prompt = PromptTemplate.from_template(
                    "동양생명 전문 FC 어시스턴트로서 아래 공식 약관 내용을 기반으로 고객 질문에 정확하고 친절하게 답변해 주세요.\n\n"
                    "[공식 약관]:\n{context}\n\n"
                    "[질문]:\n{question}"
                )
                qa_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
                    | qa_prompt 
                    | llm 
                    | StrOutputParser()
                )
                response = qa_chain.invoke(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
