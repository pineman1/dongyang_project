import streamlit as st
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from pdf_generator import generate_proposal_pdf

# ==========================================
# 🔑 환경 변수 로드 (Google Gemini API 키)
# ==========================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

from ml_model import get_recommendation
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Google Gemini 연동 라이브러리
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
# 🧠 RAG 및 추천 로직 (Gemini 3.6 Flash 적용)
# ==========================================
@st.cache_resource
def create_rag_chain(file_path, google_api_key):
    if not os.path.exists(file_path):
        st.error(f"❌ 약관 PDF 파일('{file_path}')을 찾을 수 없습니다. 프로젝트 루트에 배치해 주세요.")
        st.stop()
        
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100).split_documents(docs)
    
    # 1. Gemini 임베딩 모델
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=google_api_key
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. 메인 대화 및 스크립트 작성용 LLM (Gemini 3.6 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        google_api_key=google_api_key
    )
    
    # 3. 의도 분류기용 JSON 모드 LLM (Gemini 3.6 Flash)
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        google_api_key=google_api_key,
        model_kwargs={"response_mime_type": "application/json"}
    )
    return retriever, llm, router_llm

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def process_recommendation(customer_info, retriever, llm):
    ml_result = get_recommendation(customer_info)
    if not ml_result["자료있음"] or not ml_result["특약자료있음"]:
        st.session_state.show_tuning = False
        for key in ("last_recommended_product", "last_recommended_rider", "last_customer_info"):
            st.session_state.pop(key, None)
        with st.chat_message("assistant"):
            st.info("자료가 없음")
        st.session_state.messages.append({"role": "assistant", "content": "자료가 없음"})
        return
    override_msg = ""
    
    # 🌟 [고객 지시 우선권 발동]
    preferred = customer_info.get("선호상품")
    if preferred:
        override_msg = f"\n\n🚨 **[고객 니즈 최우선 반영]** 통계 예측을 넘어, 고객님이 원하신 **{preferred}** 위주로 설계했습니다."

    st.session_state.last_recommended_product = ml_result['주계약']
    st.session_state.last_recommended_rider = ml_result['추천특약']
    st.session_state.last_customer_info = customer_info.copy()
    
    status_msg = f"📊 **데이터 분석 완료!**\n- 1순위 추천: **{ml_result['주계약']}**\n- 맞춤 특약: **{ml_result['추천특약']}**{override_msg}"
    
    with st.chat_message("assistant"): 
        st.info(status_msg)
    st.session_state.messages.append({"role": "assistant", "content": status_msg})
    
    prompt = PromptTemplate.from_template("고객 정보와 추천 상품을 바탕으로 세일즈 스크립트를 작성하세요. [정보]: {info} / [주계약]: {product} / [특약]: {rider} / [약관]: {context}")
    rag_chain = (
        {"context": retriever | format_docs, "info": lambda x: str(customer_info), "product": lambda x: ml_result['주계약'], "rider": lambda x: ml_result['추천특약']} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    with st.chat_message("assistant"):
        with st.spinner("스크립트 생성 중..."):
            response = rag_chain.invoke(f"{ml_result['주계약']} {ml_result['추천특약']}")
            st.markdown(response)
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
if user_input := st.chat_input("채팅으로 대화하세요 (예: 30대 남성 고혈압인데 암보험 원해)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): 
        st.markdown(user_input)

    with st.spinner("의도를 파악하고 있습니다..."):
        router_prompt = f"""
        의도 분류: "recommend" (추천), "design" (설계서 발급), "objection" (고객 거절/핑계 대응), "qa" (질문)
        [메시지]: {user_input}
        [반드시 아래 JSON 형식으로만 응답하세요]:
        {{
            "intent": "recommend" 또는 "qa" 또는 "design" 또는 "objection",
            "extracted_info": {{
                "나이": 숫자 또는 null, "성별": "남성" 또는 "여성" 또는 null, "연소득_만원": 숫자 또는 null,
                "직업위험등급": 1, 2, 3 또는 null, "결혼여부": "미혼" 또는 "기혼" 또는 null,
                "자녀수": 숫자 또는 null, "흡연여부": "비흡연" 또는 "흡연" 또는 null,
                "만성질환": "없음", "고혈압", "당뇨" 또는 null, "가족력": "없음", "암", "심혈관" 또는 null,
                "선호상품": "암보험", "종신보험", "연금보험", "유병자보험" 중 하나 또는 null,
                "예산_만원": 숫자 또는 null
            }}
        }}
        """
        try:
            raw_res = router_llm.invoke(router_prompt).content.strip()
            # 마크다운 블록(```json) 감싸짐 안전 제거
            if raw_res.startswith("```json"):
                raw_res = raw_res[7:]
            elif raw_res.startswith("```"):
                raw_res = raw_res[3:]
            if raw_res.endswith("```"):
                raw_res = raw_res[:-3]
            parsed_intent = json.loads(raw_res.strip())
        except Exception:
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
                obj_prompt = PromptTemplate.from_template(
                    """당신은 동양생명의 20년 차 최고 에이스 영업 지점장입니다.
                    설계사(사용자)가 고객의 거절(비싸다, 나중에 하겠다 등)에 부딪혀 조언을 구하고 있습니다.
                    아래 [약관 자료]와 [고객의 거절 내용]을 바탕으로, 설계사가 고객을 다시 설득할 수 있는 '거절 극복(Objection Handling) 스크립트'를 작성해주세요.
                    무조건 가르치려 들지 말고, [공감 -> 논리적 반박(비용 분할, 리스크 강조 등) -> 대안 제시]의 흐름으로 현장감 있고 강력하게 작성하세요.
                    
                    [약관 자료]: {context}
                    [고객의 거절 내용]: {question}
                    """
                )
                obj_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | obj_prompt | llm | StrOutputParser())
                response = obj_chain.invoke(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # [의도 4] 일반 약관 Q&A
    else:
        with st.chat_message("assistant"):
            with st.spinner("약관을 확인 중입니다..."):
                qa_prompt = PromptTemplate.from_template("참고자료를 바탕으로 답하세요. 자료: {context}\n질문: {question}")
                qa_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | qa_prompt | llm | StrOutputParser())
                response = qa_chain.invoke(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
