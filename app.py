import streamlit as st
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# ----------------------------------------------------
# 📌 분리된 UI 모듈 임포트 (ui.py에 작성될 함수들)
# ----------------------------------------------------
import ui 

# ----------------------------------------------------
# 📌 내부 비즈니스 로직 및 AI 모듈 임포트
# ----------------------------------------------------
from ml_model import get_recommendation
from login import check_login
from privacy import mask_pii
from pdf_generator import generate_proposal_pdf

from langchain_community.document_loaders import PyMuPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 🔑 환경 변수 및 설정
# ==========================================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

PDF_FILE_PATH = "dongyang_products.pdf"
INDEX_SAVE_PATH = "faiss_index"

REQUIRED_KEYS = {
    "나이": "나이(숫자)", "성별": "성별(남/여)", "연소득_만원": "연소득", 
    "직업위험등급": "직업", "결혼여부": "결혼여부", "자녀수": "자녀 수", 
    "흡연여부": "흡연여부", "만성질환": "만성질환", "가족력": "가족력"
}

st.set_page_config(page_title="동양생명 AI FC 어시스턴트", layout="wide")

# ==========================================
# 🛡️ 보안 / 유틸리티 함수
# ==========================================
def mask_rrn(text):
    if not isinstance(text, str):
        return text
    pattern = r'(\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))[-?\s]*([1-4])\d{6}'
    return re.sub(pattern, r'\1-\2******', text)

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
# 🧠 RAG 및 추천 로직
# ==========================================
@st.cache_resource
def create_rag_chain(file_path, google_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=google_api_key
    )
    
    if os.path.exists(INDEX_SAVE_PATH):
        vectorstore = FAISS.load_local(
            INDEX_SAVE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(file_path):
            st.error(f"❌ 약관 PDF 파일('{file_path}')을 찾을 수 없습니다.")
            st.stop()
            
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100).split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_SAVE_PATH)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    router_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key,
        model_kwargs={"response_mime_type": "application/json"}
    )
    return retriever, llm, router_llm

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

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
        override_msg = f"\n\n🚨 **[고객 니즈 최우선 반영]** 통계 예측을 넘어, 고객님이 원하신 **{preferred}** 위주로 설계했습니다."

    st.session_state.last_recommended_product = ml_result['주계약']
    st.session_state.last_recommended_rider = ml_result['추천특약']
    st.session_state.last_customer_info = customer_info.copy()
    
    status_msg = f"📊 **데이터 분석 완료!**\n- 1순위 추천: **{ml_result['주계약']}**\n- 맞춤 특약: **{ml_result['추천특약']}**{override_msg}"
    
    with st.chat_message("assistant"): 
        st.info(status_msg)
    st.session_state.messages.append({"role": "assistant", "content": status_msg})
    
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
            try:
                response = rag_chain.invoke(f"{ml_result['주계약']} {ml_result['추천특약']}")
                st.markdown(response)
            except Exception as e:
                response = "⚠️ 현재 구글 AI 서버 트래픽이 많아 스크립트 생성이 지연되었습니다."
                st.error(response)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.collected_info = {}

# ==========================================
# 🔐 로그인 및 앱 초기화
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    ui.render_login_screen(check_login)
    st.stop()

if not api_key:
    st.error("⚠️ `.env` 파일에 `GOOGLE_API_KEY`를 설정해주세요.")
    st.stop()

if "rag_components" not in st.session_state: 
    st.session_state.rag_components = create_rag_chain(PDF_FILE_PATH, api_key)
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "고객 정보를 입력하시고, 언제든 '설계서 뽑아줘'라고 말씀하세요!"}]
if "collected_info" not in st.session_state: 
    st.session_state.collected_info = {}

retriever, llm, router_llm = st.session_state.rag_components

st.title("💼 동양생명 하이브리드 세일즈 어시스턴트")

# ==========================================
# 🖥️ UI 컴포넌트 렌더링 (탭 분할 및 빠른 질문 적용)
# ==========================================
ui.render_sidebar(REQUIRED_KEYS, process_recommendation, retriever, llm)

# 1. 탭(Tabs) 생성
tab_chat, tab_tuning = st.tabs(["💬 AI 상담 챗봇", "📊 가입설계 및 튜닝"])

# 2. 첫 번째 탭: 챗봇 영역
with tab_chat:
    ui.render_chat_history()
    
    # 빠른 질문 버튼 표시
    quick_query = ui.render_quick_actions()
    
    # 사용자 입력창 (빠른 질문을 눌렀거나, 직접 타이핑한 경우 모두 처리)
    user_input = st.chat_input("질문이나 요청사항을 입력하세요...")
    final_query = quick_query or user_input
    
    if final_query:
        # 화면에 사용자 메시지 출력
        with st.chat_message("user"):
            st.markdown(final_query)
        st.session_state.messages.append({"role": "user", "content": final_query})
        
        # ----------------------------------------------------
        # 여기에 기존의 RAG 검색 및 LLM 답변 생성 로직이 들어갑니다.
        # (기존 app.py에 있던 사용자 입력 처리 로직을 이 아래에 유지해주세요)
        # ----------------------------------------------------

# 3. 두 번째 탭: 가입설계 튜닝 영역
with tab_tuning:
    if st.session_state.get("show_tuning", False):
        ui.render_tuning_panel(mock_design_api_v2, generate_proposal_pdf)
    else:
        st.info("👈 왼쪽 사이드바에서 '입력 정보로 추천 받기'를 실행하면 가입설계 패널이 활성화됩니다.")

# ==========================================
# 💬 채팅 인터페이스 및 의도 라우팅 (비즈니스 핵심)
# ==========================================
raw_user_input = st.chat_input("채팅으로 대화하세요 (예: 30대 남성 고혈압인데 암보험 원해)")  
if raw_user_input:
    user_input = mask_pii(raw_user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): 
        st.markdown(user_input)

    with st.spinner("의도를 파악하고 있습니다..."):
        current_year = datetime.now().year
        
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
            if isinstance(content_obj, list):
                raw_res = content_obj[0].get("text", "")
            else:
                raw_res = str(content_obj)
            
            raw_res = raw_res.replace("```json", "").replace("```", "").strip()
            parsed_intent = json.loads(raw_res)
            
        except Exception as e:
            print(f"🚨 라우팅 파싱 에러 발생: {e}")
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
                    obj_prompt = PromptTemplate.from_template("...") # 거절 극복 프롬프트
                    obj_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | obj_prompt | llm | StrOutputParser())
                    response = obj_chain.invoke(user_input)
                    response = mask_pii(response)
                    st.markdown(response)
                except Exception as e:
                    response = "⚠️ 현재 이용자가 많아 AI 서버 통신이 지연되고 있습니다. 약 30초 후 다시 시도해 주세요."
                    st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # [의도 4] 일반 약관 Q&A
    else:
        with st.chat_message("assistant"):
            with st.spinner("약관을 확인 중입니다..."):
                try:
                    qa_prompt = PromptTemplate.from_template("...") # 일반 약관 Q&A 프롬프트
                    qa_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | qa_prompt | llm | StrOutputParser())
                    raw_response = qa_chain.invoke(user_input)
                    response = mask_rrn(raw_response) 
                    st.markdown(response)
                except Exception as e:
                    response = "⚠️ 현재 이용자가 많아 AI 서버 통신이 지연되고 있습니다. 약 30초 후 다시 질문해 주세요."
                    st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})