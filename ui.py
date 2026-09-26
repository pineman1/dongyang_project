import streamlit as st

def apply_custom_theme():
    """
    동양생명 브랜드 컬러(시안/하늘색) 강제 적용 CSS
    """
    st.markdown(
        """
        
        """,
        unsafe_allow_html=True
    )


def render_login_screen(check_login_fn):
    """ 로그인 화면 렌더링 """
    apply_custom_theme()
    st.title("💼 동양생명 AI FC 어시스턴트 로그인")
    
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submit = st.form_submit_button("로그인", use_container_width=True)
        
        if submit:
            if check_login_fn(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")


def render_sidebar(required_keys, process_rec_fn, retriever, llm):
    """ 사이드바 렌더링 (고객 정보 폼) """
    apply_custom_theme()
    
    with st.sidebar:
        st.title("👤 FC 전용 메뉴")
        if "username" in st.session_state:
            st.write(f"접속자: **{st.session_state.username}** FC님")
            if st.button("로그아웃"):
                st.session_state.logged_in = False
                st.rerun()
        
        st.divider()
        st.header("📋 고객 정보 수동 입력")
        
        with st.form("customer_input_form"):
            age = st.number_input("나이", min_value=0, max_value=100, value=40)
            gender = st.selectbox("성별", ["남성", "여성"])
            income = st.number_input("연소득(만원)", min_value=0, value=4000)
            job = st.selectbox("직업위험등급", [1, 2, 3])
            married = st.selectbox("결혼여부", ["미혼", "기혼"])
            children = st.number_input("자녀수", min_value=0, value=0)
            smoking = st.selectbox("흡연여부", ["비흡연", "흡연"])
            disease = st.selectbox("만성질환", ["없음", "고혈압", "당뇨"])
            family_history = st.selectbox("가족력", ["없음", "암", "심혈관"])
            preferred = st.selectbox("선호상품 (선택)", ["선택안함", "암보험", "종신보험", "연금보험", "유병자보험"])
            
            submit_btn = st.form_submit_button("입력 정보로 추천 받기", use_container_width=True)
            
            if submit_btn:
                info = {
                    "나이": age, "성별": gender, "연소득_만원": income,
                    "직업위험등급": job, "결혼여부": married, "자녀수": children,
                    "흡연여부": smoking, "만성질환": disease, "가족력": family_history
                }
                if preferred != "선택안함":
                    info["선호상품"] = preferred
                
                st.session_state.collected_info = info
                process_rec_fn(info, retriever, llm)


def render_tuning_panel(mock_design_api_fn, generate_pdf_fn):
    """ 실시간 가입설계 튜닝 패널 """
    with st.expander("📊 실시간 조율 패널", expanded=True):
        st.subheader("💡 가입 조건 실시간 조정")
        col1, col2 = st.columns(2)
        with col1:
            product_name = st.text_input("상품명", value=st.session_state.get("last_recommended_product", "무배당 수호천사 암/건강보험"))
            include_rider = st.checkbox("추천 특약 포함", value=True)
        with col2:
            payment_term = st.selectbox("납입 기간", ["10년납", "20년납", "30년납"], index=1)
            
        cust_info = st.session_state.get("last_customer_info", {"나이": 40, "만성질환": "없음", "흡연여부": "비흡연"})
        design_result = mock_design_api_fn(cust_info, product_name, include_rider, payment_term)
        
        st.markdown("---")
        st.markdown(f"### 💵 예상 월 납입 보험료: **{design_result['monthly_premium']:,} 원**")
        st.write(f"- 보장 내용 1: {design_result['coverage_1']}")
        st.write(f"- 보장 내용 2: {design_result['coverage_2']}")
        
        pdf_bytes = generate_pdf_fn(cust_info, design_result)
        st.download_button(
            label="📄 가입설계서 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"동양생명_가입설계서_{cust_info.get('나이', 40)}세.pdf",
            mime="application/pdf",
            use_container_width=True
        )


def render_chat_history():
    """ 대화내용 렌더링 """
    apply_custom_theme()
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def render_quick_actions():
    """
    채팅창 위에 자주 묻는 질문(Quick Actions) 버튼 렌더링
    버튼을 클릭하면 해당 질문 텍스트를 반환합니다.
    """
    st.write("💡 **빠른 질문 추천**")
    cols = st.columns(3)
    quick_queries = [
        "📝 수호천사 암보험 가입 조건",
        "🔍 고혈압 환자 인수 지침",
        "📄 최근 개정된 약관 요약"
    ]
    
    # 클릭된 버튼의 텍스트를 반환
    for i, col in enumerate(cols):
        if col.button(quick_queries[i], use_container_width=True):
            return quick_queries[i]
    return None