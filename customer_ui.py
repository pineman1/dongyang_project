"""Presentation helpers for the customer lab; all data stays in customer_ml."""

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"


def apply_theme():
    st.html(ASSETS / "customer.css")


def sidebar_brand():
    st.html('''<div class="lab-brand"><span class="brand-symbol" aria-hidden="true">d/</span>
        <div>DONGYANG<span>DATA LAB</span></div></div>
        <div class="sidebar-caption">고객 데이터, 더 깊이 이해하기.</div>''')


def hero(data):
    st.html('''<div class="masthead"><span>WORKSPACE <b>/</b> CUSTOMER INTELLIGENCE</span>
        <span class="status-tag"><i></i> DEMO DATA</span></div>''')
    with st.container(key="hero"):
        words, art = st.columns([1.1, 1], gap="small", vertical_alignment="center")
        with words:
            st.html('<div class="eyebrow">EXPLORE THE PATTERNS</div>')
            st.html('<h1 class="hero-title">고객 조건별<br>상품·특약 추천</h1>')
            st.html('<p class="hero-description">고객의 조건을 연결하고,<br>데이터 속 가입 패턴을 발견하세요.</p>')
        with art:
            # st.html sanitizes inline SVG; an image keeps the artwork intact.
            mesh = b64encode((ASSETS / "data-mesh.svg").read_bytes()).decode("ascii")
            st.html(f'<div class="hero-art" aria-hidden="true"><img src="data:image/svg+xml;base64,{mesh}" alt=""><span>CONDITIONS → PATTERNS → INSIGHTS</span></div>')
    st.html(f'''<div class="dataset-strip" aria-label="학습 데이터 요약">
        <div><span class="stat-index">01 / CUSTOMERS</span><strong>{len(data):,}<small>명</small></strong><span>학습 고객</span></div>
        <div><span class="stat-index">02 / FEATURES</span><strong>09<small>개</small></strong><span>고객 입력 조건</span></div>
        <div><span class="stat-index">03 / PRODUCTS</span><strong>{data["가입상품"].nunique():02d}<small>종</small></strong><span>가입상품</span></div>
        <p>더미데이터로 탐색하는<br>보험 상품과 특약의 연결.</p></div>''')


def section_heading(number, title, subtitle=""):
    st.html(f'<div class="section-heading"><span>{escape(number)}</span><div><h2>{escape(title)}</h2>'
            f'<p>{escape(subtitle)}</p></div></div>')


def field_group(number, title):
    st.html(f'<div class="field-group"><span>{escape(number)}</span>{escape(title)}</div>')


def empty_result():
    with st.container(key="empty_result"):
        st.html('''<div class="empty-graphic" aria-hidden="true"><span></span><span></span><span></span><i>∅</i></div>
            <div class="empty-kicker">NO MATCHING DATA</div>''')
        st.info("자료가 없음")
        st.caption("입력한 9개 고객 조건이 모두 일치하는 학습 자료가 없습니다.")
        st.html('''<p class="empty-help">고객 조건을 변경하거나<br>‘학습 고객 예시 불러오기’로 탐색을 시작해 보세요.</p>
            <div class="empty-footnote"><span>확인 가능한 데이터가 있을 때</span><b>상품·특약 결과 표시 ↗</b></div>''')


def ranking_bars(ranking):
    rows = []
    for i, item in enumerate(ranking):
        score = float(item["score"])
        label = escape(item["label"])
        rows.append(f'''<div class="ranking-row"><div><span><small>{i + 1:02d}</small>{label}</span><b>{score:.1%}</b></div>
            <div class="ranking-track" role="meter" aria-label="{label} 모델 점수" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score * 100:.1f}">
            <span style="width:{score * 100:.3f}%"></span></div></div>''')
    st.html('<div class="ranking-list">' + "".join(rows) + '</div>')


def selection_summary(selected, total):
    st.html(f'''<div class="selection-summary"><div><span>선택한 고객</span><strong>{selected:,}<small>명</small></strong></div>
        <div><span>전체 고객 중</span><strong>{selected / total:.1%}</strong></div>
        <p>조건에 맞는 고객을 살펴보고<br>필요한 자료만 CSV로 저장하세요.</p></div>''')


def split_summary(training, test):
    percent = training / (training + test) * 100
    st.html(f'''<div class="split-summary"><div><span>TRAIN / 학습 <b>{training:,}명</b></span>
        <span>TEST / 검증 <b>{test:,}명</b></span></div><div class="split-track" aria-hidden="true"><span style="width:{percent:.3f}%"></span></div></div>''')


def footer():
    st.html('''<footer class="lab-footer"><span>DONGYANG / DATA LAB</span><span>고객 더미데이터 기반 · 분석용 데모</span><span>EXPLORE WITH CONTEXT ↗</span></footer>''')
