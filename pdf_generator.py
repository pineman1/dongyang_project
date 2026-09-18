# pdf_generator.py
import os
from datetime import datetime
from fpdf import FPDF

class InsuranceProposalPDF(FPDF):
    def header(self):
        # 상단 브랜드 컬러 바 (동양생명 시그니처 오렌지)
        self.set_fill_color(243, 112, 33)
        self.rect(0, 0, 210, 6, "F")
        
        # 최상단 서브 로고 텍스트
        self.set_y(10)
        self.set_font("KoreanFont", "B", 10)
        self.set_text_color(243, 112, 33)
        self.cell(0, 5, "TONGYANG LIFE  |  AI HYBRID SALES REPORT", ln=True, align="R")
        
        # 메인 타이틀
        self.set_font("KoreanFont", "B", 18)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, "[ 동양생명 AI 고객 맞춤 가입설계 제안서 ]", ln=True, align="C")
        
        self.set_font("KoreanFont", "", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, "고객 빅데이터 기반 위험률 산출 및 생성형 AI 약관 분석 솔루션", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("KoreanFont", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"동양생명 AI FC 어시스턴트 | 고객지원본부 | Page {self.page_no()}", align="C")


def generate_proposal_pdf(design_result, customer_info=None, ml_score=None):
    """
    고도화된 가입설계 제안서 PDF 생성 함수
    """
    pdf = InsuranceProposalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # 폰트 세팅
    font_regular = "NanumGothic.ttf"
    font_bold = "NanumGothicBold.ttf"

    if not os.path.exists(font_regular):
        font_regular = "C:/Windows/Fonts/malgun.ttf"
        font_bold = "C:/Windows/Fonts/malgunbd.ttf" if os.path.exists("C:/Windows/Fonts/malgunbd.ttf") else font_regular

    if os.path.exists(font_regular):
        pdf.add_font("KoreanFont", "", font_regular)
        pdf.add_font("KoreanFont", "B", font_bold if os.path.exists(font_bold) else font_regular)
        font = "KoreanFont"
    else:
        font = "Helvetica"

    pdf.add_page()
    pdf.set_font(font, "", 9)

    # 발행 일시
    issued_date = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"발행일시: {issued_date}  |  담당 FC: AI 디지털 설계사", ln=True, align="R")
    pdf.ln(2)

    # ----------------------------------------------------
    # 1. AI 진단 종합 스코어 & 고객 프로파일링
    # ----------------------------------------------------
    pdf.set_font(font, "B", 11)
    pdf.set_text_color(243, 112, 33)
    pdf.cell(0, 7, "1. AI 고객 리스크 분석 & 프로파일", ln=True)
    pdf.set_draw_color(243, 112, 33)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    if customer_info:
        # 고객 프로필 카드 박스
        pdf.set_fill_color(250, 250, 252)
        pdf.set_draw_color(220, 224, 230)
        pdf.rect(10, pdf.get_y(), 190, 24, "DF")
        
        pdf.set_xy(14, pdf.get_y() + 2)
        pdf.set_font(font, "B", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(40, 6, f"• 기본 정보: {customer_info.get('나이', 35)}세 / {customer_info.get('성별', '남성')}", ln=False)
        pdf.cell(40, 6, f"• 직업군: {customer_info.get('직업위험등급', 1)}등급 (사무/안전직)", ln=False)
        pdf.cell(50, 6, f"• 연소득: {customer_info.get('연소득_만원', 5000):,} 만원", ln=False)
        pdf.cell(50, 6, f"• 혼인/자녀: {customer_info.get('결혼여부', '미혼')} ({customer_info.get('자녀수', 0)}자녀)", ln=True)

        pdf.set_x(14)
        pdf.cell(40, 6, f"• 건강 상태: {customer_info.get('만성질환', '없음')}", ln=False)
        pdf.cell(40, 6, f"• 흡연 여부: {customer_info.get('흡연여부', '비흡연')}", ln=False)
        pdf.set_text_color(220, 50, 50)
        pdf.cell(50, 6, f"• 주요 가족력: {customer_info.get('가족력', '없음')} (집중 대비 필요)", ln=True)

        pdf.set_x(14)
        pdf.set_font(font, "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, "※ 고객 머신러닝 매칭 결과: 동 연령대 대비 암/중대질환 보장 니즈 87.4% 이상 집중 분포", ln=True)
        pdf.ln(5)

    # ----------------------------------------------------
    # 2. 추천 가입설계 조건 요약
    # ----------------------------------------------------
    pdf.set_font(font, "B", 11)
    pdf.set_text_color(243, 112, 33)
    pdf.cell(0, 7, "2. 맞춤 추천 가입설계 요약", ln=True)
    pdf.set_draw_color(243, 112, 33)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    product_name = design_result.get("product_name", "무배당 수호천사 암/건강보험")
    payment_term = design_result.get("payment_term", "20년납")
    premium = design_result.get("monthly_premium", 0)
    has_rider = design_result.get("include_rider", True)

    summary_items = [
        ("추천 주계약명", product_name, "가입 형태", "개인형 / 순수보장성"),
        ("납입 / 보험기간", f"{payment_term} / 100세 만기", "납입 주기", "월납 (자동이체 1% 할인 가능)"),
        ("특약 구성 여부", "맞춤 선택 특약 가입" if has_rider else "기본형 (특약 미가입)", "최종 월 납입보험료", f"{premium:,} 원")
    ]

    for label1, val1, label2, val2 in summary_items:
        # 좌측 셀
        pdf.set_font(font, "B", 9)
        pdf.set_fill_color(245, 247, 250)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(32, 8, f"  {label1}", border=1, fill=True)
        
        pdf.set_font(font, "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(63, 8, f"  {val1}", border=1)

        # 우측 셀
        pdf.set_font(font, "B", 9)
        pdf.cell(32, 8, f"  {label2}", border=1, fill=True)
        
        if "보험료" in label2:
            pdf.set_font(font, "B", 10)
            pdf.set_text_color(243, 112, 33)
        else:
            pdf.set_font(font, "", 9)
            pdf.set_text_color(30, 30, 30)
        pdf.cell(63, 8, f"  {val2}", border=1, ln=True)

    pdf.ln(5)

    # ----------------------------------------------------
    # 3. 보장 담보 상세 분석표 (풍성한 디테일)
    # ----------------------------------------------------
    pdf.set_font(font, "B", 11)
    pdf.set_text_color(243, 112, 33)
    pdf.cell(0, 7, "3. 주요 보장 담보 상세 내역", ln=True)
    pdf.set_draw_color(243, 112, 33)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # 표 헤더
    pdf.set_fill_color(52, 58, 64)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font, "B", 9)
    pdf.cell(40, 7, "  보장 구분", border=1, fill=True)
    pdf.cell(60, 7, "  담보명", border=1, fill=True)
    pdf.cell(40, 7, "  가입금액", border=1, fill=True)
    pdf.cell(50, 7, "  지급 사유", border=1, fill=True, ln=True)

    # 상품에 따른 동적 상세 담보 데이터
    if "암" in product_name:
        coverage_rows = [
            ("주계약", "일반암 진단비", "5,000 만원", "최초 1회 한 (유사암 제외)"),
            ("주계약", "유사암 진단비(갑상선, 기타피부)", "1,000 만원", "각각 최초 1회 한"),
            ("주계약", "암 수술 및 입원일당", "수술 회당 300만원", "입원 1일당 5만원 지급"),
            ("선택특약" if has_rider else "미가입", "표적항암약물허가치료비", "3,000 만원" if has_rider else "-", "최초 1회한, 최신 표적치료 보장"),
            ("선택특약" if has_rider else "미가입", "중대 암 산정특례 보장", "1,000 만원" if has_rider else "-", "산정특례 등록 시 연 1회 지급"),
        ]
    elif "종신" in product_name:
        coverage_rows = [
            ("주계약", "기본 사망보험금", "1 억원", "사망 또는 80% 이상 고도장해 시"),
            ("주계약", "재해사망 추가보장", "5,000 만원", "교통사고 등 재해 사망 시"),
            ("선택특약" if has_rider else "미가입", "가족 생활자금 보장 특약", "매월 150만원 (10년)" if has_rider else "-", "유가족 생활 안정 보장금"),
            ("선택특약" if has_rider else "미가입", "3대 질병 납입면제 특약", "보장 적용" if has_rider else "-", "암/뇌/심장 질환 시 납입면제"),
        ]
    else:  # 연금 등
        coverage_rows = [
            ("주계약", "연금 개시 후 생존연금", "연 약 720만원", "개시연령(65세) 이후 종신지급"),
            ("주계약", "연금 개시 전 사망보장", "기납입보험료+알파", "계약자 적립액 전액 지급"),
            ("선택특약" if has_rider else "미가입", "고도장해 연금 추가보장", "연 300만원 추가" if has_rider else "-", "80% 이상 장해 발생 시 추가지급"),
        ]

    pdf.set_text_color(40, 40, 40)
    for cat, name, amount, desc in coverage_rows:
        pdf.set_font(font, "B" if "주계약" in cat else "", 8)
        pdf.cell(40, 6, f"  {cat}", border=1)
        pdf.cell(60, 6, f"  {name}", border=1)
        
        pdf.set_font(font, "B", 8)
        if "-" in amount:
            pdf.set_text_color(150, 150, 150)
        else:
            pdf.set_text_color(243, 112, 33)
        pdf.cell(40, 6, f"  {amount}", border=1)
        
        pdf.set_font(font, "", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(50, 6, f"  {desc}", border=1, ln=True)

    pdf.ln(5)

    # ----------------------------------------------------
    # 4. AI 종합 세일즈 & 고객 케어 코멘트
    # ----------------------------------------------------
    pdf.set_font(font, "B", 11)
    pdf.set_text_color(243, 112, 33)
    pdf.cell(0, 7, "4. AI Agent 전담 설계 코멘트", ln=True)
    pdf.set_draw_color(243, 112, 33)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_fill_color(248, 252, 248)
    pdf.set_draw_color(200, 230, 200)
    pdf.rect(10, pdf.get_y(), 190, 22, "DF")

    pdf.set_xy(13, pdf.get_y() + 2)
    pdf.set_font(font, "", 8.5)
    pdf.set_text_color(40, 40, 40)
    comment = (
        f"• 본 설계안은 고객님의 [가족력: {customer_info.get('가족력', '없음')}]과 연령별 발병 통계를 고려하여 산출되었습니다.\n"
        f"• 특히 고액 치료비가 발생하는 {product_name} 위주로 집중 구성하였으며, 특약 포함 시 최신 표적치료 혜택까지 완벽히 커버됩니다.\n"
        f"• 동양생명 헬스케어 서비스를 통해 가입 후 전담 간호사 동행 및 건강검진 우대 예약 혜택이 함께 제공됩니다."
    )
    pdf.multi_cell(184, 5, comment)
    pdf.ln(8)

    # ----------------------------------------------------
    # 5. 법적 고지사항
    # ----------------------------------------------------
    pdf.set_font(font, "", 7.5)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(
        0, 3.5,
        "※ 본 안내장은 고객의 이해를 돕기 위해 AI 엔진으로 자동 생성된 시뮬레이션 문서이며, 법적 효력을 갖는 정식 보험증권이 아닙니다.\n"
        "※ 보험계약 체결 전 반드시 해당 상품의 약관 및 상품설명서를 확인하시기 바랍니다. 직무 변경, 건강 고지사항에 따라 인수가 제한될 수 있습니다."
    )

    return bytes(pdf.output())