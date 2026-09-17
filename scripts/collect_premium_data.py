"""Reproduce 72 factual premiums from the visually verified, hash-pinned PDF.

Run: python scripts/collect_premium_data.py [--pdf local.pdf]
The original PDF stays local and is not redistributed in Git.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from premium.data import DATA_PATH, PRODUCT, SOURCE_ID, SOURCE_PATH, VERSION, validate_data

URL = "https://image.kebhana.com/cont/download/insdocument/leaflet/L74B14160_lf.pdf"
SHA256 = "b2a6231242aac91d5f4a8f3d8637827b78f8202b9bc9628d82e416d83bc60467"


def extract(pdf_bytes: bytes) -> pd.DataFrame:
    if hashlib.sha256(pdf_bytes).hexdigest() != SHA256:
        raise ValueError("검증한 원문과 PDF가 다릅니다. 개정 여부와 표를 검토한 뒤 파서를 갱신하세요.")
    records = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if "2023.9 개정상품" not in document[0].get_text():
            raise ValueError("상품 개정일 불일치")
        for table in document[2].find_tables():
            cells = table.extract()
            if cells[0][:2] != ["성별", "연령"]:
                continue
            terms = [int(value.removesuffix("년납")) for value in cells[1][2:]]
            if terms == [5, 7, 20]:
                coverage = "치매보장형"
            elif terms == [7, 10, 15]:
                coverage = "생활비보장형(장해)"
            else:
                raise ValueError("예상하지 못한 납입기간")
            refund = cells[0][2]
            sex = None
            for row in cells[2:]:
                sex = row[0] or sex
                age = int(row[1].removesuffix("세"))
                for term, amount in zip(terms, row[2:], strict=True):
                    records.append({
                        "record_id": f"ansim-{len(records) + 1:03d}",
                        "product_name": PRODUCT, "product_version": VERSION,
                        "age": age, "sex": sex, "coverage_type": coverage,
                        "refund_type": refund, "payment_years": term,
                        "sum_assured_krw": 10_000_000, "coverage_end_age": 110,
                        "payment_frequency": "monthly",
                        "monthly_premium_krw": int(amount.replace(",", "")),
                        "source_id": SOURCE_ID, "source_pdf_page": 3,
                    })
    return validate_data(pd.DataFrame(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="다운로드 없이 검증할 원본 PDF")
    args = parser.parse_args()
    if args.pdf:
        pdf_bytes = args.pdf.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=30) as response:
            pdf_bytes = response.read()
    data = extract(pdf_bytes)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Fully validate before replacing either output.
    data.to_csv(DATA_PATH, index=False, encoding="utf-8", lineterminator="\n")
    metadata = {
        "source_id": SOURCE_ID, "publisher": "동양생명", "host": "하나은행",
        "source_url": URL, "source_pdf_sha256": SHA256,
        "product_name": PRODUCT, "product_version": VERSION,
        "document_produced": "2023-08", "pdf_page": 3, "printed_pages": [4, 5],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "rows": len(data), "data_kind": "public_premium_examples",
        "synthetic_rows": 0, "customer_records": 0,
        "extraction": "PyMuPDF table extraction; original page visually verified",
        "limitations": [
            "2023년 9월 개정 상품의 공개 예시이며 현재 판매 보험료가 아닙니다.",
            "보험가입금액 1,000만원, 110세 만기, 월납, 주계약 기준입니다.",
            "30/40/50세만 관측되었으며 개별 고객의 심사 결과나 할인은 없습니다.",
            "표준형과 해약환급금일부지급형을 구분해야 합니다.",
            "실제 계약/청구 내역, 특약별 보험료, 질병/흡연 할증 데이터가 아닙니다.",
        ],
    }
    SOURCE_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified and wrote {len(data)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
