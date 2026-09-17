"""Command-line prediction; trains on the 72 local examples without an API key."""

import argparse
import json

from premium.model import PremiumPredictor, PremiumRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", type=int, required=True, help="보험나이 30~50세")
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--coverage", choices=["dementia", "disability"], default="dementia")
    parser.add_argument("--refund", choices=["standard", "partial"], default="standard")
    parser.add_argument("--payment-years", type=int, default=20)
    args = parser.parse_args()
    request = PremiumRequest(
        age=args.age, sex={"male": "남자", "female": "여자"}[args.sex],
        coverage_type={"dementia": "치매보장형", "disability": "생활비보장형(장해)"}[args.coverage],
        refund_type={"standard": "표준형", "partial": "해약환급금일부지급형"}[args.refund],
        payment_years=args.payment_years,
    )
    try:
        result = PremiumPredictor().predict(request)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
