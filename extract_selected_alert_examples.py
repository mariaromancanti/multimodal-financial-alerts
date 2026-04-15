import argparse
import json
import re
import unicodedata
from pathlib import Path


FIRST_10_NEWS = [
    "17-00217-UT to amend and extend the Company 's Revolving Credit Facility (\" RCF \"), issue up to $ 350.0 million in long-term debt and to redeem and refinance the $ 63.5 million 2009 Series A 7.25% Pollution Control Bonds (\" PCBs \") and the $ 37.1 million 2009 Series B 7.25% PCBs.",
    "2 increased the Company's borrowing under Term Loan A from $ 608 million to $ 1 billion and increased the Company's borrowing capacity under the Revolving Credit Facility from $ 800 million to $ 1 billion.",
    "Furthermore, with respect to the 2020 Delayed-Draw Term Loan, the applicable margin (as determined based on our long-term debt credit rating) for any LIBOR rate loans will increase from a range of 75.0 and 97.5 basis points to a range of 155.0 and 180.0 basis points and for any base rate loans from 0.0 to a range of 55.0 and 80.0 basis points.",
    "In May 2017, the Company amended the credit agreement governing its unsecured revolving credit facility (the \" Credit Facility \") to increase the maximum borrowings from $ 1.8 billion to $ 2.0 billion and extend the maturity on $ 1.4 billion of the Credit Facility from June 2020 to June 2022, with $ 160 million maturing in June 2018 and the remaining $ 50 million maturing in June 2020.",
    "Facility being increased from $ 285.0 million to $ 313.5 million and (y) the ABL Canadian Facility being increased from $ 75.0 million to $ 82.5 million. On November 16, 2018, Ply Gem Midco entered into an incremental asset-based revolving credit facility of $ 215.0 million in connection with the Merger, which upsized the Current ABL Facility to $ 611.0 million in the aggregate, and with (x) the ABL U.S.",
    "In addition, with respect to the Revolving Credit Facility and for any Adjusted Four Quarters in which the consolidated net leverage ratio is greater than the Unadjusted Maximum Ratio, the applicable margin (as determined based on our long-term debt credit rating) for any LIBOR rate loans will increase from a range of 80.5 and 117.5 basis points to a range of 118.0 and 155.0 basis points and for any base rate loans from a range of 0.0 and 17.5 basis points to a range of 18.0 and 55.0 basis points.",
    "The total borrowing limit under the amended credit agreements was increased to $ 3.1 billion, with the following changes : Maturity extended from June 2021 to June 2024. Borrowing limit for Xcel Energy was increased from $ 1.0 billion to $ 1.25 billion. Borrowing limit for SPS was increased from $ 400 million to $ 500 million. Added swingline subfacility for Xcel Energy up to $ 75 million. Xcel Energy Inc., NSP-Minnesota, PSCo, and SPS each have the right to request an extension of the revolving credit facility termination date for two additional one year periods.",
    "The amended credit agreements have substantially the same terms and conditions as the prior credit agreements with the following exceptions : Maturity extended from June 2021 to June 2024. Borrowing limit for Xcel Energy was increased from $ 1.0 billion to $ 1.25 billion. Borrowing limit for SPS was increased from $ 400 million to $ 500 million. Added swingline subfacility for Xcel Energy up to $ 75 million. Xcel Energy Inc., NSP-Minnesota, PSCo, and SPS each have the right to request an extension of the revolving credit facility termination date for two additional one year periods.",
    "On December 31, 2014, the Company amended its Prior Credit Facilities to (i) increase the amount of the revolving credit facility in the Original Credit Agreement to $ 350.0 million from $ 200.0 million, (ii) increase the amount of the letter of credit subfacility in the Original Credit Agreement to $ 50.0 million from $ 25.0 million, (iii) eliminate the swing line subfacility in the amount of up to $ 35.0 million in the Original Credit Agreement, and (iv) reduce to $ 100.0 million from $ 250.0 million the amount of additional incremental revolving or term loans in the Original Credit Agreement.",
    "91 Table of Contents LENNAR CORPORATION AND SUBSIDIARIES NOTES TO CONSOLIDATED FINANCIAL STATEMENTS-(Continued) The carrying amounts of the senior notes listed above are net of debt issuance costs of $ 31.2 million and $ 33.5 million, as of November 30, 2018 and 2017, respectively. In February 2018, the Company amended the credit agreement governing its unsecured revolving credit facility (the \" Credit Facility \") to increase the maximum borrowings from $ 2.0 billion to $ 2.6 billion and extended the maturity on $ 2.2 billion of the Credit Facility from June 2022 to April 2023, with $ 70 million maturing in June 2018 and the remaining $ 50 million maturing in June 2020.",
]


TRAIN_NEGATIVE = [
    "We have incurred pre-tax expenses totaling $ 40,708 related to these restructuring actions, of which $ 30,987 was recorded as restructuring expenses and $ 9,721 was recorded in cost of revenues, with a total of $ 28,417, $ 2,518, $ 664, and $ 7,794 related to the Healthcare Products, Healthcare Specialty Services, Life Sciences, and Applied Sterilization Technologies segments, respectively.",
    "Since inception of the Fiscal 2019 Restructuring Plan we have incurred pre-tax expenses totaling $ 43,015 related to these restructuring actions, of which $ 32,376 was recorded as restructuring expenses and $ 10,639 was recorded in cost of revenues, with a total of $ 30,713, $ 2,518, $ 668 and $ 7,798 related to the Healthcare Products, Healthcare Specialty Services, Life Sciences, and Applied Sterilization Technologies segments, respectively.",
    "Since inception of the Fiscal 2019 Restructuring Plan we have incurred pre-tax expenses totaling $ 43,651 related to these restructuring actions, of which $ 32,102 was recorded as restructuring expenses and $ 11,549 was recorded in cost of revenues, with a total of $ 31,116, $ 2,518, $ 668 and $ 7,798 related to the Healthcare Products, Healthcare Specialty Services, Life Sciences, and Applied Sterilization Technologies segments, respectively.",
    "Since inception of the Fiscal 2019 Restructuring Plan we have incurred pre-tax expenses totaling $ 44,036 related to these restructuring actions, of which $ 31,654 was recorded as restructuring expenses and $ 12,382 was recorded in cost of revenues, with a total of $ 31,278, $ 2,518, $ 668 and $ 7,798 related to the Healthcare Products, Healthcare Specialty Services, Life Sciences, and Applied Sterilization Technologies segments, respectively.",
    "The following tables provide segment reporting of the Company for the three and six months ended May 31, 2018 and 2019, respectively (in thousands): Intersegment revenues were approximately $ 0.5 million and $ 0.6 million for the three months ended May 31, 2018 and 2019, respectively, and approximately $ 0.9 million and $ 0.9 million for the six months ended May 31, 2018 and 2019, respectively. During the three and six months ended May 31, 2019, revenues in the Motorsports Event segment included approximately $ 5.0 million and $ 6.9 million, respectively related to Racing Electronics, for which there was no comparable activity in the same periods of the prior year. During the three and six months ended May 31, 2019, revenues in the All Other segment have decreased by approximately $ 5.3 million and $ 4.8 million, respectively, as compared to the same periods in the prior year.",
    "The impairment charge of $ 428 recorded in the second quarter of 2019 impacted properties, plant and equipment; intangible assets; and certain other noncurrent assets by $ 198, $ 197 and $ 33, respectively.",
    "For the nine months ended September 30, 2019, the charge of $ 77.4 included $ 18.5 for termination of take-or-pay supply agreements, $ 20.2 for supplemental unemployment and other employee benefit costs, pension and OPEB termination benefits of $ 13.3, an estimated multiemployer plan withdrawal liability of $ 18.0, and $ 7.4 for other costs.",
    "During the first nine months of 2018, we closed 129 Core U.S. stores and 9 locations in Mexico, resulting in pre-tax charges of $ 10.5 million, consisting of $ 7.9 million in lease obligation costs, $ 1.5 million in disposal of fixed assets, $ 0.9 million in other miscellaneous shutdown and holding costs, and $ 0.2 million in severance and other payroll-related costs.",
    "In connection with the redemption, the Company will record a loss on extinguishment of debt of approximately $ 65,000 in the fourth quarter of 2019, representing the unamortized discount and deferred financing costs as of the redemption date.",
    "Income Tax The effective tax rate for the three and nine months ended September 30, 2019 was 162.3% and 278.3%, respectively, compared to 16.4% and (12.4)% for the same periods in 2018.",
]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "under",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"([,.;:()\"%])", r" \1 ", text)
    text = re.sub(r"(?<=\w)'s\b", " 's", text)
    text = re.sub(r"(?<=\w)'(?=\w)", " ' ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " - ", text)
    text = re.sub(r"^\s*\d+\.\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def significant_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9.,%-]*", normalize_text(text))
    return {token for token in tokens if token not in STOPWORDS}


def find_best_match(query: str, processed_examples: list[dict]) -> tuple[int | None, str]:
    normalized_query = normalize_text(query)
    query_tokens = significant_tokens(query)

    for idx, example in enumerate(processed_examples):
        normalized_text = normalize_text(example["text"])
        if normalized_text == normalized_query:
            return idx, "exact"

        if normalized_query in normalized_text or normalized_text in normalized_query:
            return idx, "contains"

    best_idx = None
    best_score = 0.0

    for idx, example in enumerate(processed_examples):
        example_tokens = significant_tokens(example["text"])
        if not query_tokens:
            continue

        overlap = len(query_tokens & example_tokens) / len(query_tokens)
        if overlap > best_score:
            best_score = overlap
            best_idx = idx

    if best_idx is not None and best_score >= 0.75:
        return best_idx, "token_overlap"

    return None, "not_found"


def build_queries() -> list[dict]:
    queries = []

    for idx, text in enumerate(FIRST_10_NEWS, start=1):
        queries.append(
            {
                "group": "FIRST_10",
                "index": idx,
                "query_text": text,
            }
        )

    for idx, text in enumerate(TRAIN_NEGATIVE, start=1):
        queries.append(
            {
                "group": "TRAIN_NEGATIVE",
                "index": idx,
                "query_text": text,
            }
        )

    return queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected full-news examples from alert_train_processed.json."
    )
    parser.add_argument(
        "--input",
        default="final-project-nlp-dl-opositores/data/alert_train_processed.json",
        help="Path to alert_train_processed.json",
    )
    parser.add_argument(
        "--output",
        default="selected_alert_train_examples.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        processed_examples = json.load(f)

    queries = build_queries()
    results = []
    missing = []

    for query in queries:
        match_idx, match_type = find_best_match(query["query_text"], processed_examples)

        if match_idx is None:
            missing.append(query)
            results.append(
                {
                    "text": None,
                    "ner_output": None,
                    "sa_output": None,
                }
            )
            continue

        example = processed_examples[match_idx]
        results.append(
            {
                "text": example["text"],
                "ner_output": example["ner_output"],
                "sa_output": example["sa_output"],
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} examples to {output_path}")
    print(f"Matched: {len(results) - len(missing)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nExamples not found:")
        for item in missing:
            print(f"- {item['group']} #{item['index']}")


if __name__ == "__main__":
    main()
