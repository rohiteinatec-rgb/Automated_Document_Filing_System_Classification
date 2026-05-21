import os
import json
import time
import csv
import asyncio
import sys
import re
from pathlib import Path
from collections import defaultdict
from main import DocumentAutoFiler

async def run_matrix_evaluation(test_folder: str, ground_truth_file: str, output_csv: str):
    print("\n🚀 Starting GESDOC AI Matrix Evaluation...")

    # 1. Load the Ground Truth
    if not os.path.exists(ground_truth_file):
        print(f"❌ Error: Ground truth file '{ground_truth_file}' not found.")
        print("Please create it first with your 10 test files.")
        return

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # 2. Initialize the AI in Dry-Run mode
    auto_filer = DocumentAutoFiler(debug=False, dry_run=True)

    results = []
    correct_tags = 0
    correct_companies = 0
    test_dir = Path(test_folder)

    # ---------------------------------------------------------
    # 🆕 UPDATE 2: Matrix Trackers setup
    # 'confusion_matrix' stores the exact number of guesses (e.g., expected factura, guessed uncertain)
    # 'unique_tags' remembers every type of document seen so it can build the table headers
    # ---------------------------------------------------------
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    unique_tags = set()

    pdfs_to_test = [p for p in test_dir.glob("*.pdf") if p.name in ground_truth]

    if not pdfs_to_test:
        print(f"❌ Error: No PDFs found in '{test_folder}' that match the names in your ground truth file.")
        return

    # 3. Process the files
    for pdf_file in pdfs_to_test:
        fname = pdf_file.name
        expected = ground_truth[fname]
        expected_tag = expected["expected_tag"]
        unique_tags.add(expected_tag)
        print(f"  [Testing] {fname}...")
        t0 = time.time()

        # Suppress stdout temporarily to keep the console clean (optional, but nice for reports)

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')

        try:
            actual = await auto_filer.process(str(pdf_file))
            actual = actual if isinstance(actual, dict) else {}
        except Exception as e:
            # If the pipeline crashes (e.g., corrupted PDF), catch it and force a safe failure
            actual = {"tag": "uncertain", "company": "unknown"}
            # We print the error to standard error so it shows up even though stdout is muted
            print(f"\n      ⚠️ Pipeline crashed on {fname}: {e}", file=sys.stderr)
        finally:
            sys.stdout = original_stdout # Restore console printing

        processing_time = round(time.time() - t0, 2)

        # Extract predictions
        actual_tag = actual.get("tag", "ERROR_OR_UNCERTAIN")
        actual_company = actual.get("company", "ERROR")

        # UPDATE 5: Log the result into the matrix trackers
        # ---------------------------------------------------------
        unique_tags.add(expected_tag)
        unique_tags.add(actual_tag)
        confusion_matrix[expected_tag][actual_tag] += 1

        # Helper to normalize text: lowercase, swap underscores for spaces, remove punctuation
        def normalize_text(text):
            return re.sub(r'[^a-z0-9]', '', text.lower())

        # Check matches
        tag_match = (actual_tag == expected["expected_tag"])
        company_match = (normalize_text(actual_company) == normalize_text(expected["expected_company"]))

        if tag_match: correct_tags += 1
        if company_match: correct_companies += 1

        results.append({
            "File": fname,
            "Expected Tag": expected["expected_tag"],
            "AI Tag": actual_tag,
            "Tag Match": "✅" if tag_match else "❌",
            "Expected Company": expected["expected_company"],
            "AI Company": actual_company,
            "Company Match": "✅" if company_match else "❌",
            "Time (s)": processing_time
        })

    # 4. Generate Terminal Report
    total_files = len(results)
    tag_accuracy = (correct_tags / total_files) * 100
    comp_accuracy = (correct_companies / total_files) * 100

    print("\n" + "="*80)
    print("📊 AI TEST RESULTS REPORT")
    print("="*80)
    print(f"  Total Files Tested : {total_files}")
    print(f"  Tag Accuracy       : {tag_accuracy:.1f}%")
    print(f"  Company Accuracy   : {comp_accuracy:.1f}%")
    print("-" *80)

    tags = sorted(list(unique_tags))

    print("\n" + "="*80)
    print("🧩 FULL CONFUSION MATRIX (Performance Overview)")
    print("="*80)

    header = f"{'Actual \\ Predicted':<20} | " + " | ".join([f"{t:<14}" for t in tags]) + " | TOTAL ROW"
    print(header)
    print("-" * len(header))

    for actual_t in tags:
        row_str = f"{actual_t:<20} | "
        row_total = 0
        for pred_t in tags:
            count = confusion_matrix[actual_t][pred_t]
            row_total += count
            display_count = f"[{count}]" if actual_t == pred_t else str(count)
            row_str += f"{display_count:<14} | "
        row_str += f"{row_total}"
        print(row_str)

    print("\n" + "="*80)
    print("🚨 ERROR MATRIX (Misclassifications Only)")
    print("="*80)
    print(header.replace("TOTAL ROW", "ERR TOTAL"))
    print("-" * len(header))

    for actual_t in tags:
        row_str = f"{actual_t:<20} | "
        row_err_total = 0
        for pred_t in tags:
            if actual_t == pred_t:
                count = 0  # Zero out the correct answers so only errors show
            else:
                count = confusion_matrix[actual_t][pred_t]
            row_err_total += count
            row_str += f"{count:<14} | "
        row_str += f"{row_err_total}"
        print(row_str)

    print("\n" + "-" * 80)

    for r in results:
        status = "✅" if r["Tag Match"] == "✅" else "❌"
        print(f"  {status} {r['File']}")
        if r["Tag Match"] == "❌":
            print(f"      Tag Alert -> Expected: '{r['Expected Tag']}' | AI Guessed: '{r['AI Tag']}'")
        if r["Company Match"] == "❌":
            print(f"      Company Alert -> Expected: '{r['Expected Company']}' | AI Guessed: '{r['AI Company']}'")

    # 5. Export to CSV for the Stakeholders
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["File", "Expected Tag", "AI Tag", "Tag Match", "Expected Company", "AI Company", "Company Match", "Time (s)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("="*60)
    print(f"💾 Detailed report saved to: {output_csv}")

if __name__ == "__main__":
    # Make sure these match your actual folder and file names!
    asyncio.run(run_matrix_evaluation(
        test_folder="./Stress_Test_101",
        ground_truth_file="./Stress_Test_101/ground_truth.json",
        output_csv="./evaluation_report.csv"
        )
    )