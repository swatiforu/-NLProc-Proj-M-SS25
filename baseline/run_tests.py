import json
import csv
from run_pipeline import run_pipeline

with open("test_inputs.json") as f:
    test_cases = json.load(f)

results = []

for case in test_cases:
    question = case["question"]
    expected_keywords = case["expected_answer_keywords"]
    answer = run_pipeline(question)

    grounded = any(kw.lower() in answer.lower() for kw in expected_keywords)
    results.append({
        "question": question,
        "answer": answer,
        "grounded_in_context": grounded
    })

# Save to CSV
with open("logs/test_results.csv", "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["question", "answer", "grounded_in_context"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Tests completed. Results saved in logs/test_results.csv")
