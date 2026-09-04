import json
import pickle
import time
from typing import Dict, List, Tuple
from answer import process_question, load_index
from tqdm import tqdm

# Load the full document texts
def load_full_texts():
    with open("index/full_texts.pkl", "rb") as f:
        return pickle.load(f)

# Check if a citation's quote exists verbatim in the cited document
def verify_citation(citation: Dict, full_texts: Dict) -> Tuple[bool, str]:
    doc_id = citation.get("doc_id")
    quote = citation.get("quote", "").strip()
    page = citation.get("page")
    
    # Check if doc exists
    if doc_id not in full_texts:
        return False, f"Document {doc_id} not found"
    
    # Check for empty quote
    if not quote:
        return False, "Empty quote"
    
    # Check if page number exists
    if not page:
        return False, "Missing page number"
    
    # Check if quote exists verbatim
    if quote not in full_texts[doc_id]:
        return False, f"Quote not found verbatim in {doc_id}"
    
    return True, "Valid"

# Evaluate a single question
def evaluate_question(q, index, metadata, full_texts, model):
    qid = q["id"]
    question = q["question"]
    qtype = q.get("type", "unknown")
    
    # Process the question
    result = process_question(question, qid, index, metadata, full_texts, model)
    
    # Verify citations
    valid_citations = []
    for citation in result.get("citations", []):
        is_valid, msg = verify_citation(citation, full_texts)
        if is_valid:
            valid_citations.append(citation)
    
    # Determine if the answer is supported
    is_supported = len(valid_citations) > 0
    abstained = result.get("abstained", True)
    
    return {
        "id": qid,
        "type": qtype,
        "question": question,
        "answer": result.get("answer", "I don't know"),
        "abstained": abstained,
        "has_valid_citation": is_supported,
        "total_citations": len(result.get("citations", [])),
        "valid_citations": len(valid_citations),
        "citations": result.get("citations", []),
        "valid_citations_list": valid_citations
    }

# Process questions in batches to show progress
def evaluate_batch(questions, batch_size=5):
    # Load the index once
    print("Loading index and model...")
    index, metadata, full_texts, model = load_index()
    
    results = []
    
    # Process in batches
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(questions)+batch_size-1)//batch_size}")
        
        batch_results = []
        for q in tqdm(batch, desc=f"Questions {i+1}-{min(i+batch_size, len(questions))}"):
            try:
                result = evaluate_question(q, index, metadata, full_texts, model)
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing {q['id']}: {e}")
                # Add a failure result
                batch_results.append({
                    "id": q["id"],
                    "type": q.get("type", "unknown"),
                    "question": q["question"],
                    "answer": "Error processing",
                    "abstained": True,
                    "has_valid_citation": False,
                    "total_citations": 0,
                    "valid_citations": 0,
                    "citations": [],
                    "valid_citations_list": [],
                    "error": str(e)
                })
        
        results.extend(batch_results)
        
        # save intermediate results
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    return results

# Generate evaluation report from results.
def generate_report(results):
    by_type = {}    # Group by type
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    total = len(results)
    print(f"\nTotal questions processed: {total}")
    
    # Overall statistics
    total_abstained = sum(1 for r in results if r["abstained"])
    total_supported = sum(1 for r in results if r["has_valid_citation"])
    total_citations = sum(r["total_citations"] for r in results)
    valid_citations = sum(r["valid_citations"] for r in results)
    
    print(f"\nOverall Statistics:")
    print(f"  Abstained: {total_abstained}/{total} ({total_abstained/total*100:.1f}%)")
    print(f"  Supported by citations: {total_supported}/{total} ({total_supported/total*100:.1f}%)")
    print(f"  Citation verification: {valid_citations}/{total_citations} ({valid_citations/total_citations*100:.1f}%)")
    
    # Break down by type
    for qtype, items in by_type.items():
        print(f"\n--- {qtype.upper()} questions ({len(items)}) ---")
        
        if qtype == "unanswerable":
            # Check if they correctly abstained
            correctly_abstained = sum(1 for r in items if r["abstained"])
            print(f"  Correctly abstained: {correctly_abstained}/{len(items)} ({correctly_abstained/len(items)*100:.1f}%)")
            failures = [r for r in items if not r["abstained"]]
            if failures:
                print(f"  Failures (answered unanswerable): {len(failures)}")
                for f in failures[:3]:
                    print(f"    - {f['id']}: {f['answer'][:100]}...")
        else:
            # Answerable questions - check if they have citations
            supported = sum(1 for r in items if r["has_valid_citation"])
            print(f"  Supported with citations: {supported}/{len(items)} ({supported/len(items)*100:.1f}%)")
            failures = [r for r in items if not r["has_valid_citation"]]
            if failures:
                print(f"  Failures (no valid citations): {len(failures)}")
                for f in failures[:3]:
                    reason = "abstained" if f["abstained"] else "no valid citations"
                    print(f"    - {f['id']}: {reason} - {f['answer'][:100]}...")
    
    print("\n" + "="*60)
    
    # List all failed questions
    failed = [r for r in results if (r["type"] != "unanswerable" and not r["has_valid_citation"]) or (r["type"] == "unanswerable" and not r["abstained"])]
    if failed:
        print(f"\nFAILED QUESTIONS ({len(failed)})")
        for f in failed:
            print(f"\n  {f['id']} [{f['type']}]:")
            print(f"    Question: {f['question']}")
            print(f"    Answer: {f['answer'][:150]}...")
            if f["citations"]:
                print(f"    Citations: {len(f['citations'])} total, {f['valid_citations']} valid")
            else:
                print(f"    No citations provided")
    
    return {
        "total": total,
        "abstained": total_abstained,
        "supported": total_supported,
        "citation_rate": valid_citations/total_citations if total_citations > 0 else 0,
        "by_type": by_type,
        "failed": failed
    }

def main():
    """Main evaluation function."""
    print("Loading evaluation questions...")
    
    # Load questions
    with open("eval_questions.json", "r") as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Run evaluation
    results = evaluate_batch(questions, batch_size=3)  # Process 3 at a time
    
    # Generate report
    report = generate_report(results)
    
    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to eval_results.json")
    print(f"Failed questions: {len(report['failed'])}")

if __name__ == "__main__":
    main()