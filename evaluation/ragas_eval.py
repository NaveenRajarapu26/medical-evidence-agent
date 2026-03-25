import os
import json
from typing import Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "dummy"
os.environ["OPENAI_API_KEY"] = "dummy"

from ingest.embedder import embed_query
from retrieval.vectorstore import query_vectorstore
from retrieval.qa_chain import generate_answer


def run_ragas_evaluation(golden_set_path: str = "evaluation/golden_set.json") -> Dict[str, float]:

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    print(f"Running evaluation on {len(golden_set)} questions...")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden_set:
        query = item["question"]
        ground_truth = item["ground_truth"]

        query_emb = embed_query(query)
        dense_results = query_vectorstore(query_emb, top_k=5)
        final_chunks = dense_results[:3]
        result = generate_answer(query, final_chunks)

        questions.append(query)
        answers.append(result["answer"])
        contexts.append([chunk["text"] for chunk in final_chunks])
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Setup Groq as LLM for RAGAS
    groq_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    wrapped_llm = LangchainLLMWrapper(groq_llm)

    faithfulness.llm = wrapped_llm
    answer_relevancy.llm = wrapped_llm
    context_recall.llm = wrapped_llm

    print("Running RAGAS metrics...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    scores = {
        "faithfulness": round(float(results["faithfulness"]), 3),
        "answer_relevancy": round(float(results["answer_relevancy"]), 3),
        "context_recall": round(float(results["context_recall"]), 3)
    }

    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        print(f"{metric}: {score}")

    return scores


if __name__ == "__main__":
    run_ragas_evaluation()