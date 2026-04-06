from langchain_core.tools import tool
from ingest.embedder import embed_query
from retrieval.vectorstore import query_vectorstore


@tool
def search_medical_documents(query: str) -> str:
    """Search through indexed medical documents for relevant information."""
    try:
        query_emb = embed_query(query)
        results = query_vectorstore(query_emb, top_k=3)
        
        if not results:
            return "No relevant information found in the documents."
        
        context_parts = []
        for i, chunk in enumerate(results):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "?")
            context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def calculate_dosage(weight_kg: float, dose_per_kg: float, frequency: int) -> str:
    """Calculate medication dosage based on patient weight.
    
    Args:
        weight_kg: Patient weight in kilograms
        dose_per_kg: Dose in mg per kg of body weight
        frequency: Number of doses per day
    """
    try:
        total_daily_dose = weight_kg * dose_per_kg * frequency
        single_dose = weight_kg * dose_per_kg
        
        return (
            f"Dosage Calculation:\n"
            f"- Single dose: {single_dose:.2f} mg\n"
            f"- Daily total ({frequency}x/day): {total_daily_dose:.2f} mg\n"
            f"Note: Always verify with a healthcare professional."
        )
    except Exception as e:
        return f"Calculation error: {str(e)}"


# List of available tools
medical_tools = [search_medical_documents, calculate_dosage]