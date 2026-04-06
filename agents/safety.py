from typing import Dict, Any
from guardrails import Guard
from guardrails.validator_base import (
    Validator,
    ValidationResult,
    PassResult,
    FailResult,
    register_validator
)


@register_validator(name="unsafe-medical-query", data_type="string")
class UnsafeMedicalQuery(Validator):
    """Validates that a medical query is safe to process."""
    
    UNSAFE_PATTERNS = [
        "how to overdose",
        "lethal dose",
        "how to poison",
        "suicide method",
        "self harm instructions",
        "how to kill"
    ]
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        value_lower = value.lower()
        for pattern in self.UNSAFE_PATTERNS:
            if pattern in value_lower:
                return FailResult(
                    error_message=f"Unsafe content detected: '{pattern}'",
                    fix_value="I cannot process this query as it contains potentially harmful content."
                )
        return PassResult()


@register_validator(name="medical-topic-only", data_type="string")
class MedicalTopicOnly(Validator):
    """Validates that query is medical in nature."""
    
    NON_MEDICAL_PATTERNS = [
        "write code",
        "hack ",
        "password",
        "credit card",
        "political opinion",
        "stock price"
    ]
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        value_lower = value.lower()
        for pattern in self.NON_MEDICAL_PATTERNS:
            if pattern in value_lower:
                return FailResult(
                    error_message=f"Off-topic query detected: '{pattern}'",
                    fix_value="This system only answers medical questions."
                )
        return PassResult()


@register_validator(name="grounded-answer", data_type="string")
class GroundedAnswer(Validator):
    """Validates that answer has citations."""
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        citations = metadata.get("citations", [])
        if not citations:
            return FailResult(
                error_message="Answer has no citations - possible hallucination",
                fix_value="I cannot provide a grounded answer for this query."
            )
        if len(value) < 20:
            return FailResult(
                error_message="Answer too short",
                fix_value="Insufficient information found in documents."
            )
        return PassResult()


def check_query_safety(query: str) -> Dict[str, Any]:
    """Check if a query is safe using Guardrails AI validators directly."""
    try:
        # Check unsafe patterns
        unsafe_validator = UnsafeMedicalQuery(on_fail="fix")
        result1 = unsafe_validator.validate(query, {})
        
        if result1.__class__.__name__ == "FailResult":
            return {
                "safe": False,
                "message": result1.fix_value
            }
        
        # Check topic
        topic_validator = MedicalTopicOnly(on_fail="fix")
        result2 = topic_validator.validate(query, {})
        
        if result2.__class__.__name__ == "FailResult":
            return {
                "safe": False,
                "message": result2.fix_value
            }
        
        return {"safe": True, "message": None}
    
    except Exception as e:
        return {"safe": True, "message": None}


def check_answer_safety(answer: str, citations: list) -> Dict[str, Any]:
    """Check if answer is grounded using Guardrails AI."""
    try:
        grounded_validator = GroundedAnswer(on_fail="fix")
        result = grounded_validator.validate(answer, {"citations": citations})
        
        if result.__class__.__name__ == "FailResult":
            return {
                "safe": False,
                "message": result.fix_value
            }
        
        return {"safe": True, "message": None}
    
    except Exception as e:
        return {"safe": True, "message": None}