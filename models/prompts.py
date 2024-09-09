CLASSIFICATION_PROMPT = """
You are an AI assistant trained to classify content based on specific guidelines. Your task is to analyze the given item in the context of the provided information and classify it according to the specifications provided.

Use the following context to inform your decision:

Context: {context}

Specification Book Description: {spec_book_description}

Item Description: {item_description}

Weighted Specification: {weighted_spec}

Consider the guidelines provided in the context and classify the following item:

{item}

Provide your classification along with a brief explanation for your decision and a confidence score between 0 and 1. If a weighted specification is provided, use it to determine the primary classification if there are multiple possible classifications with similar confidence levels.

Your response should be in JSON format with the following structure:
{{
    "primary_classification": "The primary classification category, considering the weighted specification if provided",
    "classification": "The overall classification category",
    "reasoning": "A brief explanation for the classification, including why the primary classification was chosen if different from the overall classification",
    "confidence": A number between 0 and 1 representing your confidence in the classification
}}
"""

GUIDED_JSON = {
    "primary_classification": "string",
    "classification": "string",
    "reasoning": "string",
    "confidence": "number"
}