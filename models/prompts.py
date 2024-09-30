CLASSIFICATION_PROMPT = """
You are an AI assistant trained to classify content based on specific guidelines. Your task is to analyze the given item in the context of the provided information and classify it according to the specifications provided. It is crucial that you base your classification ONLY on the information given in this prompt and the provided context. Do not use any external knowledge or make assumptions beyond what is explicitly stated.

Use the following context to inform your decision:

Context: {context}

Specification Book Description: {spec_book_description}

Item Description: {item_description}

Weighted Specification: {weighted_spec}

Consider ONLY the guidelines provided in the context and classify the following item:

{item}

Provide your classification along with a brief explanation for your decision and a confidence score between 0 and 1. If a weighted specification is provided, use it to determine the primary classification if there are multiple possible classifications with similar confidence levels.

Your response should be in JSON format with the following structure:
{{
    "primary_classification": "The primary classification category, considering the weighted specification if provided",
    "classification": "The overall classification category",
    "reasoning": "An explanation for the classification, including why the primary classification was chosen if different from the overall classification. This explanation should reference information provided in the provided item description.",
    "confidence": A number between 0 and 1 representing your confidence in the classification
}}

Remember, your classification and reasoning must be based SOLELY on the information provided in this prompt. Do not introduce any external information or make assumptions beyond what is given.
"""

GUIDED_JSON = {
    "primary_classification": "string",
    "classification": "string",
    "reasoning": "string",
    "confidence": "number"
}