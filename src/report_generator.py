from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config_loader import config
from utils.logger import logger

class ReportGenerator:
    def __init__(self):
        self.model_name = config['llm_model_name']
        self.tokenizer = None
        self.model = None

    def load_model(self):
        logger.info(f"Loading LLM model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise

    def generate_report(self, item, similar_chunks):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        logger.info("Generating report")
        try:
            prompt = self._create_prompt(item, similar_chunks)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=500)
            report = self.tokenizer.decode(outputs[0])
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _create_prompt(self, item, similar_chunks):
        prompt = f"Analyze the following item and its similar specification chunks:\n\nItem: {item}\n\nSimilar Chunks:\n"
        for i, chunk in enumerate(similar_chunks, 1):
            prompt += f"{i}. {chunk.page_content}\n"
        prompt += "\nGenerate a report analyzing the similarities and rationalizing the classification:"
        return prompt