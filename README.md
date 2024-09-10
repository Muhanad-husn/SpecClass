
# Document Classification Pipeline

This project implements an optimized document classification pipeline that processes specification documents, classifies items based on these specifications, and provides a confidence score for each classification.

## Features

- Process and store specification documents
- Create and manage embeddings for documents and items
- Perform similarity search using a vector store
- Classify items using various language models (Ollama, OpenAI, Claude)
- Batch processing and caching for improved performance
- Flexible configuration options

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Muhanad-husn/SpecClass.git
   cd SpecClass

   ```

2. **Set up a virtual Conda environment:**
   ```bash
   conda create --name doc_classification python=3.12.4
   conda activate doc_classification
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the NLM-Ingestor Server:**

   - Ensure you have Docker installed, and the Docker daemon is running.
   - Pull the Docker image:
     ```bash
     docker pull jamesmtc/nlm-ingestor:latest
     ```
   - Run the container:
     ```bash
     docker run -p 5010:5001 jamesmtc/nlm-ingestor:latest
     ```

   **Note:** The Docker image is meant for development environments only. For production, users must set up their own server configuration.

   The `nlm-ingestor` server uses a modified version of Apache Tika for document parsing. The server can be deployed locally and provides an easy way to parse and intelligently chunk various document types, including "HTML", "PDF", "Markdown", and "Text". There is an option to enable OCR; refer to the documentation for more details.

   This Docker image was adapted from the interesting project [meta_expert](https://github.com/brainqub3/meta_expert/blob/main/prompt_engineering/jar3d_meta_prompt.md).

5. **Configure the application:**
   - Copy the `.env.example` file to `.env` and fill in your API keys:
     ```bash
     cp .env.example .env
     ```
   - Edit the `config/config.yaml` file to adjust settings as needed.

## Usage

Run the pipeline using the following command:

```bash
python pipeline.py [--reset] [--model-type {ollama|openai|claude}] [--model-name MODEL_NAME]
```

Options:
- `--reset`: Reset the vector store before processing
- `--model-type`: Specify the model type to use (ollama, openai, or claude)
- `--model-name`: Specify the model name to use.

The pipeline will process the documents, classify the items, and output the results to a CSV file in the specified output directory.

## Project Structure

- `pipeline.py`: Main entry point for running the classification pipeline.
- `src/`: Contains the core components of the pipeline.
- `models/`: Defines the base agent and language model interfaces.
- `utils/`: Utility functions for configuration, logging, and file handling.
- `config/`: Configuration files.
- `data/`: Input and output data directories.
- `logs/`: Log files.

## Contributing

Contributions are welcome! Please feel free to fork the repo or submit a Pull Request.

## License

This project is not licensed.
