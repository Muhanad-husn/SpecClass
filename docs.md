# Document Classification Pipeline Documentation

## Overview

This document classification pipeline is designed to process specification documents, create embeddings, store them in a vector database, and then use these to classify new items. The system is built with modularity and efficiency in mind, allowing for easy configuration and optimization.

## Components

### 1. DocumentProcessor

The DocumentProcessor is responsible for processing specification documents. It uses the NLM-Ingestor server to parse various document types (PDF, HTML, Markdown, Text) and chunk them appropriately.

Key features:
- Supports multiple document formats
- Chunking strategies for optimal processing
- OCR capability for processing scanned documents

### 2. EmbeddingManager

The EmbeddingManager handles the creation and caching of embeddings for both documents and items to be classified.

Key features:
- Uses the sentence-transformers library for creating embeddings
- Implements caching to avoid redundant embedding calculations
- Supports batch processing for improved performance

### 3. VectorStore

The VectorStore component uses ChromaDB to store and retrieve document embeddings for similarity search.

Key features:
- Efficient storage and retrieval of document embeddings
- Supports similarity search for finding relevant documents
- Allows for collection management (switching, resetting)

### 4. ClassificationManager

The ClassificationManager is responsible for classifying items using a specified language model (Ollama, OpenAI, or Claude).

Key features:
- Supports multiple language model providers
- Implements caching for classification results
- Batch processing of items for improved efficiency

### 5. FileHandler

The FileHandler manages input and output file operations, including streaming of large input files.

Key features:
- Supports CSV and Excel input files
- Handles large input files efficiently
- Writes classification results to CSV output

## Workflow

1. Document Processing:
   - Specification documents are processed and chunked
   - Embeddings are created for each chunk
   - Chunks and embeddings are stored in the vector store

2. Item Classification:
   - Items to be classified are read from the input file
   - For each item:
     a. An embedding is created
     b. Similar documents are retrieved from the vector store
     c. A context is created from these similar documents
     d. The item is classified using the chosen language model

3. Result Output:
   - Classification results (primary classification, overall classification, reasoning, confidence score) are written to a CSV file

## Configuration

The system is highly configurable through the `config/config.yaml` file. Key configuration options include:

- Input/output directories
- Embedding model selection
- Vector store settings
- Language model selection and settings
- Logging configuration

## Optimization Techniques

- Caching: Embeddings and classification results are cached to avoid redundant computations
- Batch Processing: Documents and items are processed in batches for improved performance
- Modular Design: Components are designed to be interchangeable and easily extendable

## Future Enhancements

- Implement a user interface (CLI or GUI) for easier interaction
- Add support for Arabic language processing
- Implement parallel processing for document embedding and classification

## Troubleshooting

- Check the log files in the `logs/` directory for detailed error messages and debugging information
- Ensure that the NLM-Ingestor server is running and accessible
- Verify that all required API keys are correctly set in the `.env` file

For more information on usage and setup, please refer to the README.md file.
