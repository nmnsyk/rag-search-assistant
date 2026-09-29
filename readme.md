# AI-Powered-Search-Assistant-Using-Retrieval-Augmented-Generation-RAG-and-Dynamic-Metadata-Filtering

This project implements an AI-powered search assistant for scientific experiment data. The search pipeline combines BM25-based lexical search, semantic search with ChromaDB, dynamic metadata filtering, and a locally running Large Language Model.

The user query is first analyzed to identify the main search topic and possible metadata requirements. The system then performs both a keyword-based BM25 search and a semantic search. The retrieved results are provided to the language model as context for generating the final response.

## Features

- BM25 search based on experiment titles and subtitles
- Semantic search with ChromaDB
- Embeddings using `BAAI/bge-m3`
- Dynamic filtering based on experiment metadata
- Query analysis and answer generation using `llama3.1:8b`
- Fully local execution with Ollama

## Project Files

The `main.py` file contains the complete search pipeline, including query analysis, BM25 search, semantic search, metadata filtering, and answer generation.

The `data_prep.py` file handles preprocessing of the raw experiment data and generates the cleaned dataset used by the search pipeline.

The `pdf_extraction.py` file contains the logic used to download experiment manuals and extract their textual content from PDF files.

The JSON files included in this repository contain anonymized and reduced example data used solely to demonstrate the expected data structure and the functionality of the pipeline.

## Requirements

The following components are required to run the project:

- Python 3
- Ollama
- Ollama model `llama3.1:8b`
- Python packages listed in `requirements.txt`

The language model can be installed with:

```bash
ollama pull llama3.1:8b
```

## Installation

First, create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the required Python packages:

```bash
python3 -m pip install -r pipeline/requirements.txt
```

## Running the Application

The application should be executed from the `pipeline` directory because the file paths used in the project are relative to this directory:

```bash
cd pipeline
python3 main.py
```

Ollama must be running locally while the application is being executed.

## Data

The files `raw_data.json`, `categories.json`, and `final_cleaned_data.json` included in this repository contain only anonymized and reduced example data.

These files do not represent the original company dataset and are included solely for demonstration purposes. The original production dataset and any confidential company data are not part of this repository.

The example data is intended to illustrate the data structure, preprocessing workflow, and general functionality of the search pipeline. Therefore, results produced using the example dataset may differ from those obtained with the original full dataset.
