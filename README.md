# Building an AI Agent for PDF Question Answering with LangChain and Ollama

This repository contains an example of building an AI agent that can answer questions from PDF documents using LangChain and Ollama. The agent utilizes the `gemma3:4b` model to process and respond to queries based on the content of the PDF files.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- The following Ollama models pulled:
  - `gemma3:4b`
  - `nomic-embed-text:latest`

## Setup

1. **Clone this repository**

```sh
git clone git@github.com:woliveiras/reader-agent.git
cd reader-agent
```

2. **Pull the required models with Ollama**

```sh
ollama pull gemma3:4b
ollama pull nomic-embed-text:latest
```

3. **Create and activate a virtual environment**

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. **Install dependencies**

```sh
pip install -r requirements.txt
```

5. **Create the data directory**

```sh
mkdir data
```

6. **Download the PDF files**

Add some PDF files to the `data` directory. You can use any PDF files you want to test the agent.

## Running the Examples

To run the agent and test its functionality, execute the following command:

```sh
python agent.py
```

This will start the agent, which will process the PDF files in the `data` directory and allow you to ask questions about their content.

## Files

- `agent.py`: The main script that initializes the agent and handles user queries.
- `requirements.txt`: The list of Python dependencies required to run the agent.

## References

[Building an AI Agent for PDF Question Answering with LangChain and Ollama](https://woliveiras.github.io/posts/building-ai-agent-pdf-question-answering-langchain-ollama/)
