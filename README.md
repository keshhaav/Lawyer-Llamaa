# Lawyer Llama 🦙⚖️

Lawyer Llama is an intelligent legal assistant powered by LLaMA and built with Streamlit. This application provides precise, context-aware legal information by leveraging advanced language models and vector search technology.

## Overview

Lawyer Llama is designed to:
- Process and understand legal documents
- Provide accurate, context-based legal information
- Maintain conversation history for seamless interaction
- Generate precise responses using the LLaMA model
- Search through legal documents using Pinecone vector database

## Features

- 💬 Interactive chat interface
- 📚 Context-aware responses from legal documents
- 🔍 Advanced vector search capabilities
- 🔄 Streaming responses for better user experience
- 🧠 Powered by Meta's LLaMA model
- 📝 Chat history management
- 🔒 Secure API key handling

## Technical Stack

- **Frontend**: Streamlit
- **Language Model**: Meta-LLaMA 3.2-1B-Instruct
- **Vector Store**: Pinecone
- **Embeddings**: Sentence Transformers
- **Document Processing**: LangChain

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Hugging Face API key
- Pinecone API key
- Required Python packages (see `requirements.txt`)

### Environment Variables

Before running the application, make sure to set up the following secrets:
```
HUGGINGFACE_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=your_pinecone_environment
```
