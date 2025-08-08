# Setup Guide for Mini RAG System

## Quick Fix for the Import Error

The `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` error is due to version conflicts between dependencies. Here's how to fix it:

### Option 1: Automated Installation (Recommended)

```bash
# Run the installation script
python install.py
```

### Option 2: Manual Installation

```bash
# Create a new virtual environment (recommended)
python -m venv rag_env

# Activate the virtual environment
# On macOS/Linux:
source rag_env/bin/activate
# On Windows:
rag_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install compatible versions
pip install -r requirements.txt
```

### Option 3: Force Reinstall (if issues persist)

```bash
# Uninstall conflicting packages
pip uninstall huggingface-hub sentence-transformers transformers -y

# Install specific versions
pip install huggingface-hub==0.20.3
pip install sentence-transformers==2.2.2
pip install transformers==4.36.2

# Install remaining dependencies
pip install -r requirements.txt
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space for models and dependencies

## Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.29.0 | Web application framework |
| `langchain` | 0.1.4 | RAG orchestration |
| `openai` | 1.12.0 | OpenAI API client |
| `sentence-transformers` | 2.2.2 | Embedding models |
| `chromadb` | 0.4.18 | Vector database |
| `huggingface-hub` | 0.20.3 | Model repository access |

## Common Issues and Solutions

### 1. Import Errors

**Problem**: `ImportError: cannot import name 'cached_download'`

**Solution**: 
```bash
pip install --force-reinstall huggingface-hub==0.20.3 sentence-transformers==2.2.2
```

### 2. Version Conflicts

**Problem**: Dependency version conflicts

**Solution**: Create a fresh virtual environment:
```bash
python -m venv fresh_rag_env
source fresh_rag_env/bin/activate  # or fresh_rag_env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Memory Issues

**Problem**: Out of memory when loading models

**Solution**: 
- Use a machine with at least 8GB RAM
- Close other applications
- Consider using smaller embedding models

### 4. Network Issues

**Problem**: Downloads fail due to network restrictions

**Solution**:
```bash
# Set up proxy if needed
pip install --proxy http://proxy.company.com:8080 -r requirements.txt

# Or download models manually and use offline mode
```

## Environment Setup

### 1. Create `.env` file

```bash
# Copy the sample environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

### 2. Required Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_EMBEDDING_MODEL=text-embedding-ada-002
```

## Running the Application

### 1. Start the Streamlit App

```bash
streamlit run app.py
```

### 2. Access the Application

Open your browser and navigate to: `http://localhost:8501`

### 3. Initial Setup

1. Enter your OpenAI API key in the sidebar
2. Upload sample documents or use the provided examples
3. Click "Process Documents" to initialize the RAG system
4. Start asking questions!

## Testing the Installation

### 1. Quick Test

```python
# Run this in a Python shell
import streamlit
import langchain
import openai
import sentence_transformers
print("All packages imported successfully!")
```

### 2. Component Test

```bash
# Test individual components
python -c "from sentence_transformers import SentenceTransformer; print('Sentence Transformers OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "from langchain_openai import OpenAIEmbeddings; print('LangChain OK')"
```

## Performance Optimization

### 1. Model Caching

The application caches models locally. First run may take longer:

```python
# Models are cached in ~/.cache/huggingface/
# To clear cache if needed:
rm -rf ~/.cache/huggingface/transformers/
```

### 2. Resource Management

- **CPU**: Multi-core processors recommended for embedding generation
- **Memory**: 8GB+ RAM for optimal performance
- **Storage**: SSD recommended for faster model loading

## Troubleshooting Commands

### Check Python Environment

```bash
python --version
pip list | grep -E "(streamlit|langchain|openai|sentence-transformers)"
```

### Reset Installation

```bash
# Remove virtual environment
rm -rf rag_env

# Start fresh
python -m venv rag_env
source rag_env/bin/activate
python install.py
```

### Debug Mode

```bash
# Run with debug output
STREAMLIT_LOGGER_LEVEL=debug streamlit run app.py
```

## Docker Setup (Alternative)

If you prefer using Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t mini-rag .
docker run -p 8501:8501 mini-rag
```

## Getting Help

If you continue to experience issues:

1. **Check the GitHub Issues**: Look for similar problems
2. **Environment Details**: Note your Python version, OS, and error messages
3. **Dependency Versions**: Run `pip list` and share relevant package versions
4. **Error Logs**: Include full error traces when reporting issues

## Sample Data

The repository includes sample documents in the `sample_data/` directory:
- `art_history.txt`: Renaissance art history content
- `provenance_example.txt`: Art provenance documentation

Use these to test the system before adding your own documents.

## Next Steps

Once the system is running:

1. **Explore Features**: Try different document types and queries
2. **Customize Settings**: Adjust chunk sizes and model parameters
3. **Add Your Data**: Upload your own documents for analysis
4. **Review Metrics**: Monitor performance and quality scores
5. **Extend Functionality**: Explore the art domain adaptations

## Security Notes

- **API Keys**: Never commit API keys to version control
- **Data Privacy**: Ensure uploaded documents comply with privacy policies
- **Network Security**: Consider using HTTPS in production environments

---

For more detailed information, see the main [README.md](README.md) file. 