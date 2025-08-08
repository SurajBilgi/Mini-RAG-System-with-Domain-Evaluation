# Mini RAG System with Domain Evaluation

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and OpenAI, featuring advanced evaluation metrics and a beautiful user interface.

## 🌟 Features

### Core RAG Functionality
- **Multi-format Document Support**: PDF, TXT, CSV, DOC, DOCX files
- **Text Input**: Direct text pasting for quick content processing
- **Intelligent Chunking**: Configurable document chunking with overlap
- **Semantic Search**: OpenAI embeddings for accurate retrieval
- **Conversational Memory**: Maintains context across conversations

### Advanced UI Features
- **Streaming Responses**: Real-time response generation
- **Beautiful Interface**: Modern, responsive design with custom CSS
- **Drag & Drop**: Easy document upload interface
- **Interactive Charts**: Performance metrics visualization
- **Mobile Responsive**: Works seamlessly on all devices

### Evaluation & Analytics
- **Relevance Scoring**: Semantic similarity between query and answer
- **Faithfulness Metrics**: Answer grounding in source documents
- **Readability Analysis**: Multiple readability scores
- **Performance Tracking**: Latency and quality metrics over time
- **Source Attribution**: Clear source document references

### Customization Options
- **Model Selection**: Choose from GPT-3.5, GPT-4, or GPT-4-Turbo
- **Temperature Control**: Adjust response creativity
- **Chunk Size Configuration**: Optimize for your content type
- **Embedding Models**: Multiple OpenAI embedding options

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mini-rag-system.git
   cd mini-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Setup

1. **Enter your OpenAI API Key** in the sidebar
2. **Upload documents** or paste text content
3. **Configure settings** (optional) in the Advanced Settings
4. **Click "Process Documents"** to initialize the RAG system
5. **Start asking questions!**

## 📖 Usage Guide

### Document Processing
- **Upload Files**: Drag and drop or browse for PDF, TXT, CSV, DOC, DOCX files
- **Text Input**: Paste content directly into the text area
- **Batch Processing**: Upload multiple files simultaneously

### Configuration Options
- **Model Selection**: Choose the OpenAI model that fits your needs
- **Temperature**: Control response creativity (0.0 = deterministic, 1.0 = creative)
- **Chunk Size**: Adjust based on document type (500-2000 characters)
- **Chunk Overlap**: Ensure context continuity (50-500 characters)

### Chat Interface
- **Ask Questions**: Type natural language questions about your documents
- **View Sources**: Expand source documents to see where answers come from
- **Track Performance**: Monitor relevance, faithfulness, and response times

## 📊 Evaluation Metrics

### Relevance Score
Measures semantic similarity between the user question and the generated answer using sentence transformers.

### Faithfulness Score
Evaluates how well the answer is grounded in the source documents by comparing answer embeddings with document embeddings.

### Readability Metrics
- **Flesch Reading Ease**: General readability score
- **Flesch-Kincaid Grade**: US grade level equivalent
- **Automated Readability Index**: Alternative readability measure

### Performance Tracking
- **Response Latency**: Time from question to complete answer
- **Quality Trends**: Visualize metrics over time
- **Question History**: Track all interactions

## 🎨 Domain Adaptation

### Art History & Provenance Analysis

This RAG system can be adapted for art-world applications:

#### **Provenance Tracking**
- Upload auction records, gallery documents, and exhibition catalogs
- Query ownership history, exhibition dates, and authentication records
- Cross-reference multiple sources for comprehensive provenance chains

#### **Stylistic Analysis**
- Process art historical texts, criticism, and scholarly articles
- Ask about artistic techniques, influences, and stylistic evolution
- Compare works across periods and movements

#### **Market Analysis**
- Integrate auction results, gallery prices, and market reports
- Query price trends, market performance, and collector preferences
- Analyze factors affecting artwork valuation

### Implementation Examples

```python
# Example queries for art domain:
"What is the provenance of Monet's Water Lilies series?"
"How did Picasso's style evolve during his Blue Period?"
"What factors influenced the price of contemporary art in 2023?"
```

## 🛠️ Technical Architecture

### Components
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: LangChain orchestration framework
- **Embeddings**: OpenAI text-embedding models
- **Vector Store**: ChromaDB for semantic search
- **LLM**: OpenAI GPT models for generation
- **Evaluation**: Custom metrics with sentence-transformers

### Data Flow
1. **Document Ingestion** → Text extraction and preprocessing
2. **Chunking** → Intelligent text splitting with overlap
3. **Embedding** → Vector representation generation
4. **Storage** → Persistent vector database
5. **Retrieval** → Semantic search for relevant chunks
6. **Generation** → LLM response with streaming
7. **Evaluation** → Quality metrics calculation

## 📈 Performance Optimization

### Best Practices
- **Chunk Size**: Optimize based on document type and query complexity
- **Retrieval Count**: Balance context richness with processing speed
- **Model Selection**: Choose appropriate model for quality/speed trade-offs
- **Caching**: Vector store persistence reduces reprocessing time

### Scaling Considerations
- **Document Volume**: ChromaDB handles thousands of documents efficiently
- **Concurrent Users**: Stateless design supports multiple simultaneous users
- **Memory Management**: Conversation window prevents memory overflow

## 🔧 Customization

### Adding New Document Types
```python
# Add custom loaders in the load_documents method
elif uploaded_file.name.endswith('.custom'):
    loader = CustomLoader(tmp_file_path)
```

### Custom Evaluation Metrics
```python
# Extend EvaluationMetrics class
def custom_metric(self, query: str, answer: str) -> float:
    # Your custom evaluation logic
    return score
```

### UI Modifications
Edit the `load_css()` function to customize styling and branding.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For questions, issues, or feature requests:
- Create an issue on GitHub


## 🎯 Sample Questions & Answers

### Example 1: Literature Analysis
**Question**: "What are the main themes in the uploaded literary works?"
**Expected Response**: Detailed analysis of recurring themes with specific references to source documents.

### Example 2: Technical Documentation
**Question**: "How do I configure the authentication system?"
**Expected Response**: Step-by-step instructions extracted from technical documentation.

### Example 3: Research Papers
**Question**: "What methodologies were used in these research studies?"
**Expected Response**: Summary of research methodologies with citations from source papers.

---

Built with ❤️ using Streamlit, LangChain, and OpenAI