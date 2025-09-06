# Document Similarity Analysis Tool - Business Case v6

## 📋 Project Overview

This tool provides automated analysis of text documents (books, articles, reports, academic papers) to detect repetitive content patterns. It uses advanced NLP techniques to group similar paragraphs into clusters and generates concise summaries for each group.

### Key Benefits
- **Time Savings**: Automatically identifies redundant content that would take hours to find manually
- **Quality Improvement**: Helps authors and editors identify unintentional repetitions
- **Content Optimization**: Enables creation of more concise documents by highlighting duplicate information
- **Academic Integrity**: Useful for detecting self-plagiarism in academic works

## 🎯 Use Cases

1. **Academic Papers**: Identify repetitive explanations or definitions
2. **Technical Documentation**: Find duplicate instructions or specifications
3. **Business Reports**: Detect redundant analysis or conclusions
4. **Books & Articles**: Locate repeated themes or examples
5. **Legal Documents**: Identify duplicate clauses or terms

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Install required packages
pip install openai scikit-learn sentence-transformers numpy pandas python-docx PyMuPDF
```

### Environment Setup
```bash
# Windows
setx OPENAI_API_KEY "your_api_key_here"

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"
```

### Project Structure
```
C:/projekty/your_project_name/
├── input_document.pdf        # Your source document
├── similarity_tool.py         # Core analysis engine
├── run_document_test.py      # Main execution script
├── analyze_results_v2.py     # Content type classification
├── analyze_results_v3.py     # AI-powered summarization
├── extract_clusters.py       # Cluster extraction utility
├── extract_top_examples.py   # Representative examples selector
├── results_doc.csv           # Raw analysis results
├── clusters/                 # Extracted cluster files
├── clusters_v2/              # Classified clusters
└── clusters_v3/              # Summarized clusters
```
or
C:\User)name\Documents\DocumentAnalysis\
├── input_document.pdf        # Your source document
├── similarity_tool.py         # Core analysis engine
├── run_document_test.py      # Main execution script
├── analyze_results_v2.py     # Content type classification
├── analyze_results_v3.py     # AI-powered summarization
├── extract_clusters.py       # Cluster extraction utility
├── extract_top_examples.py   # Representative examples selector
├── results_doc.csv           # Raw analysis results
├── clusters/                 # Extracted cluster files
├── clusters_v2/              # Classified clusters
└── clusters_v3/              # Summarized clusters

## 📊 Workflow

### Step 1: Initial Analysis
```python
# Edit run_document_test.py
PROJECT_NAME = "your_project"
PROJECT_FILE = "your_document.pdf"

# Run analysis
python run_document_test.py
```

### Step 2: Classification
```python
# Classify content types (TOC, Tables, Content)
python analyze_results_v2.py
```

### Step 3: AI Summarization
```python
# Generate cluster summaries with GPT-4
python analyze_results_v3.py
```

### Step 4: Extract Examples
```python
# Get representative samples from each cluster
python extract_top_examples.py
```

## 🔧 Configuration Options

### Model Selection
- **gpt-4o**: Best quality, especially for Polish language
- **gpt-4o-mini**: Budget-friendly option for testing

### Clustering Parameters
- **eps** (0.3): Similarity threshold - lower = stricter matching
- **min_samples** (2): Minimum cluster size
- **chunk_size** (20): Paragraphs processed per batch

## 📝 XML Framework for Custom Summarization

Use this framework for generating high-quality cluster summaries:

```xml
<prompt>
    <role>Expert in analyzing [document type: reports/articles/academic papers/books]</role>
    <task>Summarize a group of similar paragraphs (cluster) in 3-4 sentences</task>
    <objective>
        <goal>Identify the main recurring idea across paragraphs</goal>
        <expected_output>Concise, coherent description in original language</expected_output>
    </objective>
    <context>
        <source_file>[file_name]</source_file>
        <file_path>[full_path]</file_path>
        <cluster_id>[SIMILAR-XX]</cluster_id>
        <paragraph_count>[number]</paragraph_count>
    </context>
    <audience>User preparing a condensed version of the document</audience>
    <constraints>
        <max_words>150</max_words>
        <style>Clear and concise</style>
        <avoid>Full quotations, redundant phrases</avoid>
    </constraints>
    <input_prompt>
        <data_format>text</data_format>
        <example_input>
            <![CDATA[
            SIMILAR-00 (5 paragraphs):
            1. Lorem Ipsum is simply dummy text...
            2. Lorem Ipsum has been the industry's standard...
            3. It is a long established fact that Lorem Ipsum...
            ]]>
        </example_input>
    </input_prompt>
    <format>
        <output_type>summary</output_type>
        <example_output>
            <![CDATA[
            **Cluster SIMILAR-00**: This cluster focuses on explaining the origin and 
            significance of "Lorem Ipsum" text in the publishing and typographic industry. 
            The paragraphs describe its historical use as placeholder text and its 
            continued relevance in modern design workflows.
            ]]>
        </example_output>
    </format>
</prompt>
```

## 📈 Interpreting Results

### Similarity Categories
- **<10%**: Low similarity - likely unique content
- **10-25%**: Some similarity - related topics
- **25-50%**: Similar - overlapping concepts
- **50-75%**: Very similar - likely repetitive
- **>75%**: Nearly identical - strong duplication

### Cluster Types
- **UNIQUE**: Standalone paragraphs with no duplicates
- **SIMILAR-XX**: Groups of related/duplicate content
- **TOC**: Table of contents entries
- **Tabela/Case**: Tables or case studies

## 🎯 Best Practices

1. **Pre-processing**: Clean your document of headers/footers before analysis
2. **Language**: The tool works with mixed Polish/English content
3. **File Size**: For documents >100 pages, consider splitting into chapters
4. **API Usage**: Monitor your OpenAI API usage to control costs
5. **Offline Mode**: Tool works without API but won't generate summaries

## 🔍 Advanced Usage

### Custom Similarity Thresholds
```python
# In similarity_tool.py, adjust clustering parameters:
labels = cluster_paragraphs(embeddings, eps=0.25, min_samples=3)
```

### Batch Processing Multiple Documents
```python
import glob
for file in glob.glob("*.pdf"):
    process_document(file, model_name="gpt-4o")
```

### Export to Different Formats
```python
# CSV for data analysis
df.to_csv("results.csv", encoding="utf-8")

# JSON for web applications
df.to_json("results.json", orient="records")

# Excel for business users
df.to_excel("results.xlsx", index=False)
```

## 📊 Output Files Explained

| File | Description | Use Case |
|------|-------------|----------|
| output.docx | Annotated document with cluster labels | Review duplicates in context |
| results_doc.csv | Raw paragraph-cluster mapping | Data analysis & filtering |
| cluster_summary_v2.csv | Classified clusters by type | Content categorization |
| cluster_summary_v3.csv | AI-generated summaries | Quick overview of duplicates |
| cluster_examples.csv | Sample paragraphs per cluster | Quality verification |

## 🚨 Troubleshooting

### Common Issues

1. **"No API Key"**: Set OPENAI_API_KEY environment variable
2. **"UnicodeDecodeError"**: Ensure UTF-8 encoding in file operations
3. **"Memory Error"**: Reduce chunk_size for large documents
4. **"No clusters found"**: Adjust eps parameter (try 0.4 or 0.5)

### Performance Tips

- Process documents in sections for better memory management
- Use gpt-4o-mini for initial testing to reduce costs
- Cache embeddings for repeated analyses
- Consider local LLM models for sensitive documents

## 📚 Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [DBSCAN Clustering Algorithm](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [OpenAI API Best Practices](https://platform.openai.com/docs/guides/best-practices)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

## 📄 License & Support

This tool is provided as-is for document analysis purposes. For support or custom implementations, please refer to the project repository.

---
*Version 6.0 - Enhanced with comprehensive documentation and best practices*