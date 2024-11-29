
# Retrieval-Augmented Generator (RAG) for SAFe® for Teams Workbook

![](asset/rag-sample.png)

## Overview

This project demonstrates a **Retrieval-Augmented Generator (RAG)** system designed to answer multiple-choice questions based on the content of an uploaded PDF file to allow users to quickly extract and synthesize information from complex documents. 

In this implementation, the example document is the *SAFe® for Teams: Establishing Team Agility for Agile Release Trains 6.0.2* workbook. The system can be accessed via a Command Line Interface (CLI) or through sample notebooks provided in the `notebooks/` folder.

---

## Features

- **Efficient PDF Parsing**: Extracts content from PDF files and splits it into manageable chunks for processing.
- **AI-Powered Question Answering**: Leverages OpenAI's GPT-4o-mini and FAISS (a fast vector search library) to answer multiple-choice questions accurately.
- **Streamlined CLI Interface**: Interact with the RAG system directly through a terminal-based command-line interface.

---

## Technologies Used

- **OpenAI GPT-4o-mini**: Basse LLM.
- **PyPDFLoader**: To extract text from PDF documents.
- **LangChain**: For document loading, vector store management, and tools integration.
- **FAISS**: For efficient vector-based document retrieval.

---

## Installation

### Prerequisites
- Python 3.8 or later
- An OpenAI API key
- Dependencies listed in `requirements.txt`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ask-your-pdf.git
   cd ask-your-pdf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

4. Place your PDF document in the `data/` directory (e.g., `data/CERT_RAG.pdf`).

---

## Usage

### CLI Interface
Run the CLI script to interact with the system:
```bash
python rag_cli.py
```

Follow the prompts to input your question and multiple-choice options. For example:
```text
Enter your question (or type 'exit' to quit):
What is the primary purpose of Agile Release Trains?
A. To deliver value to customers
B. To manage team dependencies
C. To ensure compliance with regulations
D. To create detailed project plans
```

The system will process the input and provide an AI-generated answer based on the PDF content.

---

## File Structure

```
.
├── data/
│   └── CERT_RAG.pdf              # Example PDF document
├── notebooks/
│   └── example_notebook.ipynb    # Sample notebook for experimentation
├── assets/
│   └── rag-sample.png            # Example output image
├── requirements.txt              # Python dependencies
├── rag_cli.py                    # CLI script for RAG system
├── README.md                     # Project documentation
```

---

## Example Output

Here is a sample output from the system:

```text
Enter your question (or type 'exit' to quit):
What is the primary purpose of Agile Release Trains?
A. To deliver value to customers
B. To manage team dependencies
C. To ensure compliance with regulations
D. To create detailed project plans

Generating answer based on the document...

================================== AI Message ==================================
The primary purpose of Agile Release Trains is: A. To deliver value to customers.
```
---

## Future Enhancements

- Add support for additional document formats (e.g., Word, HTML).
- Extend the CLI to include batch processing for multiple questions.
- Develop a web-based GUI for easier accessibility.
- Enhance support for multilingual documents.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contact

For questions or collaboration, feel free to reach out:
- **Email**: reinbugnot@gmail.com
- **LinkedIn**: [Rein Bugnot](https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/reinbugnot/)

---
