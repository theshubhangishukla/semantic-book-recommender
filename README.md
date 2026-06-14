# Semantic Book Recommender

An intelligent book recommendation system powered by Large Language Models. Instead of matching keywords, this system understands the *meaning* behind your query - so you can search for something like *"a story about forgiveness and second chances"* and get back books that actually fit.

---

## What It Does

- Takes a natural language query from the user (e.g. *"an adventure story with a strong female lead"*)
- Searches a vector database of book descriptions using **semantic similarity**
- Filters results by **category** and **emotional tone** (joyful, suspenseful, sad, etc.)
- Displays ranked recommendations with **book covers, titles, and descriptions** in an interactive dashboard

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Hugging Face Transformers | Embeddings, zero-shot classification + sentiment analysis |
| `all-MiniLM-L6-v2` | Sentence embedding model (via `HuggingFaceEmbeddings`) |
| LangChain | LLM orchestration and vector search |
| Chroma | Vector database |
| Gradio | Interactive web dashboard |
| Pandas / NumPy | Data cleaning and manipulation |
| Jupyter Notebook | Exploration and prototyping |
| Kaggle Dataset | Source book data |

---

## How It Works

### 1. Data Cleaning
Raw book data from Kaggle is cleaned - handling missing values, removing short or uninformative descriptions, and standardising categories.

### 2. Vector Embeddings
Book descriptions are chunked and converted into vector embeddings using LangChain + the `all-MiniLM-L6-v2` model from Hugging Face (`HuggingFaceEmbeddings`). These embeddings capture the *semantic meaning* of each description as a mathematical representation - entirely open-source, no API key needed.

### 3. Vector Database
Embeddings are stored in a **Chroma** vector database. At query time, the user's input is also embedded and compared against all stored books using cosine similarity.

### 4. Zero-Shot Classification
A Hugging Face LLM classifies each book description into categories - without needing labelled training data.

### 5. Sentiment / Emotion Analysis
A fine-tuned LLM extracts emotional tone from descriptions, tagging books as joyful, suspenseful, sad, surprising, and so on.

### 6. Gradio Dashboard
Everything is bundled into a Gradio UI where users can enter a query, pick a category filter, choose an emotional tone, and browse results with book cover thumbnails.

---

## Project Structure

```
book-recommender/
│
├── data-exploration.ipynb       # Data cleaning and preparation
├── vector-search.ipynb          # Building the vector database
├── text-classification.ipynb    # Zero-shot classification
├── sentiment-analysis.ipynb     # Emotion extraction
├── gradio-dashboard.py          # Final interactive dashboard
│
├── requirements.txt             # Project dependencies
├── .env.example                 # Template for API keys (don't share your real .env)
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- The book dataset from [Kaggle](https://www.kaggle.com/)
- No API key needed - all models run locally via Hugging Face Transformers

### Installation

```bash
# Clone the repo
git clone https://github.com/theshubhangishukla/semantic-book-recommender.git
cd semantic-book-recommender

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```



### Run the dashboard

```bash
python gradio-dashboard.py
```

Then open the local URL Gradio gives you in your browser.

---

## Dashboard Preview

*will add a screenshot of my Gradio dashboard here!*

---

## What I Learned

- How vector embeddings represent text as meaning, not just characters
- The difference between keyword search and semantic search
- How to use LangChain to orchestrate LLM workflows
- Zero-shot classification - making a model label data it was never trained on
- How fine-tuned models detect emotional tone in text
- Building and deploying an ML-backed UI with Gradio

---

## Connect

Made by [Shubhangi Shukla](https://www.linkedin.com/in/theshubhangishukla) - feel free to reach out!
