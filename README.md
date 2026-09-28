# AgroSearch

A text search engine for technical agriculture manuals, with **TFIDF implemented from scratch**. It doesn't use scikit-learn or `TfidfVectorizer`: all the TF, IDF, TFIDF and cosine similarity math is written by hand.

Built for the fictional startup **AgroTech Solutions**.

## Features

- **Preprocessing pipeline:** tokenization → normalization (lowercasing and accent removal) → stopword removal → suffix based stemming
- **Inverted index kept in memory**
- **TFIDF ranking** of documents for the user's query
- **Bonus:** ranking by **cosine similarity**
- **Streamlit UI** to turn stopwords, stemming and cosine similarity on or off, and to inspect each pipeline step

## Tech stack

- Python 3.10+
- Streamlit
- Pandas

## Getting started

```bash
pip install streamlit pandas
streamlit run agrosearch.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
agro-search/
├── agrosearch.py              # Streamlit app + search engine implementation
└── relatorio_agrosearch.pdf   # Project report (Portuguese)
```

## Document base

The corpus has 5 short documents about agriculture (irrigation, biological pest control, green manure and more), hardcoded in the script as the specification required. The documents are in Brazilian Portuguese.
