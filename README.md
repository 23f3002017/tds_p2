# LLM Quiz Solver

An automated system that solves data analysis quizzes using LLMs (Groq API).

## Features
- Automated web scraping with JavaScript rendering (Playwright)
- CSV/PDF data processing and analysis
- Intelligent cutoff-based filtering for data aggregation
- LLM-powered question solving with Groq API
- Quiz chain solving with 3-minute time limit
- Automatic retry mechanism

## Setup

### Prerequisites
- Python 3.9+
- Groq API key (from https://console.groq.com)

### Installation
```bash
pip install -r requirements.txt
playwright install
```

### Configuration
Create a `.env` file: