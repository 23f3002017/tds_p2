# LLM Quiz Solver

An intelligent automated quiz solving system that uses LLMs to analyze data, solve complex questions, and handle multi-question quiz chains within strict time constraints.

## Overview

This Flask-based application solves data analysis quizzes by:
- **Fetching** quiz pages with JavaScript rendering
- **Downloading** and parsing data files (CSV, PDF, JSON, etc.)
- **Processing** data with intelligent analysis
- **Solving** questions using Groq LLM API
- **Managing** multi-question quiz chains with submission tracking

## Features

**Automated web scraping with JavaScript rendering (Playwright)**
- Headless browser automation for dynamic content
- Network idle detection for full page loads
- JavaScript execution for rendered content extraction

**CSV/PDF data processing and analysis**
- Intelligent header detection for headerless datasets
- Multi-page PDF extraction with OCR support
- Numeric aggregation and statistics calculation

**Intelligent cutoff-based filtering for data aggregation**
- Automatic cutoff value detection in questions
- Filtered sum calculations (above/below cutoff)
- Multiple aggregation perspectives for LLM context

**LLM-powered question solving with Groq API**
- Dynamic model selection from available options
- Multi-stage response parsing (JSON → Regex → Numbers)
- Graceful fallback mechanisms

**Quiz chain solving with 3-minute time limit per question**
- Fresh timer for each question
- Immediate timeout enforcement
- Prevents excessive processing

**Automatic retry mechanism**
- Duplicate answer detection
- Smart state management across attempts
- Graceful escalation to next question

## Architecture

### Core Components

#### 1. **QuizSolver Class**
Main orchestrator for solving individual quiz questions.

**Key Methods:**
- `initialize_browser()` - Launches Playwright chromium instance
- `fetch_quiz_page(url)` - Renders JS and extracts content
- `download_file(file_url)` - Fetches files with error handling
- `parse_csv_data()` - Analyzes CSV with intelligent header detection
- `extract_pdf_text()` - Page-by-page PDF text extraction
- `solve_with_groq()` - LLM inference with JSON parsing

#### 2. **Quiz Chain Management**
Handles multiple sequential questions with state tracking.

**Workflow:**
```
POST /quiz → Validate credentials → Start timer
├─ Fetch question page
├─ Download & analyze data
├─ Call LLM for answer
├─ Submit answer
├─ Check: correct? → Next URL? → Repeat
└─ Timeout? → Break
```

### Techniques Used

#### 1. Playwright Browser Automation
- JavaScript rendering with network idle detection
- Dynamic content extraction
- Handles content injection and DOM manipulation

#### 2. Base64 Decoding
- Detects base64-encoded quiz content
- Automatic decoding for processing
- Integrates with text extraction pipeline

#### 3. Intelligent CSV Header Detection
- Tests first row for numeric patterns
- Automatically chooses parsing strategy
- Handles headerless and structured datasets

#### 4. Cutoff-Based Aggregation
- Regex pattern matching for cutoff values
- Filtered sum calculations
- Provides multiple data perspectives to LLM

#### 5. URL Resolution & Normalization
- Handles absolute, relative, and partial URLs
- Automatic base URL construction
- Prevents broken download links

#### 6. LLM Response Parsing
- Multi-stage extraction (JSON → Regex → Numbers)
- Handles malformed LLM responses
- Graceful degradation

#### 7. Time Management
- Per-question 3-minute limit
- Immediate timeout enforcement
- Prevents resource exhaustion

#### 8. Smart Retry & State Management
- Duplicate answer prevention
- Attempt tracking per question
- Graceful failure handling

#### 9. Security & Validation
- Email & secret verification
- Proper HTTP status codes
- Environment-based configuration

#### 10. Dynamic Model Selection
- Queries Groq for available models
- Prefers high-performance models
- Automatic fallback to defaults

## Setup

### Prerequisites
- Python 3.9+
- Groq API key (from https://console.groq.com)

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd iitm_p2

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### Configuration

Create a `.env` file:
```env
YOUR_EMAIL=your-email@example.com
YOUR_SECRET=your-secret-string
GROQ_API_KEY=your-groq-api-key
PORT=5000
```

**Configuration Details:**
- `YOUR_EMAIL` - Student email for verification
- `YOUR_SECRET` - Secret string for request validation
- `GROQ_API_KEY` - API key from Groq console
- `PORT` - Server port (default: 5000)

## API Endpoints

### POST `/quiz`
Solves a quiz question chain.

**Request:**
```json
{
  "email": "student@example.com",
  "secret": "student-secret",
  "url": "https://example.com/quiz-834"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "message": "Quiz processing completed"
}
```

**Response (Invalid Secret):**
```json
{
  "error": "Invalid secret"
}
```

### GET `/health`
Health check endpoint.

```bash
curl http://localhost:5000/health
# Response: {"status": "healthy"}
```

### GET `/`
Server info endpoint.

```bash
curl http://localhost:5000/
# Response: {"status": "ok", "message": "LLM Quiz Solver (Groq)", "email": "..."}