from flask import Flask, request, jsonify
import asyncio
from playwright.async_api import async_playwright
import json
import re
import base64
from datetime import datetime
import traceback
import requests
from io import BytesIO, StringIO
import PyPDF2
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

app = Flask(__name__)

YOUR_EMAIL = os.getenv("YOUR_EMAIL", "your-email@example.com")
YOUR_SECRET = os.getenv("YOUR_SECRET", "your-secret-string")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print("DEBUG: GROQ_API_KEY configured:", "✓" if GROQ_API_KEY else "✗")
print(f"Email: {YOUR_EMAIL}")


class QuizSolver:
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.submit_url = None

    async def initialize_browser(self):
        """Initialize Playwright browser."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def close_browser(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()

    async def fetch_quiz_page(self, url):
        """Fetch and render quiz page with JavaScript."""
        try:
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            await self.page.wait_for_timeout(3000)  # Wait for JS rendering
            
            # Get rendered text content
            text_content = await self.page.evaluate('() => document.body.innerText')
            
            # Get all HTML
            content = await self.page.content()
            
            # Extract base64 content if present
            base64_match = re.search(r"atob\(`([^`]+)`\)", content)
            if base64_match:
                encoded = base64_match.group(1)
                try:
                    decoded = base64.b64decode(encoded).decode('utf-8')
                    text_content = decoded + "\n" + text_content
                    print(f"✓ Decoded base64 content")
                except:
                    pass
            
            # Extract submit URL from rendered text
            self.extract_submit_url(text_content, content)
            
            return text_content, content
        except Exception as e:
            print(f"Error fetching page: {e}")
            return None, None

    def extract_submit_url(self, text_content, html_content):
        """Extract submit URL from quiz page."""
        url_pattern = r'https?://[^\s<>"]+/submit[^\s<>"]*'
        matches = re.findall(url_pattern, text_content + html_content)
        if matches:
            self.submit_url = matches[0]
            print(f"✓ Found submit URL: {self.submit_url}")

    def download_file(self, file_url):
        """Download file from URL."""
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            print(f"✓ Downloaded: {file_url[:60]}...")
            return response.content
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            return None

    def extract_pdf_text(self, pdf_content):
        """Extract text from PDF."""
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                page_text = page.extract_text() or ""
                text += page_text
            print(f"✓ Extracted {len(pdf_reader.pages)} PDF pages")
            return text
        except Exception as e:
            print(f"✗ Error extracting PDF: {e}")
            return None

    def parse_csv_data(self, csv_content):
        """Parse CSV and extract statistics."""
        try:
            csv_text = csv_content.decode('utf-8', errors='ignore') if isinstance(csv_content, bytes) else csv_content
            
            # Try parsing with header=None first to check if first row is data
            df_test = pd.read_csv(StringIO(csv_text), header=None, nrows=2)
            
            # If first row looks like data (all numbers), use header=None
            if df_test.shape[1] == 1 and all(str(x).replace('.','',1).isdigit() for x in df_test.iloc[0]):
                df = pd.read_csv(StringIO(csv_text), header=None)
                df.columns = ['value']  # Give it a meaningful name
                print(f"✓ Parsed CSV (no header): {df.shape[0]} rows × {df.shape[1]} cols")
            else:
                df = pd.read_csv(StringIO(csv_text))
                print(f"✓ Parsed CSV (with header): {df.shape[0]} rows × {df.shape[1]} cols")
            
            print(f"   Columns: {list(df.columns)}")
            print(f"   Data types: {df.dtypes.to_dict()}")
            print(f"   First 5 rows:\n{df.head()}")
            
            # Calculate sum of all numeric values
            numeric_sum = 0
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                numeric_sum = df[numeric_cols].sum().sum()
                print(f"   Sum of numeric values: {numeric_sum}")
            
            # Return both raw data and summary
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "numeric_columns": numeric_cols,
                "numeric_sum": numeric_sum,
                "data_preview": df.to_string(),
                "statistics": df.describe().to_string() if numeric_cols else "No numeric columns"
            }
            return df, summary
        except Exception as e:
            print(f"✗ Error parsing CSV: {e}")
            traceback.print_exc()
            return None, None

    def get_best_model(self):
        """Get the best available model from Groq."""
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get('data', [])
                model_ids = [m['id'] for m in models]
                
                preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
                for model in preferred:
                    if model in model_ids:
                        return model
                return model_ids[0] if model_ids else "llama-3.3-70b-versatile"
        except:
            pass
        return "llama-3.3-70b-versatile"

    async def solve_with_groq(self, question_text, additional_data=None):
        """Use Groq API to solve the quiz question."""
        try:
            if not GROQ_API_KEY:
                return {"error": "Groq API key not configured"}

            full_context = question_text
            if additional_data:
                full_context = f"{question_text}\n\n=== PROVIDED DATA ===\n{additional_data[:10000]}"

            prompt = f"""ROLE: Elite data specialist. Precision & accuracy ONLY. ZERO tolerance for errors.

QUESTION:
{full_context}

PROVIDED DATA ANALYSIS:
{additional_data if additional_data else "No additional data"}

CRITICAL INSTRUCTIONS:
1. If a CUTOFF VALUE is mentioned in the data, the question is asking for filtered aggregation
2. When cutoff exists, use the pre-calculated "Sum of values ABOVE cutoff" NOT the total sum
3. If the question says "above", "greater than", "exceeds", or "over" - use the above_cutoff value
4. If the question says "below", "less than", "within", or "below" - use the below_cutoff value
5. Always check the provided data analysis section for pre-calculated sums first
6. Extract EXACT numerical values only - use what's given, don't recalculate
7. NEVER guess - use only pre-calculated values provided in the data

OUTPUT ONLY THIS JSON:
{{"reasoning": "explain which sum you selected and why", "answer": <answer>, "files_needed": []}}"""

            model = self.get_best_model()
            print(f"🔄 Calling Groq with {model}...")
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"✗ API error: {response.status_code}")
                return None

            result = response.json()
            response_text = result['choices'][0]['message']['content'].strip()
            print(f"✓ LLM response: {response_text[:150]}...")

            # Try to extract JSON object from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    answer = parsed.get('answer')
                    print(f"✓ Extracted answer: {answer}")
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing failed: {e}")
                    # Try to extract just the answer value
                    answer_match = re.search(r'"answer"\s*:\s*([^,}\]]+)', response_text)
                    if answer_match:
                        answer_str = answer_match.group(1).strip().strip('"')
                        try:
                            answer = int(answer_str)
                        except:
                            answer = answer_str
                        print(f"✓ Extracted answer from text: {answer}")
                        return {"answer": answer, "reasoning": "Extracted from response"}

            # Last resort: try to find any number in the response
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                print(f"⚠️  Extracting number from response: {numbers[0]}")
                return {"answer": int(numbers[0]), "reasoning": "Extracted number"}

            return {"answer": response_text}

        except Exception as e:
            print(f"✗ Error with Groq: {e}")
            traceback.print_exc()
            return None

    async def solve_quiz(self, quiz_url):
        """Main quiz solving logic."""
        try:
            print(f"\n{'='*70}")
            print(f"🧩 SOLVING: {quiz_url}")
            print(f"{'='*70}")

            # Fetch the quiz page
            text_content, html_content = await self.fetch_quiz_page(quiz_url)
            if not text_content:
                return None, "Failed to fetch quiz page"

            print(f"📄 Retrieved {len(text_content)} chars of content")
            print(f"Preview: {text_content[:300]}...\n")

            additional_data = ""
            
            # Find all download links (CSV, PDF, JSON, etc.)
            file_urls = re.findall(
                r'(?:href=|download|link|file)["\']?([^\s"\'<>]+\.(?:csv|pdf|json|xlsx?|txt|tsv))["\']?',
                text_content + html_content,
                re.IGNORECASE
            )
            
            # Also find URLs in text
            file_urls.extend(re.findall(
                r'https?://[^\s<>"]+\.(?:csv|pdf|json|xlsx?|txt)',
                text_content,
                re.IGNORECASE
            ))
            
            file_urls = list(set(file_urls))  # Remove duplicates
            
            # Download and process files
            if file_urls:
                print(f"📥 Found {len(file_urls)} file(s):")
                for file_url in file_urls:
                    # Handle relative URLs
                    if file_url.startswith('/'):
                        base = quiz_url.split('?')[0].rsplit('/', 1)[0]
                        file_url = base + file_url
                    elif not file_url.startswith('http'):
                        # Relative path like "demo-audio-data.csv"
                        base = quiz_url.split('?')[0].rsplit('/', 1)[0]
                        file_url = base + '/' + file_url
                    
                    print(f"   → {file_url[:60]}...")
                    file_content = self.download_file(file_url)
                    
                    if file_content:
                        if file_url.lower().endswith('.pdf'):
                            pdf_text = self.extract_pdf_text(file_content)
                            if pdf_text:
                                additional_data += f"\n[PDF DATA]\n{pdf_text[:8000]}\n"
                        
                        elif file_url.lower().endswith('.csv'):
                            df, summary = self.parse_csv_data(file_content)
                            if df is not None:
                                # Look for cutoff value in the question
                                cutoff_match = re.search(r'[Cc]utoff:\s*(\d+)', text_content)
                                cutoff = int(cutoff_match.group(1)) if cutoff_match else None
                                
                                # Calculate different aggregations
                                total_sum = summary['numeric_sum']
                                above_cutoff = None
                                below_cutoff = None
                                count_above = None
                                count_below = None
                                
                                if cutoff and summary['numeric_columns']:
                                    numeric_col = summary['numeric_columns'][0]
                                    above_cutoff = df[df[numeric_col] > cutoff][numeric_col].sum()
                                    below_cutoff = df[df[numeric_col] <= cutoff][numeric_col].sum()
                                    count_above = len(df[df[numeric_col] > cutoff])
                                    count_below = len(df[df[numeric_col] <= cutoff])
                                
                                # Include the actual sum in the data sent to LLM
                                additional_data += f"\n[CSV ANALYSIS]\n"
                                additional_data += f"File: {file_url}\n"
                                additional_data += f"Rows: {summary['shape'][0]}, Columns: {summary['shape'][1]}\n"
                                additional_data += f"Column names: {summary['columns']}\n"
                                additional_data += f"Numeric columns: {summary['numeric_columns']}\n"
                                additional_data += f"**TOTAL SUM OF ALL VALUES: {total_sum}**\n"
                                
                                if cutoff:
                                    additional_data += f"\n**CUTOFF VALUE FOUND: {cutoff}**\n"
                                    additional_data += f"Sum of values ABOVE cutoff ({cutoff}): {above_cutoff}\n"
                                    additional_data += f"Count of values ABOVE cutoff: {count_above}\n"
                                    additional_data += f"Sum of values <= cutoff ({cutoff}): {below_cutoff}\n"
                                    additional_data += f"Count of values <= cutoff: {count_below}\n"
                                
                                additional_data += f"\nFirst 20 rows:\n{df.head(20).to_string()}\n"
                                additional_data += f"\nStatistics:\n{summary['statistics']}\n"
                        
                        else:
                            try:
                                text_data = file_content.decode('utf-8', errors='ignore')
                                additional_data += f"\n[FILE DATA]\n{text_data[:5000]}\n"
                            except:
                                pass
            
            # Find and scrape relative URLs (like /demo-scrape-data)
            scrape_urls = re.findall(r'(/[\w\-/?.=&@]+)', text_content)
            scrape_urls = [u for u in scrape_urls if 'scrape' in u.lower() or 'data' in u.lower()]
            
            if scrape_urls:
                print(f"📥 Found {len(scrape_urls)} endpoint(s) to scrape:")
                for scrape_path in scrape_urls[:3]:  # Limit to 3
                    base = quiz_url.split('?')[0].rsplit('/', 1)[0]
                    full_url = base + scrape_path
                    
                    print(f"   → {full_url[:60]}...")
                    try:
                        await self.page.goto(full_url, wait_until='networkidle', timeout=10000)
                        await self.page.wait_for_timeout(500)
                        scraped_text = await self.page.evaluate('() => document.body.innerText')
                        additional_data += f"\n[SCRAPED DATA]\n{scraped_text[:3000]}\n"
                        print(f"   ✓ Scraped {len(scraped_text)} chars")
                    except Exception as e:
                        print(f"   ✗ Error: {e}")
            
            # Solve with LLM
            if additional_data:
                print(f"\n🔍 Solving with {len(additional_data)} chars of data...\n")
                solution = await self.solve_with_groq(text_content, additional_data)
            else:
                print(f"\n🔍 Solving with question only...\n")
                solution = await self.solve_with_groq(text_content)

            if not solution:
                return None, "Failed to solve"

            answer = solution.get('answer')
            print(f"✓ ANSWER: {answer}\n")
            return answer, None

        except Exception as e:
            print(f"✗ Error solving quiz: {e}")
            traceback.print_exc()
            return None, str(e)


solver = QuizSolver()


async def solve_quiz_chain(initial_url, email, secret):
    """Solve chain of quizzes within 3-minute time limit."""
    try:
        await solver.initialize_browser()

        current_url = initial_url
        start_time = datetime.now()
        max_duration = 180
        attempt_count = 0
        tried_answers = set()

        while current_url:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_duration:
                print(f"\n⏰ TIME LIMIT EXCEEDED")
                break

            print(f"\n⏱️ ELAPSED: {elapsed:.0f}s / {max_duration}s")

            answer, error = await solver.solve_quiz(current_url)

            if error or answer is None:
                print(f"✗ FAILED: {error}")
                break

            answer_str = str(answer)
            if answer_str in tried_answers:
                print(f"⚠️  Already tried this answer, skipping...")
                next_url = None
            else:
                tried_answers.add(answer_str)
                print(f"📤 SUBMITTING: {answer}")
                submit_response = submit_answer(current_url, email, secret, answer, solver.submit_url)

                if not submit_response:
                    break

                next_url = submit_response.get('url')
                
                if submit_response.get('correct'):
                    print("✅ CORRECT!")
                    attempt_count = 0
                    tried_answers.clear()
                    if not next_url:
                        print("🎉 QUIZ COMPLETED!")
                        break
                else:
                    print(f"❌ INCORRECT: {submit_response.get('reason')}")
                    attempt_count += 1
                    
                    if attempt_count >= 2 or "sum" not in submit_response.get('reason', '').lower():
                        if next_url and next_url != current_url:
                            print(f"📍 Moving to next quiz")
                        else:
                            print(f"❌ Quiz failed, no next URL provided")
                            break
            
            if next_url and next_url != current_url:
                current_url = next_url
            else:
                break

        await solver.close_browser()

    except Exception as e:
        print(f"✗ Error in quiz chain: {e}")
        traceback.print_exc()
        try:
            await solver.close_browser()
        except:
            pass


def submit_answer(quiz_url, email, secret, answer, submit_url=None):
    """Submit answer to the quiz endpoint."""
    try:
        if not submit_url:
            base_url = quiz_url.rsplit('/', 1)[0]
            submit_url = f"{base_url}/submit"

        payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": answer
        }

        response = requests.post(submit_url, json=payload, timeout=30)
        return response.json()

    except Exception as e:
        print(f"✗ Error submitting: {e}")
        return None


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "message": "LLM Quiz Solver (Groq)",
        "email": YOUR_EMAIL
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/quiz", methods=["POST"])
def handle_quiz():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        email = data.get('email')
        secret = data.get('secret')
        url = data.get('url')

        if not all([email, secret, url]):
            return jsonify({"error": "Missing required fields"}), 400

        if secret != YOUR_SECRET:
            return jsonify({"error": "Invalid secret"}), 403

        if email != YOUR_EMAIL:
            return jsonify({"error": "Invalid email"}), 403

        print(f"\n{'#'*70}")
        print(f"# NEW QUIZ REQUEST")
        print(f"# Email: {email}")
        print(f"# URL: {url}")
        print(f"{'#'*70}\n")

        asyncio.run(solve_quiz_chain(url, email, secret))

        return jsonify({
            "status": "success",
            "message": "Quiz processing completed"
        }), 200

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 LLM QUIZ SOLVER - GROQ EDITION (ENHANCED)")
    print(f"{'='*70}")
    print(f"Email: {YOUR_EMAIL}")
    print(f"Secret: {'✓' if YOUR_SECRET else '✗'}")
    print(f"Groq API: {'✓' if GROQ_API_KEY else '✗'}")
    print(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)