# AI Meeting Notes Analyzer using LangGraph

An AI-powered Meeting Notes Analyzer that converts unstructured meeting transcripts into structured, actionable insights using a multi-agent **LangGraph** workflow powered by **Google Gemini (`gemini-3.5-flash-lite`)**.

🚀 **Live Streamlit Web App:** <https://ai-meeting-notes-analyzer-langgraph.streamlit.app/>

## 📌 Project Overview

Companies conduct countless meetings daily, but capturing key decisions and action items manually is slow and prone to errors. This project automates post-meeting documentation by routing transcripts through a 7-step multi-agent LangGraph workflow:

1. **Input Node**: Receives and validates the meeting transcript.
2. **Topic Extraction Agent**: Identifies main discussion topics.
3. **Meeting Summary Agent**: Generates a concise 3–5 sentence meeting summary.
4. **Action Item Extraction Agent**: Extracts tasks and owners (labeling unspecified owners as `'Owner: Not specified'`).
5. **Priority Classification Agent**: Classifies task urgency (`High Priority`, `Medium Priority`, `Low Priority`) based on timelines.
6. **Conditional Logic**: Automatically skips priority classification if no action items are detected.
7. **Final Output Node**: Compiles all outputs into a structured meeting report.

## 🛠️ Project File Structure

* `project.ipynb`: Jupyter notebook containing step-by-step implementation, testing, and graph compilation.
* `app.py`: Interactive Streamlit web application with user API key input.
* `requirements.txt`: Python package dependencies.
* `.env`: Environment variable configuration file.

## ⚙️ Setup and Installation Instructions

1. **Clone or Download** this repository into your local machine or VS Code.

2. **Install Dependencies**: You can use either standard `pip` or Astral's fast `uv` package manager.

   * **Option 1: Using `uv` (Recommended for speed)**
     ```bash
     uv sync
     ```

   * **Option 2: Using standard `pip`**
     ```bash
     pip install -r requirements.txt
     ```

3. **Get a Gemini API Key**:
   Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to generate your free Gemini API key.

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## 🚀 Running the Project

### Option A: Run the Jupyter Notebook
Open `project.ipynb` in VS Code or Jupyter Notebook, select your Python kernel, and run all cells sequentially to test the multi-agent graph with sample transcripts.

### Option B: Run the Streamlit Web App Locally
In your terminal, launch the Streamlit app:
```bash
streamlit run app.py
```
Then paste your Gemini API key in the app interface, enter your meeting transcript, and click **Analyze**!