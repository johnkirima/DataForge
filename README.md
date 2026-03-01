# DataForge

**Personal Data Science Automation Tool**

A terminal-based Python tool that automates common data science workflows using a 9-agent pipeline powered by LLMs.

## Tech Stack

- **DeepSeek V3.2** - Narrative generation (cost-effective)
- **Claude Sonnet 4.6** - Reasoning and judgment tasks
- **Budget**: $10-15/month API costs

## Project Structure

```
dataforge/
├── agents/           # Agent implementations (Phase 2)
│   └── __init__.py
├── logs/             # Pipeline logs
├── reports/          # Generated reports
├── uploads/          # User dataset uploads
├── config.py         # Configuration constants
├── pipeline_context.py  # Shared pipeline state
├── logger.py         # Logging setup
├── main.py           # Entry point
├── requirements.txt  # Dependencies
├── README.md         # This file
└── .env              # API keys (not in git)
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Edit the `.env` file and add your API keys:

```
ANTHROPIC_API_KEY=your_actual_key_here
OPENAI_API_KEY=your_actual_key_here
DEEPSEEK_API_KEY=your_actual_key_here
```

### 5. Run DataForge

```bash
python main.py
```

## Phase 1 Verification

After setup, verify everything works:

1. ✅ `pip install -r requirements.txt` completes without errors
2. ✅ `python main.py` prints the banner with no import errors
3. ✅ `logs/pipeline.log` is created

## Performance Tip (Windows)

**Exclude the `dataforge/` folder from Windows Defender real-time scanning** for better performance:

1. Open Windows Security
2. Go to Virus & threat protection → Manage settings
3. Scroll to Exclusions → Add or remove exclusions
4. Add the `dataforge/` folder path

## Current Status

- **Phase 1**: ✅ Project skeleton complete
- **Phase 2**: 🔜 Agent implementations (coming next)

## License

Personal use project.
