"""DataForge Configuration Constants"""
import os
from dotenv import load_dotenv

# Set UTF-8 encoding for console output (Fix for UnicodeEncodeError)
os.environ['PYTHONIOENCODING'] = 'utf-8'
PYTHONIOENCODING = 'utf-8'

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Data Processing
RANDOM_SEED = 42
MAX_ROWS = 200_000
MAX_CATS_UNIQUE = 50
SHAP_SAMPLE_SIZE = 500

# Logging
LOG_MAX_BYTES = 5_242_880  # 5MB
LOG_BACKUP_COUNT = 3

# Plotting
PLOT_DPI = 100
PLOT_FIGSIZE = (6, 4)

# Random Forest Defaults
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS = 1
