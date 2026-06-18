import sys
import os

# Add the parent directory to sys.path so that the 'src' package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.main import app
