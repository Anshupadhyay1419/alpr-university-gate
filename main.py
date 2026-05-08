"""ALPR University Gate — Entry Point"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.run_pipeline import main
if __name__ == "__main__":
    main()
