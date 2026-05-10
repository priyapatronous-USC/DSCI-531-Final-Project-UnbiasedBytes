import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation_pipeline import run_full_pipeline

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    run_full_pipeline(output_dir=output_dir)




