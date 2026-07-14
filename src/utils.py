from src.config import (FIGURES_DIR, HEADER_WIDTH, MODELS_DIR, REPORTS_DIR)
from datetime import datetime

def print_header(title: str) -> None:
    print(f"\n{'=' * HEADER_WIDTH}")
    print(f"{title:^{HEADER_WIDTH}}")
    print(f"{'=' * HEADER_WIDTH}")

def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))

def create_output_directories() -> None:
    for directory in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")