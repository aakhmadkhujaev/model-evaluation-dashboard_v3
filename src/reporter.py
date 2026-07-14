from pathlib import Path

from typing import Any

from src.config import (
    REPORTS_DIR,
    APP_NAME,
    VERSION,
    REPORT_WIDTH,
    HEADER_SEPARATOR,
    SECTION_SEPARATOR,
)

from src.utils import get_timestamp

def _build_report(
    task_type: str,
    model_name: str,
    metrics: dict[str, float],
) -> str:
    """
    Build a formatted evaluation report.

    Args:
        task_type: The machine learning task type.
        model_name: The name of the trained model.
        metrics: The evaluation metrics.

    Returns:
        A formatted report as a string.
    """
    lines = []

    # Header
    lines.append(HEADER_SEPARATOR)
    lines.append(f"{APP_NAME:^{REPORT_WIDTH}}")
    lines.append(HEADER_SEPARATOR)
    lines.append("")

    lines.append(f"Version   : {VERSION}")
    lines.append(f"Generated : {get_timestamp()}")
    lines.append("")

    # Model Information
    lines.append(SECTION_SEPARATOR)
    lines.append("Model Information")
    lines.append(SECTION_SEPARATOR)
    lines.append("")

    lines.append(f"Task Type : {task_type}")
    lines.append(f"Model     : {model_name}")
    lines.append("")

    # Evaluation Metrics
    lines.append(SECTION_SEPARATOR)
    lines.append("Evaluation Metrics")
    lines.append(SECTION_SEPARATOR)
    lines.append("")
    for metric_name, metric_value in metrics.items():
        lines.append(f"{metric_name:<20}: {metric_value:.4f}")
    return "\n".join(lines)
def _save_report(report: str, filename: str) -> Path:
    """
    Save the report to a file.

    Args:
        report: The report content as a string.
        filename: The name of the report file.

    Returns:
        The path to the saved report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{filename}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path

def generate_report(
    task_type: str,
    model_name: str,
    metrics: dict[str, float],
) -> tuple[str, Path]:
    """
    Generate and save an evaluation report.

    Builds a formatted evaluation report, saves it to the
    reports directory, and returns both the report text
    and the report file path.

    Args:
        task_type: The machine learning task type.
        model_name: The name of the trained model.
        metrics: The evaluation metrics.

    Returns:
        A tuple containing:
            - The formatted report.
            - The path to the saved report.
    """
    report = _build_report(task_type, model_name, metrics)
    filename = f"{model_name.replace(' ', '_')}_{get_timestamp()}"
    report_path = _save_report(report, filename)
    return report, report_path

