import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str = "llm_suite", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, robust logger for CLI outputs."""
    logger_inst = logging.getLogger(name)
    logger_inst.setLevel(level)

    if not logger_inst.handlers:
        try:
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                markup=False
            )
        except Exception:
            console_handler = logging.StreamHandler(sys.stdout)
            
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        console_handler.setFormatter(formatter)
        logger_inst.addHandler(console_handler)

    return logger_inst

logger = setup_logger()
