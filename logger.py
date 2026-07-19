from colorama import Fore, init
import logging
from pathlib import Path

init(autoreset=True)

LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIRECTORY / "bot.log"),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_info(message):
    print(Fore.GREEN + message)
    logging.info(message)

def log_warning(message):
    print(Fore.YELLOW + message)
    logging.warning(message)

def log_error(message):
    print(Fore.RED + message)
    logging.error(message)
