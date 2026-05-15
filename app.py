from dotenv import load_dotenv

load_dotenv()

# from src.conversationalAI import run_app
from src.ui.main import run_app


if __name__ == "__main__":
    run_app()