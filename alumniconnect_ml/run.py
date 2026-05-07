"""
AlumniConnect ML — Entry Point
Usage:
    python run.py train     # Train all models
    python run.py serve     # Start the API server
    python run.py all       # Train then serve
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|serve|all]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ("train", "all"):
        from training.train_all import train_all
        train_all()

    if command in ("serve", "all"):
        import uvicorn
        from config import API_HOST, API_PORT
        from api.routes import app

        print(f"\nStarting ML service on http://{API_HOST}:{API_PORT}")
        print("Docs at http://localhost:8000/docs\n")
        uvicorn.run(app, host=API_HOST, port=API_PORT)

    if command not in ("train", "serve", "all"):
        print(f"Unknown command: {command}")
        print("Usage: python run.py [train|serve|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
