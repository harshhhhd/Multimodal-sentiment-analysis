"""
Entry point — run the IndicSent-M Flask web application.

Usage:
    python run.py
"""

from webapp.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
