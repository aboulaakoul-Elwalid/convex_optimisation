#!/usr/bin/env python
"""
Run the optimization visualization backend server and launch the UI.

This starts the FastAPI server that powers the web UI and automatically
opens it in your default browser.
"""

import sys
import os
import webbrowser
import threading
import time

# Add the parent directory to the path so we can import optimizer_viz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer_viz.server import app
import uvicorn


def open_browser():
    """Open the UI in the default browser after a short delay."""
    time.sleep(2)  # Give the server time to start

    # Get the absolute path to the HTML file
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "convergence_analysis.html")

    if os.path.exists(html_file):
        file_url = f"file://{html_file}"
        print(f"\n📱 Opening UI in browser: {file_url}\n")
        webbrowser.open(file_url)
    else:
        print(f"\n⚠️  UI file not found at: {html_file}\n")
        print("Please open manually: web/convergence_analysis.html")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Optimizer Visualization - Backend & UI Launcher")
    print("=" * 70)
    print("")
    print("📊 Backend API: http://localhost:8000")
    print("📡 API endpoint: http://localhost:8000/simulate")
    print("")
    print("🌐 Opening UI in your default browser...")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print("")

    # Start browser opener in a background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Run the server (blocking)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
