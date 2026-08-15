"""
Helper CLI Script to run the Enterprise Contract Intelligence Platform Backend.
"""
import uvicorn
import os
import sys

# Ensure root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
