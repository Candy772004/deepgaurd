"""
DeepGuard — Multi-Modal Deepfake Detection Desktop App
Run:  python main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import DeepGuardApp

if __name__ == "__main__":
    app = DeepGuardApp()
    app.mainloop()
