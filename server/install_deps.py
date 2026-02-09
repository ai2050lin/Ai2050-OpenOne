import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import sklearn
    print("scikit-learn is already installed.")
except ImportError:
    print("scikit-learn not found. Installing...")
    install("scikit-learn")
    print("scikit-learn installed successfully.")
