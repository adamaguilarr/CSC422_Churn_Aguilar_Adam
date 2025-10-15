import os
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "Telco-Customer-Churn.csv"

def has_kaggle():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        return True
    except Exception:
        return False

def main():
    if OUT.exists():
        print(f"Already exists: {OUT}")
        return
    if not has_kaggle():
        print("Kaggle CLI not available. Please install and configure Kaggle, or download manually.")
        print("Manual: download CSV from Kaggle and place at data/Telco-Customer-Churn.csv")
        return
    print("Downloading with Kaggle CLI...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "blastchar/telco-customer-churn",
        "-p", str(DATA_DIR)
    ], check=True)
    # unzip if needed
    for f in DATA_DIR.glob("*.zip"):
        subprocess.run(["python", "-m", "zipfile", "-e", str(f), str(DATA_DIR)], check=True)
        f.unlink(missing_ok=True)
    # try to find CSV
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        print("Download finished but no CSV found. Please check the Kaggle archive contents.")
        return
    # Prefer the main CSV
    for c in candidates:
        if "Telco" in c.name or "Churn" in c.name:
            c.rename(OUT)
            break
    print(f"Ready: {OUT}")

if __name__ == "__main__":
    main()
