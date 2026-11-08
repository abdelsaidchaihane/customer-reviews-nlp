import sys
from pathlib import Path

# Permet de lancer pytest sans installer le paquet
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
