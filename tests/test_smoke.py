import sys
import importlib

def test_imports():
    """
    Basic smoke test — verifies that all main project modules import correctly.
    """
    modules = [
        "src.data",
        "src.features",
        "src.models",
        "src.metrics"
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            print(f"[OK] Imported: {m}")
        except Exception as e:
            print(f"[FAIL] Error importing {m}: {e}")
            sys.exit(1)
    print("✅ All modules imported successfully.")

if __name__ == "__main__":
    test_imports()
