try:
    import app
    print("SUCCESS: Flask app imported without errors")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
