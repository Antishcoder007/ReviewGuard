import os

for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "__pycache__":
            full_path = os.path.join(root, d)
            print("🧹 Removing:", full_path)
            os.system(f'rmdir /s /q "{full_path}"')