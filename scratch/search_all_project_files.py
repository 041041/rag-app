# scratch/search_all_project_files.py
import os
import re

search_dir = "/Users/sandeep/.gemini/antigravity/scratch/clinical_rag_app"
targets = [
    "Analyze User Input",
    "Draft - Short Definition",
    "Draft Response",
    "Mental Refinement",
    "Check Constraints",
    "Context Provided",
    "Extract Information from Context",
    "Provided 8 documents"
]

print("Starting scan of all files in:", search_dir)
count = 0
for root, dirs, files in os.walk(search_dir):
    # Do not skip any directories (except maybe .git to prevent binary match spam)
    if ".git" in dirs:
        dirs.remove(".git")
    for file in files:
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for target in targets:
                # Use word boundaries or case-insensitive search
                if re.search(re.escape(target), content, re.IGNORECASE):
                    print(f"MATCH: Found '{target}' in file '{filepath}'")
                    count += 1
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

print(f"Scan complete. Total matches: {count}")
