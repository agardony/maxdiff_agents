#!/bin/bash
# Helper script to check what files are being ignored by git

echo "=== Files being tracked by Git ==="
git ls-files

echo -e "\n=== Files being ignored by Git (in current directory) ==="
git status --ignored --porcelain | grep '^!!' | cut -c4-

echo -e "\n=== Testing common files that should be ignored ==="
test_files=(".env" ".venv/" "__pycache__/" "*.pyc" "*.log" "dist/" "build/")

for file in "${test_files[@]}"; do
    if git check-ignore "$file" &>/dev/null; then
        echo "✅ $file - IGNORED (good)"
    else
        echo "❌ $file - NOT IGNORED (check .gitignore)"
    fi
done

echo -e "\n=== .env.example should be tracked ==="
if git ls-files | grep -q ".env.example"; then
    echo "✅ .env.example is tracked (good)"
else
    echo "❌ .env.example is not tracked (should be tracked)"
fi

