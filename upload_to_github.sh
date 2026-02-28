#!/usr/bin/env bash
# upload_to_github.sh - Upload ai-fpga-loop to GitHub using the gh CLI
set -euo pipefail

REPO_NAME="ai-fpga-loop"
REPO_DESC="AI agent that autonomously designs FPGA inference hardware in a continuous optimization loop"

# ── Check for gh CLI ──────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo ""
    echo "ERROR: GitHub CLI (gh) is not installed."
    echo ""
    echo "Install instructions:"
    echo "  macOS  : brew install gh"
    echo "  Ubuntu : sudo apt install gh"
    echo "           OR: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
    echo "           sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg"
    echo "           echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null"
    echo "           sudo apt update && sudo apt install gh"
    echo "  Windows: winget install --id GitHub.cli"
    echo "           OR: scoop install gh"
    echo ""
    echo "After installing, authenticate with: gh auth login"
    exit 1
fi

# ── Check auth ────────────────────────────────────────────────────────────────
echo "Checking GitHub authentication..."
if ! gh auth status &>/dev/null; then
    echo "ERROR: Not authenticated with GitHub. Run: gh auth login"
    exit 1
fi
echo "Authenticated OK."

# ── Create .gitignore ─────────────────────────────────────────────────────────
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
build/*.v
logs/*.json
venv/
.venv/
.DS_Store
*.egg-info/
dist/
*.out
*.vcd
EOF

# ── Git init & commit ─────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    echo "Initialising git repository..."
    git init
    git checkout -b main 2>/dev/null || git branch -M main
fi

echo "Staging files..."
git add \
    main_loop.py \
    design_agent.py \
    verilog_gen.py \
    simulator.py \
    synthesizer.py \
    reward.py \
    README.md \
    upload_to_github.sh \
    .gitignore \
    LICENSE 2>/dev/null || true

# Add empty dirs via placeholder files if needed
mkdir -p build logs
for d in build logs; do
    if [ ! -f "$d/.gitkeep" ]; then
        touch "$d/.gitkeep"
        git add "$d/.gitkeep" 2>/dev/null || true
    fi
done

git commit -m "feat: AI-FPGA-Loop proof-of-concept

Autonomous AI agent that designs FPGA dot-product units using
tabular Q-learning, generates synthesizable Verilog, evaluates
performance via Python simulation or real Yosys synthesis,
and iterates to improve OPS/LUT efficiency.

Design space: 648 configurations across 6 parameters.
Supports --mode sim (pure Python) and --mode full (Yosys+iverilog)." \
    2>/dev/null || echo "(nothing new to commit)"

# ── Create or use existing GitHub repo ────────────────────────────────────────
echo "Creating GitHub repository: $REPO_NAME ..."
if gh repo create "$REPO_NAME" \
       --public \
       --description "$REPO_DESC" \
       --source=. \
       --remote=origin \
       --push 2>/dev/null; then
    echo "Repository created and pushed."
else
    echo "Repository may already exist — attempting push to existing repo..."
    GITHUB_USER=$(gh api user --jq '.login')
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    git push -u origin main
    echo "Pushed to existing repository."
fi

echo ""
GITHUB_USER=$(gh api user --jq '.login' 2>/dev/null || echo "YOUR_USERNAME")
echo "Done! Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
