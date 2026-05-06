# Push to GitHub

The project is fully built and tested (26/26 tests pass).
Run these commands to create the repo and push:

```bash
cd agent-flow-scheduler

# 1. Create a new repo on GitHub (requires gh CLI authenticated)
gh repo create agent-flow-scheduler \
  --public \
  --description "Workflow-aware SRPT scheduling for multi-agent LLM pipelines. Inspired by Pythia (arxiv 2604.25899)" \
  --push \
  --source .

# --- OR manually ---

# 2. Init git and push to an existing empty GitHub repo
git init -b main
git add -A
git commit -m "feat: initial commit — agent-flow-scheduler v0.1.0"
git remote add origin https://github.com/YOUR_USERNAME/agent-flow-scheduler.git
git push -u origin main
```

To enable GitHub Actions CI, create `.github/workflows/test.yml`.
