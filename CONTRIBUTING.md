# Contributing Guidelines

## Git Workflow

### Making Commits

We follow a commit message convention for clarity:

```
<type>: <brief description>

<detailed explanation if needed>
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Feature addition
git commit -m "feat: Add domain adaptation module with adversarial training"

# Bug fix
git commit -m "fix: Correct SST-2 label mapping issue"

# Documentation
git commit -m "docs: Update README with training instructions"

# Model updates
git commit -m "feat: Add ELECTRA model implementation"

# Results
git commit -m "docs: Add experiment results for BERT on IMDB"
```

### Regular Commit Workflow

1. **Before starting work:**
   ```bash
   git pull origin main
   ```

2. **After making changes:**
   ```bash
   git status                    # Check what changed
   git add <files>               # Stage specific files
   git commit -m "message"       # Commit with descriptive message
   git push origin main          # Push to remote
   ```

3. **Commit frequently:**
   - After implementing a feature
   - After fixing a bug
   - After running experiments
   - After updating documentation

### Branch Strategy (Optional)

For larger features, consider creating a branch:
```bash
git checkout -b feature/domain-adaptation
# Make changes
git commit -m "feat: Implement domain adaptation"
git push origin feature/domain-adaptation
# Then create a pull request on GitHub
```

### Best Practices

- ✅ Commit small, logical changes
- ✅ Write clear, descriptive commit messages
- ✅ Pull before pushing to avoid conflicts
- ✅ Don't commit large data files (they're in .gitignore)
- ✅ Don't commit model checkpoints (they're in .gitignore)
- ✅ Review `git status` before committing

