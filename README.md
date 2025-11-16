## lerobot-playground

This repository uses the [`huggingface/lerobot`](https://github.com/huggingface/lerobot) project as a Git submodule for experimentation. Make sure to initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Once the submodule is checked out you can work with the playground scripts (for example `python main.py`) while having direct access to the lerobot source under `lerobot/`.

### Updating the lerobot submodule

To pull the latest lerobot revisions referenced by this playground:

```bash
git submodule update --remote lerobot
```

If you want to update to a specific commit, check out that commit inside `lerobot/` and record it with `git add lerobot`.

### Installing lerobot into the current environment (with `uv`)

1. Create a project-local virtual environment managed by [`uv`](https://docs.astral.sh/uv/):

   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install the lerobot submodule in editable mode so local changes under `lerobot/` are instantly available:

   ```bash
   uv pip install -e ./lerobot
   ```

3. (Optional) install the playground helpers too:

   ```bash
   uv pip install -e .
   ```
