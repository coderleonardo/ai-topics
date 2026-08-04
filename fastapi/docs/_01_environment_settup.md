# Environment setup

## 1. pipx

`pipx` installs tools securely in the global environment.

```bash
pip install --user pipx
pipx ensurepath  # let the system know about the tools installed via pipx
```

## 2. uv

We use `uv` for environment management.

```bash
uv python install 3.12
```

See more in https://docs.astral.sh/uv/guides/projects/#creating-a-new-project

## 3. Running the app

Start fastapi in development mode:

```bash
uv run fastapi dev src/project/env_settup/app.py
```

The app is served at `http://127.0.0.1:8000`:

- `http`: default protocol
- `127.0.0.1`: IP used
- `8000`: reserved port to the application (in our machine)

Note that fastAPI uses `uvicorn` to act as a server to disponibilize the fastAPI app to the network.

## 4. Development tools

```bash
uv add --group dev pytest pytest-cov ruff typos poethepoet
```

### ruff

1. check python programm good practices (linter)
2. formatter: pre-define a style python code to be followed

### poe the poet

The pattern is `uv run poe <task>`. The tasks are defined in `pyproject.toml` under `[tool.poe.tasks]`:

```bash
uv run poe lint      # ruff check
uv run poe format    # ruff check --fix, then ruff format
uv run poe serve     # fastapi dev src/project/env_settup/app.py
uv run poe test      # lint -> pytest -> coverage html
```

`uv run poe` with no task name lists them all.

Two shortcuts to avoid the typing:

- Activate the venv (`source .venv/bin/activate`), then just `poe lint`.
- Alias it: `alias poe='uv run poe'` in `.bashrc`.

Note that `uv run lint` would only work if `lint` were a console script under `[project.scripts]` — it isn't, and shouldn't be. Task runner and entry points are separate systems.

### coverage

```bash
uv run coverage html --show-contexts
```

## 5. AAA: Arrange, Act, Assert

Test structure used in `tests/`:

- **Arrange**: set up the object under test
- **Act**: exercise it (the SUT — system under test)
- **Assert**: check the result
