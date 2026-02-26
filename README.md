# LangGraph Checkpoint Greenplum

Greenplum/MPP-optimized checkpoint savers for LangGraph. This package provides drop-in replacements for `PostgresSaver` and `AsyncPostgresSaver` that use a CTE-based list query instead of correlated subqueries, avoiding Broadcast Motion in Greenplum's MPP execution engine.

## Why use this?

On Greenplum (and similar MPP databases), the default list query in `langgraph-checkpoint-postgres` uses correlated subqueries. The planner may broadcast the full inner tables to every segment, so list operations can be very slow on large datasets. The Greenplum savers use a CTE + JOIN pattern so the planner uses Redistribute Motion instead, often yielding orders-of-magnitude speedups for `list()` / `alist()`.

## Installation

Install the library and its dependency:

```bash
pip install langgraph-checkpoint-greenplum
```

This will pull in `langgraph-checkpoint-postgres` as well.

## Usage

Use `AsyncGreenplumSaver` or `GreenplumSaver` exactly like the Postgres savers. Call `.setup()` the first time to create tables.

### Async

```python
from langgraph.checkpoint.greenplum import AsyncGreenplumSaver

DB_URI = "postgres://user:pass@host:5432/dbname"
async with AsyncGreenplumSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    # use as a drop-in replacement for AsyncPostgresSaver
```

### Sync

```python
from langgraph.checkpoint.greenplum import GreenplumSaver

DB_URI = "postgres://user:pass@host:5432/dbname"
with GreenplumSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    # use as a drop-in replacement for PostgresSaver
```

## Requirements

- Greenplum 6+ (or PostgreSQL 9.3+) for the CTE query (uses `CROSS JOIN LATERAL`).
- `langgraph-checkpoint-postgres` (and thus `psycopg` and `langgraph-checkpoint`) are installed automatically.

## Release to PyPI (GitHub Actions)

This repository includes a publish workflow at `.github/workflows/publish-pypi.yml`.

### One-time setup

1. In PyPI, configure a Trusted Publisher for this project:
   - Owner: `StarJourneyMingsing`
   - Repository: `langgraph-checkpoint-greenplum`
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi`
2. In GitHub, create environment `pypi` in repository settings (optional protection rules can be added).

### Publish a new version

1. Update `version` in `pyproject.toml`.
2. Commit and push changes.
3. Create and push a version tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

The workflow will verify that tag version matches `pyproject.toml`, build distributions, and publish to PyPI.

## License

MIT
