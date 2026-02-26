# type: ignore

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from langgraph.checkpoint.greenplum import AsyncGreenplumSaver
from tests.conftest import DEFAULT_POSTGRES_URI


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@asynccontextmanager
async def _greenplum_saver():
    """Fixture for AsyncGreenplumSaver (CTE-based list query)."""
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncGreenplumSaver(conn)
            await checkpointer.setup()
            yield checkpointer
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


async def test_asearch(test_data) -> None:
    """Test list/filter (CTE-based query) with AsyncGreenplumSaver."""
    async with _greenplum_saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        await saver.aput(configs[0], checkpoints[0], metadata[0], {})
        await saver.aput(configs[1], checkpoints[1], metadata[1], {})
        await saver.aput(configs[2], checkpoints[2], metadata[2], {})

        query_1 = {"source": "input"}
        query_2 = {"step": 1}
        query_3: dict[str, Any] = {}
        query_4 = {"source": "update", "step": 1}

        search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
        assert len(search_results_3) == 3

        search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
        assert len(search_results_4) == 0

        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
        ]
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


async def test_pending_sends_migration() -> None:
    """Test pending sends migration with AsyncGreenplumSaver (list applies migration)."""
    async with _greenplum_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        checkpoint_0 = empty_checkpoint()
        config = await saver.aput(config, checkpoint_0, {}, {})
        await saver.aput_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        await saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")

        tuple_0 = await saver.aget_tuple(config)
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = await saver.aput(config, checkpoint_1, {}, {})

        tuple_1 = await saver.aget_tuple(config)
        assert tuple_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in tuple_1.checkpoint["channel_versions"]

        search_results = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]
