"""Greenplum-optimized checkpoint savers.

Greenplum (and other MPP databases with similar planners) handle
correlated subqueries by broadcasting the entire inner table to all
segments (Broadcast Motion), which is extremely slow on large datasets.
These savers replace the default correlated subquery with a CTE + JOIN
approach that uses targeted Redistribute Motion instead.

Best suited for: Greenplum and other MPP databases that cannot
decorrelate subqueries and thus choose Broadcast Motion. Not all
PostgreSQL-based distributed databases have this behavior; use the
regular `PostgresSaver` / `AsyncPostgresSaver` when in doubt.

Requirements:
    Greenplum 6+ (based on PostgreSQL 9.4) or PostgreSQL 9.3+, since
    the CTE query uses `CROSS JOIN LATERAL`.

Usage::

    from langgraph.checkpoint.greenplum import AsyncGreenplumSaver

    async with AsyncGreenplumSaver.from_conn_string(DB_URI) as saver:
        await saver.setup()
        # use as a drop-in replacement for AsyncPostgresSaver
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple, get_checkpoint_id
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CTE_LIST_SELECT_SQL = """
WITH filtered_checkpoints AS (
    SELECT
        thread_id,
        checkpoint,
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        metadata
    FROM checkpoints
    {where}
    ORDER BY checkpoint_id DESC
    {limit}
),
checkpoint_data AS (
    SELECT
        fc.thread_id,
        fc.checkpoint,
        fc.checkpoint_ns,
        fc.checkpoint_id,
        fc.parent_checkpoint_id,
        fc.metadata,
        cv.key AS channel_key,
        cv.value AS channel_version
    FROM filtered_checkpoints fc
    CROSS JOIN LATERAL jsonb_each_text(fc.checkpoint -> 'channel_versions') AS cv(key, value)
),
channel_values AS (
    SELECT
        cd.thread_id,
        cd.checkpoint_id,
        cd.checkpoint_ns,
        array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) AS channel_values
    FROM checkpoint_data cd
    INNER JOIN checkpoint_blobs bl
        ON bl.thread_id = cd.thread_id
        AND bl.checkpoint_ns = cd.checkpoint_ns
        AND bl.channel = cd.channel_key
        AND bl.version = cd.channel_version
    GROUP BY cd.thread_id, cd.checkpoint_id, cd.checkpoint_ns
),
pending_writes AS (
    SELECT
        cw.thread_id,
        cw.checkpoint_ns,
        cw.checkpoint_id,
        array_agg(
            array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob]
            ORDER BY cw.task_id, cw.idx
        ) AS pending_writes
    FROM checkpoint_writes cw
    INNER JOIN filtered_checkpoints fc
        ON cw.thread_id = fc.thread_id
        AND cw.checkpoint_ns = fc.checkpoint_ns
        AND cw.checkpoint_id = fc.checkpoint_id
    GROUP BY cw.thread_id, cw.checkpoint_ns, cw.checkpoint_id
)
SELECT
    fc.thread_id,
    fc.checkpoint,
    fc.checkpoint_ns,
    fc.checkpoint_id,
    fc.parent_checkpoint_id,
    fc.metadata,
    cv.channel_values,
    pw.pending_writes
FROM filtered_checkpoints fc
LEFT JOIN channel_values cv
    ON fc.thread_id = cv.thread_id
    AND fc.checkpoint_id = cv.checkpoint_id
    AND fc.checkpoint_ns = cv.checkpoint_ns
LEFT JOIN pending_writes pw
    ON fc.thread_id = pw.thread_id
    AND fc.checkpoint_id = pw.checkpoint_id
    AND fc.checkpoint_ns = pw.checkpoint_ns
ORDER BY fc.checkpoint_id DESC
"""

# Single-checkpoint CTE for get_tuple/aget_tuple (same shape as list, at most one row).
CTE_GET_TUPLE_SQL = """
WITH filtered_checkpoints AS (
    SELECT
        thread_id,
        checkpoint,
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        metadata
    FROM checkpoints
    {where}
    {order_limit}
),
checkpoint_data AS (
    SELECT
        fc.thread_id,
        fc.checkpoint,
        fc.checkpoint_ns,
        fc.checkpoint_id,
        fc.parent_checkpoint_id,
        fc.metadata,
        cv.key AS channel_key,
        cv.value AS channel_version
    FROM filtered_checkpoints fc
    CROSS JOIN LATERAL jsonb_each_text(fc.checkpoint -> 'channel_versions') AS cv(key, value)
),
channel_values AS (
    SELECT
        cd.thread_id,
        cd.checkpoint_id,
        cd.checkpoint_ns,
        array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) AS channel_values
    FROM checkpoint_data cd
    INNER JOIN checkpoint_blobs bl
        ON bl.thread_id = cd.thread_id
        AND bl.checkpoint_ns = cd.checkpoint_ns
        AND bl.channel = cd.channel_key
        AND bl.version = cd.channel_version
    GROUP BY cd.thread_id, cd.checkpoint_id, cd.checkpoint_ns
),
pending_writes AS (
    SELECT
        cw.thread_id,
        cw.checkpoint_ns,
        cw.checkpoint_id,
        array_agg(
            array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob]
            ORDER BY cw.task_id, cw.idx
        ) AS pending_writes
    FROM checkpoint_writes cw
    INNER JOIN filtered_checkpoints fc
        ON cw.thread_id = fc.thread_id
        AND cw.checkpoint_ns = fc.checkpoint_ns
        AND cw.checkpoint_id = fc.checkpoint_id
    GROUP BY cw.thread_id, cw.checkpoint_ns, cw.checkpoint_id
)
SELECT
    fc.thread_id,
    fc.checkpoint,
    fc.checkpoint_ns,
    fc.checkpoint_id,
    fc.parent_checkpoint_id,
    fc.metadata,
    cv.channel_values,
    pw.pending_writes
FROM filtered_checkpoints fc
LEFT JOIN channel_values cv
    ON fc.thread_id = cv.thread_id
    AND fc.checkpoint_id = cv.checkpoint_id
    AND fc.checkpoint_ns = cv.checkpoint_ns
LEFT JOIN pending_writes pw
    ON fc.thread_id = pw.thread_id
    AND fc.checkpoint_id = pw.checkpoint_id
    AND fc.checkpoint_ns = pw.checkpoint_ns
"""


def _build_get_tuple_query(where: str, args: tuple[Any, ...]) -> tuple[str, list[Any]]:
    """Build the CTE-based SQL for fetching a single checkpoint (get_tuple/aget_tuple)."""
    if " ORDER BY" in where:
        where_clause = where.split(" ORDER BY")[0].strip() or " "
        order_limit = " ORDER BY checkpoint_id DESC LIMIT 1"
    else:
        where_clause = where.strip() or " "
        order_limit = ""
    query = CTE_GET_TUPLE_SQL.format(where=where_clause, order_limit=order_limit)
    return query, list(args)


class _GreenplumListMixin:
    """Mixin that switches list/get_tuple queries to the CTE-based approach."""

    def _build_list_query(
        self, where: str, args: list[Any], limit: int | None
    ) -> tuple[str, list[Any]]:
        params = list(args)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(int(limit))
        # Use a single space when where is empty so the CTE template produces
        # consistent SQL (no bare blank line between FROM and ORDER BY).
        normalized_where = where if where else " "
        query = CTE_LIST_SELECT_SQL.format(
            where=normalized_where, limit=limit_clause
        )
        return query, params


class AsyncGreenplumSaver(_GreenplumListMixin, AsyncPostgresSaver):
    """Async checkpoint saver optimized for Greenplum/MPP databases.

    Drop-in replacement for `AsyncPostgresSaver`. Uses CTE+JOIN for `alist()`
    and `aget_tuple()` instead of correlated subqueries, avoiding Broadcast
    Motion in Greenplum's MPP execution engine.
    """

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = (
                "WHERE thread_id = %s AND checkpoint_ns = %s "
                "ORDER BY checkpoint_id DESC LIMIT 1"
            )
        query, params = _build_get_tuple_query(where, args)
        async with self._cursor() as cur:
            await cur.execute(query, params, binary=True)
            value = await cur.fetchone()
            if value is None:
                return None
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                sends = await cur.fetchone()
                if sends:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )
            return await self._load_checkpoint_tuple(value)


class GreenplumSaver(_GreenplumListMixin, PostgresSaver):
    """Sync checkpoint saver optimized for Greenplum/MPP databases.

    Drop-in replacement for `PostgresSaver`. Uses CTE+JOIN for `list()` and
    `get_tuple()` instead of correlated subqueries, avoiding Broadcast
    Motion in Greenplum's MPP execution engine.
    """

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = (
                "WHERE thread_id = %s AND checkpoint_ns = %s "
                "ORDER BY checkpoint_id DESC LIMIT 1"
            )
        query, params = _build_get_tuple_query(where, args)
        with self._cursor() as cur:
            cur.execute(query, params)
            value = cur.fetchone()
            if value is None:
                return None
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                sends = cur.fetchone()
                if sends:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )
            return self._load_checkpoint_tuple(value)
