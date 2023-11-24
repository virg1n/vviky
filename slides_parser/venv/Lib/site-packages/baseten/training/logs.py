import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

import requests
from halo import Halo
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from baseten.common.api import get_training_logs

# Needed to avoid circular dependencies
if TYPE_CHECKING:
    from baseten.training.finetuning import FinetuningRun


# These two parameters are used for the backend query.
# Required for correctness and performance.
# Defaults assume that no more than 1000 lines are emitted in 5 minutes of training.
# TODO: make settable by client for more verbose jobs.
_TRAINING_LOG_QUERY_TIMESTEP = timedelta(minutes=5)
_TRAINING_LOG_QUERY_LIMIT = 1000

# Constants for streaming logs
_PENDING_STATE_DELAY_SECONDS = 5.0
_RUNNING_STATE_DELAY_SECONDS = 2.0


logger = logging.getLogger(__name__)


@contextmanager
def retry_network_errors(num_retries: int):
    for attempt in Retrying(
        reraise=True,
        stop=stop_after_attempt(num_retries),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    ):
        with attempt:
            yield


@dataclass
class LogLine:
    """
    Wrapper for log lines.
    """

    ts: int
    msg: str
    # level is optional since it's currently not handle on the server side
    level: Optional[str]

    def console_stream(self):
        # TODO: update based on level
        logger.info(self.msg)


def _typed_and_ordered_training_logs(
    training_run_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[LogLine]:
    inner_logs = get_training_logs(training_run_id, start, end, limit=limit)
    return sorted([LogLine(**d) for d in inner_logs], key=lambda logline: int(logline.ts))


def _dt_millis_str(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1000))


def _dt_from_nanos_str(timestamp: int) -> datetime:
    return datetime.fromtimestamp(float(timestamp) / 1e9)


class TrainingLogsConsumer:
    def __init__(self, run: "FinetuningRun") -> None:
        self._run = run
        self._entity_id: str = run.id
        self._refresh()

    def _refresh(self):
        self._run.refresh()
        self._start_time: Optional[datetime] = (
            datetime.fromisoformat(self._run.started) if self._run.started else None
        )
        self._stop_time: Optional[datetime] = (
            datetime.fromisoformat(self._run.stopped) if self._run.stopped else None
        )

    def _get_lastest_time(self) -> datetime:
        return (
            self._stop_time if self._stop_time else datetime.utcnow().replace(tzinfo=timezone.utc)
        )

    def _consume_logs_period(
        self, start_time: datetime, end_time: datetime, limit: Optional[int] = None
    ) -> List[LogLine]:
        if start_time is None:
            return []

        query_start = start_time
        query_end = min(start_time + _TRAINING_LOG_QUERY_TIMESTEP, end_time)

        if start_time > end_time:
            return []
        logs = []
        while query_end <= end_time:
            logs.extend(
                _typed_and_ordered_training_logs(
                    self._entity_id,
                    start=_dt_millis_str(query_start),
                    end=_dt_millis_str(query_end),
                    limit=limit or _TRAINING_LOG_QUERY_LIMIT,
                )
            )

            # Range is inclusive so we have to increment up slightly
            # NOTE: this is slightly error prone due to the discrepency of
            #       loki working in nanos and graphql taking millis
            query_start = query_end + timedelta(milliseconds=1)
            query_end = query_start + _TRAINING_LOG_QUERY_TIMESTEP
        return logs

    def _get_next_start_time(self, last_logs: List[LogLine], last_start_time: datetime) -> datetime:
        if len(last_logs) > 0:
            # Range is inclusive so we have to increment up slightly
            # NOTE: this is slightly error prone due to the discrepency of
            #       loki working in nanos and graphql taking millis
            start_time = _dt_from_nanos_str(last_logs[-1].ts) + timedelta(milliseconds=1)
        else:
            start_time = last_start_time.replace(tzinfo=timezone.utc)

        return start_time.replace(tzinfo=timezone.utc)

    def stream_logs(self):
        """
        Function to get a stream of logs for a given a FinetuningRun.
        """
        if self._run.is_pending:
            spinner = Halo(text="Waiting for finetuning job to start", spinner="dots")
            spinner.start()

            while self._run.is_pending or self._start_time is None:
                try:
                    with retry_network_errors(5):
                        time.sleep(_PENDING_STATE_DELAY_SECONDS)
                        self._refresh()
                except requests.exceptions.RequestException:
                    Halo().info("Something went wrong with your connection to Baseten.")
                    return

            spinner.succeed("Finetuning job successully started running.")

        Halo().info("Starting to stream logs")

        with retry_network_errors(5):
            retrieved_logs = self._consume_logs_period(self._start_time, self._get_lastest_time())
            period_start_time = self._get_next_start_time(retrieved_logs, self._start_time)
            for line in retrieved_logs:
                line.console_stream()

        while self._run.is_running:
            with retry_network_errors(5):
                try:
                    retrieved_logs = self._consume_logs_period(
                        period_start_time, self._get_lastest_time()
                    )
                    period_start_time = self._get_next_start_time(retrieved_logs, period_start_time)
                    for line in retrieved_logs:
                        line.console_stream()

                    time.sleep(_RUNNING_STATE_DELAY_SECONDS)
                    self._refresh()
                except requests.exceptions.RequestException:
                    Halo().info("Something went wrong with your connection to Baseten.")
                    return

        for line in self._consume_logs_period(period_start_time, self._get_lastest_time()):
            line.console_stream()

        if self._run.is_succeeded:
            Halo().succeed("Finetuning job completed successfully.")
        elif self._run.is_failed:
            Halo().fail("Finetuning job failed.")
        elif self._run.is_cancelled:
            Halo().warn("Finetuning job was cancelled.")
