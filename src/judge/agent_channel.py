import asyncio

from judge.models import JobOffer, JobReceipt


class AgentUnavailable(RuntimeError):
    pass


class AgentPollConflict(RuntimeError):
    pass


class UnknownOffer(RuntimeError):
    pass


class AgentChannel:
    """Match outbound sub-judge polls with master job offers and receipts."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._polls: dict[str, asyncio.Future[JobOffer]] = {}
        self._receipts: dict[str, tuple[str, asyncio.Future[JobReceipt]]] = {}

    async def available_ids(self) -> set[str]:
        async with self._lock:
            return set(self._polls)

    async def wait_for_offer(
        self, judge_id: str, *, timeout_seconds: float = 20
    ) -> JobOffer | None:
        """Hold one outbound poll until an offer arrives or the poll expires."""

        async with self._lock:
            if judge_id in self._polls:
                raise AgentPollConflict(f"Sub-judge {judge_id!r} already has a poll")
            future: asyncio.Future[JobOffer] = (
                asyncio.get_running_loop().create_future()
            )
            self._polls[judge_id] = future

        try:
            return await asyncio.wait_for(future, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            async with self._lock:
                if self._polls.get(judge_id) is future:
                    del self._polls[judge_id]

    async def offer(
        self, judge_id: str, offer: JobOffer, *, timeout_seconds: float = 60
    ) -> JobReceipt:
        """Send a job only to a currently polling sub-judge, then await its receipt."""

        async with self._lock:
            poll = self._polls.pop(judge_id, None)
            if poll is None or poll.done():
                raise AgentUnavailable(f"Sub-judge {judge_id!r} is not connected")
            if offer.job_id in self._receipts:
                self._polls[judge_id] = poll
                raise RuntimeError(f"Job {offer.job_id!r} already has a pending offer")

            receipt: asyncio.Future[JobReceipt] = (
                asyncio.get_running_loop().create_future()
            )
            self._receipts[offer.job_id] = (judge_id, receipt)
            poll.set_result(offer)

        try:
            return await asyncio.wait_for(receipt, timeout_seconds)
        except TimeoutError as error:
            raise AgentUnavailable(
                f"Sub-judge {judge_id!r} did not acknowledge job {offer.job_id!r}"
            ) from error
        finally:
            async with self._lock:
                if self._receipts.get(offer.job_id) == (judge_id, receipt):
                    del self._receipts[offer.job_id]

    async def acknowledge(
        self, judge_id: str, job_id: str, receipt: JobReceipt
    ) -> None:
        async with self._lock:
            pending = self._receipts.get(job_id)
            if pending is None or pending[0] != judge_id:
                raise UnknownOffer(f"No pending offer for job {job_id!r}")
            future = pending[1]
            if future.cancelled():
                raise UnknownOffer(f"No pending offer for job {job_id!r}")
            if future.done():
                if future.result() != receipt:
                    raise ValueError(f"Conflicting receipt for job {job_id!r}")
                return
            future.set_result(receipt)
