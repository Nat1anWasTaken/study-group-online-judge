import asyncio
import unittest

from judge.agent_channel import (
    AgentChannel,
    AgentPollConflict,
    AgentUnavailable,
    UnknownOffer,
)
from judge.models import JobOffer, JobReceipt, Resources, Submission


class AgentChannelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.channel = AgentChannel()
        self.offer = JobOffer(
            job_id="a" * 32,
            submission=Submission(
                repo_url="https://github.com/cerulean-works/example.git",
                commit_sha="b" * 40,
                task_id="lab1",
                github_actor="participant",
            ),
            resources=Resources(gpus=1),
        )

    async def test_offers_only_to_a_currently_polling_agent(self) -> None:
        with self.assertRaises(AgentUnavailable):
            await self.channel.offer("nano4", self.offer)

        poll = asyncio.create_task(
            self.channel.wait_for_offer("nano4", timeout_seconds=1)
        )
        await asyncio.sleep(0)
        self.assertEqual(await self.channel.available_ids(), {"nano4"})

        dispatch = asyncio.create_task(
            self.channel.offer("nano4", self.offer, timeout_seconds=1)
        )
        self.assertEqual(await poll, self.offer)
        self.assertEqual(await self.channel.available_ids(), set())

        receipt = JobReceipt(accepted=True, slurm_job_id="12345")
        await self.channel.acknowledge("nano4", self.offer.job_id, receipt)
        self.assertEqual(await dispatch, receipt)

    async def test_expired_poll_is_not_available(self) -> None:
        result = await self.channel.wait_for_offer("nano4", timeout_seconds=0.01)

        self.assertIsNone(result)
        self.assertEqual(await self.channel.available_ids(), set())
        with self.assertRaises(AgentUnavailable):
            await self.channel.offer("nano4", self.offer)

    async def test_rejects_a_second_poll_for_the_same_agent(self) -> None:
        poll = asyncio.create_task(
            self.channel.wait_for_offer("nano4", timeout_seconds=1)
        )
        await asyncio.sleep(0)

        with self.assertRaises(AgentPollConflict):
            await self.channel.wait_for_offer("nano4", timeout_seconds=1)

        poll.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await poll
        self.assertEqual(await self.channel.available_ids(), set())

    async def test_rejects_receipts_from_the_wrong_agent(self) -> None:
        poll = asyncio.create_task(
            self.channel.wait_for_offer("nano4", timeout_seconds=1)
        )
        await asyncio.sleep(0)
        dispatch = asyncio.create_task(
            self.channel.offer("nano4", self.offer, timeout_seconds=1)
        )
        await poll

        receipt = JobReceipt(accepted=True, slurm_job_id="12345")
        with self.assertRaises(UnknownOffer):
            await self.channel.acknowledge("other", self.offer.job_id, receipt)
        await self.channel.acknowledge("nano4", self.offer.job_id, receipt)
        self.assertEqual(await dispatch, receipt)

    async def test_unacknowledged_offer_times_out(self) -> None:
        poll = asyncio.create_task(
            self.channel.wait_for_offer("nano4", timeout_seconds=1)
        )
        await asyncio.sleep(0)
        dispatch = asyncio.create_task(
            self.channel.offer("nano4", self.offer, timeout_seconds=0.01)
        )
        await poll

        with self.assertRaises(AgentUnavailable):
            await dispatch
        with self.assertRaises(UnknownOffer):
            await self.channel.acknowledge(
                "nano4", self.offer.job_id, JobReceipt(accepted=True)
            )


if __name__ == "__main__":
    unittest.main()
