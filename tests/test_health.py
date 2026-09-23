import unittest

from judge.routers.health import healthz


class HealthRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_ok(self) -> None:
        self.assertEqual(await healthz(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
