"""Exercise the full-server research worker lifecycle without loading model modules."""
import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Lifecycle(unittest.TestCase):
    def run_case(self,available=True,fail=False):
        events=[]
        @asynccontextmanager
        async def legacy_lifespan(app):
            events.append('legacy-start')
            try:yield
            finally:events.append('legacy-stop')
        source=Path(__file__).resolve().parents[2]/'server/server.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')) .body if isinstance(n,ast.AsyncFunctionDef) and n.name=='workbench_lifespan')
        scope={'FastAPI':object,'asynccontextmanager':asynccontextmanager,'platform_router':object() if available else None,'lifespan':legacy_lifespan}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        service=SimpleNamespace(start_refresher=lambda:events.append('worker-start'),stop_refresher=lambda:events.append('worker-stop'))
        async def run():
            async with scope['workbench_lifespan'](object()):
                events.append('request')
                if fail:raise ValueError('request failed')
        with patch.dict(sys.modules,{'platform_service':service}):
            if fail:
                with self.assertRaises(ValueError):asyncio.run(run())
            else:asyncio.run(run())
        return events

    def test_worker_starts_and_stops_with_full_server(self):
        self.assertEqual(self.run_case(),['worker-start','legacy-start','request','legacy-stop','worker-stop'])
    def test_worker_closes_on_failure(self):
        self.assertEqual(self.run_case(fail=True),['worker-start','legacy-start','request','legacy-stop','worker-stop'])
    def test_backend_without_platform_keeps_legacy_lifespan(self):
        self.assertEqual(self.run_case(available=False),['legacy-start','request','legacy-stop'])


if __name__=='__main__':unittest.main()
