import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests

import fetch_odds_api as odds


ATP = "tennis_atp_us_open"
WTA = "tennis_wta_us_open"
SPORTS = [{"key": ATP, "group": "Tennis"}]
EVENT = {
    "home_team": "Player One",
    "away_team": "Player Two",
    "commence_time": "2026-09-13T12:00:00Z",
    "bookmakers": [{
        "title": "Example bookmaker",
        "markets": [{
            "key": "h2h",
            "outcomes": [
                {"name": "Player One", "price": 2.0},
                {"name": "Player Two", "price": 1.8},
            ],
        }],
    }],
}


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "https://api.the-odds-api.com/v4/sports?apiKey=test-secret"
    result._content = json.dumps(data).encode()
    return result


class OddsSnapshotTests(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        data_dir = self.directory / "data_files"
        data_dir.mkdir()
        self.snapshot = data_dir / "odds_api_today.json"
        self.original = b'{"date_fetched": "2000-01-01", "matches": [{"old": true}]}\n'
        self.snapshot.write_bytes(self.original)
        stack.enter_context(patch.object(odds, "DATA_DIR", str(data_dir)))
        stack.enter_context(patch.object(odds, "OUT_FILE", str(self.snapshot)))
        stack.enter_context(patch.object(odds, "_get_api_key", return_value="test-secret"))
        self.get = stack.enter_context(patch.object(odds.requests, "get"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_failed_discovery_preserves_cache_without_fallback_requests(self):
        for status in [401, 429, 503]:
            with self.subTest(status=status):
                self.get.reset_mock()
                self.get.return_value = response({}, status)
                with self.assertRaises(requests.HTTPError):
                    odds.main()
                self.assertEqual(self.snapshot.read_bytes(), self.original)
                self.assertEqual(self.get.call_count, 1)

    def test_partial_tournament_failure_preserves_cache(self):
        sports = SPORTS + [{"key": WTA, "group": "Tennis"}]
        self.get.side_effect = [response(sports), response([EVENT]), response({}, 503)]
        with self.assertRaises(requests.HTTPError):
            odds.main()
        self.assertEqual(self.snapshot.read_bytes(), self.original)
        self.assertEqual(self.get.call_count, 3)

    def test_disappearing_active_tournament_does_not_make_a_complete_snapshot(self):
        self.get.side_effect = [response(SPORTS), response({}, 404)]
        with self.assertRaises(requests.HTTPError):
            odds.main()
        self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_network_failure_does_not_create_a_cache(self):
        self.snapshot.unlink()
        self.get.side_effect = requests.ConnectionError("offline fixture")
        with self.assertRaises(requests.ConnectionError):
            odds.main()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(self.get.call_count, 1)

    def test_tournament_timeout_preserves_cache(self):
        self.get.side_effect = [response(SPORTS), requests.Timeout("offline fixture")]
        with self.assertRaises(requests.Timeout):
            odds.main()
        self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_invalid_json_preserves_cache(self):
        invalid = response(None)
        invalid._content = b"not json"
        for replies in [[invalid], [response(SPORTS), invalid]]:
            with self.subTest(replies=len(replies)):
                self.get.side_effect = replies
                with self.assertRaises(ValueError):
                    odds.main()
                self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_wrong_response_shape_preserves_cache(self):
        for invalid in [{}, {"error": "temporarily unavailable"}, ["invalid item"]]:
            for replies in [[response(invalid)], [response(SPORTS), response(invalid)]]:
                with self.subTest(payload=invalid, replies=len(replies)):
                    self.get.side_effect = replies
                    with self.assertRaises(ValueError):
                        odds.main()
                    self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_failure_can_retry_then_success_is_cached(self):
        self.get.side_effect = [response(SPORTS), response({}, 503)]
        with self.assertRaises(requests.HTTPError):
            odds.main()
        self.assertEqual(self.snapshot.read_bytes(), self.original)

        self.get.reset_mock()
        self.get.side_effect = [response(SPORTS), response([EVENT])]
        self.assertEqual(odds.main(), 1)
        payload = json.loads(self.snapshot.read_text())
        self.assertEqual(payload["date_fetched"], odds.datetime.now(odds.timezone.utc).strftime("%Y-%m-%d"))
        self.assertEqual(payload["matches"][0]["player1"], "Player One")
        self.assertEqual(payload["matches"][0]["player2_odds"], 1.8)
        self.assertEqual(odds.main(), 1)
        self.assertEqual(self.get.call_count, 2)

    def test_legitimate_empty_snapshots_are_cached(self):
        for replies in [[response([])], [response(SPORTS), response([])]]:
            with self.subTest(replies=len(replies)):
                self.snapshot.write_bytes(self.original)
                self.get.reset_mock()
                self.get.side_effect = replies
                self.assertEqual(odds.main(), 0)
                self.assertEqual(json.loads(self.snapshot.read_text())["matches"], [])
                self.assertEqual(odds.main(), 0)
                self.assertEqual(self.get.call_count, len(replies))

    def test_dry_run_does_not_replace_cache(self):
        self.get.side_effect = [response(SPORTS), response([EVENT])]
        self.assertEqual(odds.main(dry_run=True), 1)
        self.assertEqual(self.snapshot.read_bytes(), self.original)

    def run_cli(self):
        script = """
import os, runpy, sys
from unittest.mock import patch
import requests
sys.path.insert(0, sys.argv[1])
os.environ['ODDS_API_KEY'] = 'test-secret'
reply = requests.Response()
reply.status_code = 503
reply.url = 'https://api.the-odds-api.com/v4/sports?apiKey=test-secret'
reply._content = b'{}'
sys.argv = ['fetch_odds_api.py']
with patch('requests.get', return_value=reply):
    runpy.run_module('fetch_odds_api', run_name='__main__')
"""
        return subprocess.run(
            [sys.executable, "-c", script, str(Path(odds.__file__).parent)],
            cwd=self.directory, capture_output=True, text=True, timeout=30,
        )

    def test_cli_failure_returns_nonzero_and_keeps_key_out_of_output(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("snapshot", result.stderr.lower())
        self.assertNotIn("test-secret", result.stdout + result.stderr)
        self.assertEqual(self.snapshot.read_bytes(), self.original)

    def test_cli_success_does_not_use_match_count_as_exit_status(self):
        self.snapshot.write_text(json.dumps({
            "date_fetched": odds.datetime.now(odds.timezone.utc).strftime("%Y-%m-%d"),
            "matches": [EVENT, EVENT],
        }))
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipping API call", result.stdout)


if __name__ == "__main__":
    unittest.main()
