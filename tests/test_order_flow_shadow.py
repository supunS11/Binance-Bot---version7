import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from order_flow_shadow import FUTURES_PUBLIC_STREAM_BASE, OrderFlowShadowMonitor


class FakeClock:
    def __init__(self, value=1000.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class SnapshotProvider:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = []

    def __call__(self, symbol, limit=1000):
        self.calls.append((symbol, limit))

        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)

        return self.snapshots[0]


def snapshot(update_id, bids=None, asks=None):
    return {
        "lastUpdateId": update_id,
        "bids": bids or [["99.9", "20"], ["99.8", "10"]],
        "asks": asks or [["100.1", "10"], ["100.2", "10"]],
    }


def depth_event(first_id, final_id, previous_id, bids=None, asks=None):
    return {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "U": first_id,
        "u": final_id,
        "pu": previous_id,
        "b": bids or [],
        "a": asks or [],
    }


def trade_event(clock, trade_id, *, event_type="trade", maker=False, quantity=1):
    result = {
        "e": event_type,
        "s": "BTCUSDT",
        "p": "100",
        "q": str(quantity),
        "m": maker,
        "T": int(clock() * 1000),
    }

    if event_type == "aggTrade":
        result["a"] = trade_id
    else:
        result["t"] = trade_id

    return result


class DepthSequenceTests(unittest.TestCase):
    def test_snapshot_plus_one_event_waits_for_required_snapshot_overlap(self):
        clock = FakeClock()
        provider = SnapshotProvider([snapshot(100)])
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            provider,
            enabled=True,
            telemetry_enabled=False,
            clock=clock,
        )

        monitor.handle_message(depth_event(101, 101, 100))
        monitor.process_pending()
        waiting = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertFalse(waiting["book_synced"])
        self.assertEqual(waiting["last_update_id"], 100)
        self.assertIn("SNAPSHOT_BRIDGE_GAP", waiting["last_error"])
        self.assertEqual(len(provider.calls), 1)

        monitor.handle_message(depth_event(100, 102, 99))
        monitor.process_pending()
        recovered = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertTrue(recovered["book_synced"])
        self.assertEqual(recovered["last_update_id"], 102)
        self.assertEqual(len(provider.calls), 1)

    def test_callback_is_non_blocking_and_snapshot_bridge_is_validated(self):
        clock = FakeClock()
        provider = SnapshotProvider([snapshot(100)])
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            provider,
            enabled=True,
            telemetry_enabled=False,
            max_book_levels=2,
            clock=clock,
        )
        message = depth_event(
            100,
            101,
            99,
            bids=[["99.9", "30"], ["99.7", "50"]],
            asks=[["100.1", "0"], ["100.05", "5"]],
        )

        self.assertTrue(monitor.handle_message(message))
        self.assertEqual(provider.calls, [])
        self.assertEqual(monitor.process_pending(), 1)

        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertEqual(provider.calls, [("BTCUSDT", 2)])
        self.assertTrue(result["book_synced"])
        self.assertEqual(result["last_update_id"], 101)
        self.assertEqual(result["best_bid"], 99.9)
        self.assertEqual(result["best_ask"], 100.05)
        self.assertLessEqual(result["bid_levels"], 2)
        self.assertLessEqual(result["ask_levels"], 2)

    def test_pu_gap_resyncs_and_waits_for_a_valid_bridge(self):
        clock = FakeClock()
        provider = SnapshotProvider([snapshot(100), snapshot(104)])
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            provider,
            enabled=True,
            telemetry_enabled=False,
            clock=clock,
        )

        monitor.handle_message(depth_event(100, 101, 99))
        monitor.handle_message(depth_event(102, 102, 101, bids=[["99.9", "25"]]))
        monitor.process_pending()
        self.assertTrue(
            monitor.snapshot("BTCUSDT", emit_telemetry=False)["book_synced"]
        )

        monitor.handle_message(depth_event(104, 104, 999))
        monitor.process_pending()
        unsynchronised = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertTrue(unsynchronised["book_synced"])
        self.assertEqual(unsynchronised["last_update_id"], 104)
        self.assertEqual(unsynchronised["resync_count"], 1)
        self.assertEqual(unsynchronised["sequence_gaps"], 1)

        monitor.handle_message(depth_event(105, 105, 104, asks=[["100.1", "8"]]))
        monitor.process_pending()
        recovered = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertTrue(recovered["book_synced"])
        self.assertEqual(recovered["last_update_id"], 105)
        self.assertEqual(len(provider.calls), 2)

    def test_snapshot_failure_is_contained_inside_shadow_processing(self):
        clock = FakeClock()
        calls = []

        def unavailable(symbol, limit=1000):
            calls.append((symbol, limit))
            raise RuntimeError("offline")

        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            unavailable,
            enabled=True,
            telemetry_enabled=False,
            clock=clock,
        )

        self.assertTrue(monitor.handle_message(depth_event(1, 1, 0)))
        self.assertEqual(calls, [])
        monitor.process_pending()
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertFalse(result["available"])
        self.assertFalse(result["book_synced"])
        self.assertEqual(result["snapshot_errors"], 1)
        self.assertIn("SNAPSHOT_ERROR", result["last_error"])

    def test_snapshot_failure_uses_retry_cooldown(self):
        clock = FakeClock()
        calls = []

        def unavailable(symbol, limit=1000):
            calls.append((symbol, limit))
            raise RuntimeError("offline")

        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            unavailable,
            enabled=True,
            telemetry_enabled=False,
            clock=clock,
        )

        with patch(
            "config.ORDER_FLOW_SHADOW_SNAPSHOT_RETRY_SECONDS",
            10,
            create=True,
        ):
            monitor.handle_message(depth_event(1, 1, 0))
            monitor.process_pending()
            monitor.handle_message(depth_event(2, 2, 1))
            monitor.process_pending()
            self.assertEqual(len(calls), 1)
            clock.advance(10)
            monitor.handle_message(depth_event(3, 3, 2))
            monitor.process_pending()

        self.assertEqual(len(calls), 2)

    def test_stop_discards_queued_depth_without_snapshot_request(self):
        provider = SnapshotProvider([snapshot(100)])
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            provider,
            enabled=True,
            telemetry_enabled=False,
        )
        monitor.handle_message(depth_event(101, 101, 100))
        monitor.stop(timeout=0)

        self.assertEqual(monitor.health()["queue_size"], 0)
        self.assertEqual(provider.calls, [])


class BoundedStateTests(unittest.TestCase):
    def test_queue_drops_oldest_without_blocking_and_trades_remain_bounded(self):
        clock = FakeClock()
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            enabled=True,
            queue_size=2,
            max_trades_per_symbol=2,
            min_bucket_notional=0,
            telemetry_enabled=False,
            clock=clock,
        )

        for trade_id in range(3):
            self.assertTrue(monitor.handle_message(trade_event(clock, trade_id)))

        self.assertEqual(monitor.health()["queue_size"], 2)
        self.assertEqual(monitor.health()["dropped_events"], 1)
        monitor.process_pending()
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(result["dropped_events"], 1)

    def test_duplicate_trade_id_is_not_counted_twice(self):
        clock = FakeClock()
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            enabled=True,
            min_bucket_notional=0,
            telemetry_enabled=False,
            clock=clock,
        )
        event = trade_event(clock, 7, event_type="aggTrade", quantity=2)

        monitor.handle_message(event)
        monitor.handle_message(event)
        monitor.process_pending()
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["duplicate_trades"], 1)
        self.assertEqual(result["cumulative_cvd_notional"], 200)

    def test_old_trades_and_duplicate_ids_are_pruned_by_window(self):
        clock = FakeClock(1_700_000_000)
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            enabled=True,
            window_seconds=5,
            max_trades_per_symbol=100,
            telemetry_enabled=False,
            clock=clock,
        )
        monitor.handle_message(trade_event(clock, 7))
        monitor.process_pending()
        clock.advance(6)
        monitor.handle_message(trade_event(clock, 7))
        monitor.process_pending()
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["duplicate_trades"], 0)


class ShadowMetricTests(unittest.TestCase):
    def test_depth_stream_uses_current_binance_public_route(self):
        self.assertEqual(
            FUTURES_PUBLIC_STREAM_BASE,
            "wss://fstream.binance.com/public/stream?streams=",
        )

    def make_monitor(self, clock):
        provider = SnapshotProvider([
            snapshot(
                100,
                bids=[["99.9", "100"], ["99.8", "50"]],
                asks=[["100.1", "10"], ["100.2", "10"]],
            )
        ])
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            provider,
            enabled=True,
            min_bucket_notional=0,
            imbalance_ratio=3,
            impact_notional_usdt=500,
            stale_seconds=5,
            telemetry_enabled=False,
            clock=clock,
        )
        monitor.handle_message(depth_event(100, 101, 99))
        monitor.handle_message(trade_event(clock, 1, event_type="trade", quantity=5))
        monitor.handle_message(trade_event(clock, 2, event_type="aggTrade", quantity=5))
        monitor.handle_message(trade_event(clock, 3, maker=True, quantity=1))
        monitor.process_pending()
        return monitor

    def test_footprint_cvd_and_full_depth_create_bounded_shadow_score(self):
        clock = FakeClock()
        monitor = self.make_monitor(clock)
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertTrue(result["available"])
        self.assertTrue(result["shadow_only"])
        self.assertFalse(result["decision_effect"])
        self.assertFalse(result["ranking_effect"])
        self.assertGreater(result["cvd_notional"], 0)
        self.assertGreater(result["cvd_ratio"], 0)
        self.assertGreater(result["buy_imbalance_buckets"], 0)
        self.assertGreater(result["full_depth_imbalance"], 0)
        self.assertGreater(result["shadow_score"], 0)
        self.assertLessEqual(result["shadow_score"], 5)
        self.assertEqual(result["sell_shadow_score"], -result["shadow_score"])
        self.assertTrue(result["footprint_buckets"])

    def test_stale_data_is_unavailable_and_has_zero_score(self):
        clock = FakeClock()
        monitor = self.make_monitor(clock)
        clock.advance(6)
        result = monitor.snapshot("BTCUSDT", emit_telemetry=False)

        self.assertFalse(result["available"])
        self.assertEqual(result["shadow_score"], 0)
        self.assertFalse(result["book_fresh"])
        self.assertFalse(result["trade_fresh"])

    def test_stream_names_are_chunked_for_internal_websocket_owner(self):
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            enabled=True,
            max_symbols=3,
            telemetry_enabled=False,
        )

        with (
            patch("config.ORDER_FLOW_SHADOW_DEPTH_UPDATE_MS", 100),
            patch("config.ORDER_FLOW_SHADOW_DEPTH_STREAMS_PER_SOCKET", 2),
        ):
            chunks = monitor.depth_stream_chunks()

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], ("btcusdt@depth@100ms", "ethusdt@depth@100ms"))
        self.assertEqual(chunks[1], ("solusdt@depth@100ms",))


class TelemetryTests(unittest.TestCase):
    def test_unsupported_depth_speed_falls_back_to_supported_100ms(self):
        monitor = OrderFlowShadowMonitor(
            ["BTCUSDT"],
            enabled=True,
            telemetry_enabled=False,
        )

        with patch("config.ORDER_FLOW_SHADOW_DEPTH_UPDATE_MS", 250):
            self.assertEqual(
                monitor.depth_stream_chunks(),
                (("btcusdt@depth@100ms",),),
            )

    def test_csv_telemetry_is_optional_rate_limited_and_flushable_offline(self):
        clock = FakeClock()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.csv"
            monitor = OrderFlowShadowMonitor(
                ["BTCUSDT"],
                enabled=True,
                min_bucket_notional=0,
                telemetry_enabled=True,
                telemetry_path=str(path),
                telemetry_min_interval_seconds=10,
                clock=clock,
            )
            monitor.handle_message(trade_event(clock, 1, quantity=2))
            monitor.process_pending()

            first = monitor.snapshot("BTCUSDT")
            second = monitor.snapshot("BTCUSDT")

            self.assertTrue(first["available"])
            self.assertTrue(second["available"])
            self.assertEqual(monitor.flush_telemetry(), 1)
            self.assertEqual(monitor.flush_telemetry(), 0)

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "BTCUSDT")
            self.assertEqual(rows[0]["shadow_score"], str(first["shadow_score"]))


if __name__ == "__main__":
    unittest.main()

