import unittest

from liquidation_shadow import LiquidationShadowMonitor


def _forced_order(symbol="BTCUSDT", side="SELL", price="100", qty="1", ts=1000):
    return {
        "e": "forceOrder",
        "E": ts * 1000,
        "o": {
            "s": symbol,
            "S": side,
            "o": "LIMIT",
            "f": "IOC",
            "q": qty,
            "p": price,
            "ap": price,
            "X": "FILLED",
            "l": qty,
            "z": qty,
            "T": ts * 1000,
        },
    }


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LiquidationShadowMonitorMessageTests(unittest.TestCase):
    def test_disabled_start_returns_false(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=False)
        self.assertFalse(monitor.start())

    def test_handle_message_enqueues_valid_order(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=True)
        accepted = monitor.handle_message(_forced_order())
        self.assertTrue(accepted)
        self.assertEqual(monitor.event_queue.qsize(), 1)

    def test_handle_message_ignores_symbols_out_of_scope(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=True)
        accepted = monitor.handle_message(_forced_order(symbol="ETHUSDT"))
        self.assertFalse(accepted)
        self.assertEqual(monitor.event_queue.qsize(), 0)

    def test_handle_message_ignores_malformed_payload(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=True)
        self.assertFalse(monitor.handle_message({"e": "somethingElse"}))
        self.assertFalse(monitor.handle_message("not a dict"))

    def test_handle_message_noop_when_disabled(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=False)
        self.assertFalse(monitor.handle_message(_forced_order()))


class LiquidationShadowMonitorSnapshotTests(unittest.TestCase):
    def test_snapshot_unknown_symbol_is_empty(self):
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=True)
        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["sample_count"], 0)
        self.assertEqual(snapshot["long_liquidation_notional"], 0.0)
        self.assertEqual(snapshot["short_liquidation_notional"], 0.0)
        self.assertIsNone(snapshot["last_event_age_seconds"])

    def test_sell_side_order_counts_as_long_liquidation(self):
        clock = FakeClock(1000.0)
        monitor = LiquidationShadowMonitor(
            ["BTCUSDT"], enabled=True, clock=clock
        )
        monitor.handle_message(
            _forced_order(side="SELL", price="100", qty="2", ts=1000)
        )
        monitor.process_pending()

        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["sample_count"], 1)
        self.assertEqual(snapshot["long_liquidation_notional"], 200.0)
        self.assertEqual(snapshot["short_liquidation_notional"], 0.0)
        self.assertEqual(snapshot["net_liquidation_notional"], 200.0)

    def test_buy_side_order_counts_as_short_liquidation(self):
        clock = FakeClock(1000.0)
        monitor = LiquidationShadowMonitor(
            ["BTCUSDT"], enabled=True, clock=clock
        )
        monitor.handle_message(
            _forced_order(side="BUY", price="50", qty="4", ts=1000)
        )
        monitor.process_pending()

        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["long_liquidation_notional"], 0.0)
        self.assertEqual(snapshot["short_liquidation_notional"], 200.0)
        self.assertEqual(snapshot["net_liquidation_notional"], -200.0)

    def test_events_outside_window_are_excluded(self):
        clock = FakeClock(1000.0)
        monitor = LiquidationShadowMonitor(
            ["BTCUSDT"],
            enabled=True,
            clock=clock,
            window_seconds=60,
        )
        monitor.handle_message(
            _forced_order(side="SELL", price="100", qty="1", ts=1000)
        )
        monitor.process_pending()

        clock.advance(120)  # past the 60s window

        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["sample_count"], 0)
        self.assertEqual(snapshot["net_liquidation_notional"], 0.0)

    def test_last_event_age_reflects_most_recent_event(self):
        clock = FakeClock(1000.0)
        monitor = LiquidationShadowMonitor(
            ["BTCUSDT"], enabled=True, clock=clock, window_seconds=600
        )
        monitor.handle_message(
            _forced_order(side="SELL", price="100", qty="1", ts=1000)
        )
        monitor.process_pending()
        clock.advance(30)
        monitor.handle_message(
            _forced_order(side="SELL", price="100", qty="1", ts=1030)
        )
        monitor.process_pending()
        clock.advance(10)

        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["sample_count"], 2)
        self.assertEqual(snapshot["last_event_age_seconds"], 10.0)

    def test_zero_notional_orders_are_ignored(self):
        clock = FakeClock(1000.0)
        monitor = LiquidationShadowMonitor(
            ["BTCUSDT"], enabled=True, clock=clock
        )
        monitor.handle_message(
            _forced_order(side="SELL", price="0", qty="5", ts=1000)
        )
        monitor.process_pending()

        snapshot = monitor.snapshot("BTCUSDT")
        self.assertEqual(snapshot["sample_count"], 0)

    def test_stop_discards_queued_events_without_starting_a_connection(self):
        # Matches the convention in test_order_flow_shadow.py: never call
        # start() in a unit test, since that spawns a real websocket
        # connection attempt in a background thread. stop() must still be
        # safe to call on a monitor that was only ever handed messages.
        monitor = LiquidationShadowMonitor(["BTCUSDT"], enabled=True)
        monitor.handle_message(_forced_order())
        monitor.stop(timeout=0)

        self.assertEqual(monitor.event_queue.qsize(), 0)


if __name__ == "__main__":
    unittest.main()
