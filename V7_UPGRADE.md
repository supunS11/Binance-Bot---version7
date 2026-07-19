# V7 execution, order-flow shadow, and TP runner upgrade

## Included

- Exact client-order fill reconciliation, residual tracking, slippage, fee,
  latency, spread, and fallback telemetry.
- Marketable IOC entry/DCA execution with market fallback only after the IOC is
  confirmed terminal.
- Cumulative CVD, footprint, and retained top-1000 depth analytics in strict
  observation-only shadow mode. Shadow output cannot change ranking, filters,
  sizing, execution, or trade count.
- TP1 75% lifecycle hardening: exact triggered-child fill attribution,
  persisted repair intent, restart adoption, runner SL placement before TP2,
  and stale-order reconciliation.
- Fail-closed one-way-mode and runtime-state validation, rolling state backup,
  graceful SIGINT/SIGTERM cleanup, and explicit telemetry flushing.

V7's strategy rules and `1d / 4h / 1h` decision timeframes are unchanged.
Market-flow rank weight remains `1`, and the hard veto remains disabled.

## One-time local/VPS deployment

1. Stop V7 and verify no local or VPS V7 process is trading the account.
2. Back up `.env` and the configured `DCA_STATE_PATH`. Validate the state copy
   as JSON and retain it outside the checkout.
3. Deploy the code, restore the state file if necessary, and ensure the service
   user can write its directory.
4. Merge `v7_upgrade.env.example` into the existing environment. Do not replace
   API keys, symbols, timeframes, strategy thresholds, or risk settings.
5. Confirm the Binance Futures account uses one-way mode.
6. Start exactly one V7 instance. Never run local and VPS instances against the
   same account simultaneously.
7. Inspect the first pending-execution/TP reconciliation cycle, TP1/SL/TP2 IDs,
   `execution_telemetry_v7.csv`, and `order_flow_shadow_v7.csv` before leaving
   the bot unattended.

## Offline verification

```powershell
venv\Scripts\python.exe -m unittest discover -s tests
venv\Scripts\python.exe -m py_compile config.py exchange.py execution_telemetry.py main.py market_intelligence.py multi_tp.py order_flow_shadow.py trade_state.py
```

Offline tests do not replace a controlled testnet or minimum-size exchange
smoke test after deployment.
