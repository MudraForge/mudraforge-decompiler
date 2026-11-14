"""Tests for scripts/live_training_monitor.py."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from scripts.live_training_monitor import (
    create_server,
    discover_training_logs,
    load_telemetry,
    parse_training_log,
)


SAMPLE_STAGE1_LOG = """Device: cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU) | Batch: 32 | AMP: True
Architecture: v2 (Typed-Opcode + SpanUnit) | Raw Vocab: 666 | Logic Vocab: 298 tokens
Train samples: 6400 | Val samples: 1600 | Eff Batch: 32
Parameters -> DomainSensor (Model B): 1,950,507 | LogicReasoner (Model A): 5,899,682 | Total: 7,850,189
[DISCREPANCY-CHECK] v2 dataset contract verified: 120/120 clean (errors=0)
Epoch 01/25 (144.3s) | NLL: 2.3863 | DomTok: 92.81% | TrainBIO: 69.74% | TrainRelF1: 26.48% | BIO: 74.76% | Mod: 97.56% | F1: 53.74% | RelF1: 37.92% | Unit: 92.12% | EM: 10.31% | SemPass: 0.0% | Disc: 20 [BEST SAVED]
Epoch 02/25 (52.7s) | NLL: 1.3469 | DomTok: 94.20% | TrainBIO: 76.89% | TrainRelF1: 46.53% | BIO: 80.51% | Mod: 98.62% | F1: 73.77% | RelF1: 54.16% | Unit: 96.98% | EM: 35.56% | SemPass: 0.0% | Disc: 20 [BEST SAVED]
Epoch 03/25 (53.3s) | NLL: 1.2939 | DomTok: 93.23% | TrainBIO: 80.61% | TrainRelF1: 54.79% | BIO: 79.04% | Mod: 94.69% | F1: 79.17% | RelF1: 55.67% | Unit: 97.67% | EM: 34.25% | SemPass: 0.0% | Disc: 20
"""


class TestLiveTrainingMonitor(unittest.TestCase):
    def test_parse_stage1_log_metrics_gaps_and_eta(self) -> None:
        now = 1_700_000_000.0
        parsed = parse_training_log(
            SAMPLE_STAGE1_LOG,
            log_path="task-10390.log",
            file_mtime=now - 10.0,
            now=now,
        )

        self.assertEqual(parsed["status"], "RUNNING")
        self.assertEqual(parsed["stage"], "SUPERVISED")
        meta = parsed["metadata"]
        self.assertEqual(meta["device"], "cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)")
        self.assertEqual(meta["batch_size"], 32)
        self.assertTrue(meta["amp"])
        self.assertEqual(meta["train_samples"], 6400)
        self.assertEqual(meta["val_samples"], 1600)
        self.assertEqual(meta["params"]["total"], 7_850_189)
        self.assertTrue(meta["discrepancy_check"]["passed"])
        self.assertEqual(meta["discrepancy_check"]["clean"], 120)

        prog = parsed["progress"]
        self.assertEqual(prog["current_epoch"], 3)
        self.assertEqual(prog["total_epochs"], 25)
        self.assertEqual(prog["progress_pct"], 12.0)
        # Steady step time excludes epoch 1 (144.3s), averaging (52.7 + 53.3)/2 = 53.0s
        self.assertEqual(prog["avg_epoch_sec"], 53.0)
        self.assertEqual(prog["eta_sec"], round(53.0 * 22, 1))

        self.assertEqual(len(parsed["epochs"]), 3)
        latest = parsed["latest"]
        self.assertEqual(latest["epoch"], 3)
        self.assertFalse(latest["best_saved"])
        self.assertAlmostEqual(latest["bio_gap"], 1.57, places=2)
        self.assertAlmostEqual(latest["rel_gap"], -0.88, places=2)

        # Best saved points to Epoch 2 even though Epoch 3 did not set [BEST SAVED]
        self.assertEqual(parsed["best_epoch"]["epoch"], 2)
        self.assertAlmostEqual(parsed["deltas_from_prev"]["val_rel_f1"], 1.51, places=2)

    def test_status_completed_stale_and_rl_parsing(self) -> None:
        completed_log = (
            SAMPLE_STAGE1_LOG
            + "Stage 1 Supervised complete.\n"
            + "RL Epoch 01/05 (40.0s) | Mean USVM Reward: 0.6250 | Vector Pass: 74.50% | BIO Agree: 91.20% [BEST SAVED]\n"
        )
        now = 1_700_000_000.0
        rl_parsed = parse_training_log(completed_log, file_mtime=now - 5.0, now=now)
        self.assertEqual(rl_parsed["stage"], "RL")
        self.assertEqual(rl_parsed["status"], "RUNNING")
        self.assertEqual(len(rl_parsed["rl_epochs"]), 1)
        self.assertAlmostEqual(rl_parsed["rl_epochs"][0]["mean_reward"], 0.625, places=3)

        stale_parsed = parse_training_log(
            SAMPLE_STAGE1_LOG,
            file_mtime=now - 600.0,
            now=now,
            stale_after_sec=120.0,
        )
        self.assertEqual(stale_parsed["status"], "STALE")

        done_parsed = parse_training_log(
            SAMPLE_STAGE1_LOG + "\nStage 1 Supervised complete.\n",
            file_mtime=now - 2.0,
            now=now,
        )
        self.assertEqual(done_parsed["status"], "COMPLETED")

    def test_http_server_live_updates_and_discrepancy_inspector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_file = tmp / "run_live.log"
            ckpt_file = tmp / "v2_dual_stream_sup.pt"
            ckpt_file.write_bytes(b"checkpoint-bytes")
            disc_file = tmp / "v2_dual_stream_sup_discrepancies.jsonl"
            disc_file.write_text(
                json.dumps({"epoch": 2, "sample_id": "s_17", "issue": "edge_mismatch"}) + "\n",
                encoding="utf-8",
            )
            log_file.write_text(SAMPLE_STAGE1_LOG, encoding="utf-8")

            discovered = discover_training_logs(search_dirs=[str(tmp)], pinned_log=log_file)
            self.assertEqual(len(discovered), 1)

            server = create_server(
                host="127.0.0.1",
                port=0,
                log_path=log_file,
                checkpoint_path=ckpt_file,
                search_dirs=[str(tmp)],
            )
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    html = resp.read().decode("utf-8")
                    self.assertIn("MudraForge Dual-Stream Live Training Monitor", html)
                    self.assertIn("/api/metrics", html)

                with urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    data1 = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data1["progress"]["current_epoch"], 3)
                self.assertEqual(len(data1["discrepancies"]), 1)
                self.assertTrue(data1["checkpoint"]["exists"])

                # Simulate live training writing Epoch 04
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(
                        "Epoch 04/25 (51.0s) | NLL: 1.1500 | DomTok: 94.10% | TrainBIO: 83.20% | "
                        "TrainRelF1: 60.10% | BIO: 82.40% | Mod: 98.10% | F1: 81.50% | RelF1: 62.30% | "
                        "Unit: 98.05% | EM: 42.00% | SemPass: 0.0% | Disc: 15 [BEST SAVED]\n"
                    )

                with urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=5) as resp:
                    data2 = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data2["progress"]["current_epoch"], 4)
                self.assertEqual(data2["best_epoch"]["epoch"], 4)
                self.assertEqual(data2["latest"]["disc"], 15)

                telemetry_direct = load_telemetry(
                    requested_log="default",
                    default_log=log_file,
                    checkpoint_path=ckpt_file,
                    search_dirs=[str(tmp)],
                )
                self.assertEqual(telemetry_direct["progress"]["current_epoch"], 4)

                # Historical 5-epoch full run + temp smoke log (should exclude smoke log)
                prev_log = tmp / "task-prev.log"
                prev_lines = [
                    "Architecture: v2 (Typed-Opcode + SpanUnit) | Raw Vocab: 639 | Logic Vocab: 298 tokens",
                    "Train samples: 6400 | Val samples: 1600 | Eff Batch: 32",
                ]
                for ep in range(1, 6):
                    prev_lines.append(
                        f"Epoch {ep:02d}/25 (55.0s) | NLL: {2.0 - 0.1 * ep:.4f} | DomTok: 95.00% | "
                        f"TrainBIO: 80.00% | TrainRelF1: 50.00% | BIO: {75.0 + ep:.2f}% | Mod: 98.00% | "
                        f"F1: 80.00% | RelF1: {40.0 + 4.0 * ep:.2f}% | Unit: 97.00% | EM: 35.00% | "
                        f"SemPass: 10.0% | Disc: 19" + (" [BEST SAVED]" if ep == 5 else "")
                    )
                prev_log.write_text("\n".join(prev_lines) + "\n", encoding="utf-8")

                smoke_log = tmp / "task-smoke.log"
                smoke_log.write_text(
                    "Stage 1 Supervised complete in 1.2s -> D:\\Temp\\tmp_xyz\\v2_sup.pt\n"
                    + SAMPLE_STAGE1_LOG,
                    encoding="utf-8",
                )

                with urlopen(f"http://127.0.0.1:{port}/api/comparison", timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    comp = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(comp["active_epoch"], 4)
                names = [r["log_name"] for r in comp["runs"]]
                self.assertIn("run_live.log", names)
                self.assertIn("task-prev.log", names)
                self.assertNotIn("task-smoke.log", names)
                prev_run = next(r for r in comp["runs"] if r["log_name"] == "task-prev.log")
                self.assertEqual(prev_run["matched_epoch"]["epoch"], 4)
                # Active Ep 4 RelF1=62.30 vs prev Ep 4 RelF1=56.00 -> +6.30
                self.assertAlmostEqual(prev_run["delta_vs_active"]["val_rel_f1"], 6.3, places=2)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
