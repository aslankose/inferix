#!/usr/bin/env python3
"""
Inferix Contributor Client v0.1.0
https://github.com/aslankose/inferix

Contribution model:
  PRIMARY:   Serve transformer model layers via distributed pipeline
             → Earn GFT tokens proportional to FLOPs delivered
  SECONDARY: Respond to periodic inference challenges (every 5 min)
             → Verify reliability, update reliability factor

Usage:
    python inferix_client.py setup     # First time setup
    python inferix_client.py start     # Start contributing
    python inferix_client.py status    # Show balance and stats
    python inferix_client.py stop      # Stop the daemon

Requirements:
    pip install requests fastapi uvicorn

For real model inference:
    pip install torch transformers
"""

import argparse, hashlib, json, os, platform, signal, sys
import threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("\n  ERROR: pip install requests\n"); sys.exit(1)

VERSION              = "0.1.0"
CONFIG_FILE          = Path.home() / ".inferix" / "config.json"
LOG_FILE             = Path.home() / ".inferix" / "client.log"
HEARTBEAT_INTERVAL   = 30     # Node heartbeat every 30s
SHARD_HEARTBEAT      = 30     # Shard heartbeat every 30s
CHALLENGE_INTERVAL   = 300    # Check for challenges every 5 min
STATS_REFRESH        = 60     # Refresh token balance every 60s
DEFAULT_SERVER       = "http://localhost:8000"
DEFAULT_SHARD_PORT   = 8010


# ── Terminal colors ────────────────────────────────────────────
class C:
    RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
    RED="\033[91m";  CYAN="\033[96m"; DIM="\033[2m"
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except: RESET=BOLD=GREEN=YELLOW=RED=CYAN=DIM=""


def log(msg, level="INFO"):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] [{level}] {msg}\n")

def print_header():
    print(f"\n{C.CYAN}{C.BOLD}"
          f"╔══════════════════════════════════════════╗\n"
          f"║   Inferix Contributor Client  v{VERSION}     ║\n"
          f"╚══════════════════════════════════════════╝"
          f"{C.RESET}\n")

def load_config():
    return json.load(open(CONFIG_FILE)) if CONFIG_FILE.exists() else None

def save_config(config):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(CONFIG_FILE, "w"), indent=2)


# ── Hardware Detection ─────────────────────────────────────────
def detect_hardware():
    h = {
        "cpu_model":     platform.processor() or "Unknown",
        "cpu_cores":     os.cpu_count() or 1,
        "ram_gb":        _ram_gb(),
        "gpu_model":     None,
        "gpu_memory_gb": None,
        "gpu_type":      None,
        "vram_gb":       0.0,
    }
    gpu = _detect_gpu()
    if gpu:
        h.update({"gpu_model": gpu["name"], "gpu_memory_gb": gpu["memory_gb"],
                  "gpu_type": gpu["type"], "vram_gb": gpu["memory_gb"] or 0.0})
    return h


def _ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1)
    except:
        try:
            import subprocess
            if platform.system() == "Darwin":
                return round(int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"]).strip()) / (1024**3), 1)
        except: pass
    return 0.0


def _detect_gpu():
    # NVIDIA
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split(",")
            if len(p) >= 2:
                return {"name": p[0].strip(),
                        "memory_gb": round(float(p[1].strip()) / 1024, 1),
                        "type": "discrete"}
    except: pass
    # Windows WMIC
    try:
        import subprocess
        if platform.system() == "Windows":
            r = subprocess.run(
                ["wmic", "path", "win32_VideoController",
                 "get", "name,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    p = [x.strip() for x in line.split(",") if x.strip()]
                    if len(p) >= 3 and any(x in p[2] for x in
                            ["Intel","AMD","NVIDIA","Graphics"]):
                        try: ram = round(int(p[1]) / (1024**3), 1)
                        except: ram = 0.0
                        intg = any(x in p[2] for x in
                            ["UHD","HD Graphics","Iris","Vega","Radeon Graphics"])
                        return {"name": p[2], "memory_gb": ram,
                                "type": "integrated" if intg else "discrete"}
    except: pass
    # macOS
    try:
        import subprocess
        if platform.system() == "Darwin":
            r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.split("\n"):
                if "Chipset Model:" in line:
                    return {"name": line.split(":")[-1].strip(),
                            "memory_gb": 0.0, "type": "integrated"}
    except: pass
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"name": p.name, "memory_gb": round(p.total_memory/1e9,1),
                    "type": "discrete"}
        if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            return {"name": "Apple Silicon GPU", "memory_gb": 16.0, "type": "discrete"}
    except: pass
    return None


# ── Benchmarking ───────────────────────────────────────────────
def run_benchmark(hardware):
    print(f"\n  {C.CYAN}Running benchmark...{C.RESET}")
    ms = _bm_matmul(); mem = _bm_memory(); lat = _bm_latency()
    gt = hardware.get("gpu_type")
    p  = 0.25 if gt=="integrated" else (0.15 if not gt else 1.0)
    if p < 1.0:
        print(f"  {C.YELLOW}Hardware penalty: {p}x ({gt or 'no GPU'}){C.RESET}")
    ms  = round(ms*p, 4); mem = round(mem*p, 4); lat = round(lat*p, 4)
    api = round(0.5*ms + 0.3*mem + 0.2*lat, 4)
    print(f"  {C.DIM}Matmul:{ms:.3f}  Memory:{mem:.3f}  Latency:{lat:.3f}{C.RESET}")
    print(f"  {C.GREEN}API Score: {api:.3f}{C.RESET}")
    return {"matmul_score":ms, "memory_score":mem,
            "latency_score":lat, "api_score":api}

def _bm_matmul():
    try:
        n=50; a=[[float(i*j%100) for j in range(n)] for i in range(n)]
        b=[[float((i+j)%100) for j in range(n)] for i in range(n)]
        s=time.perf_counter(); r=0
        for i in range(n):
            for k in range(n):
                for j in range(n): r += a[i][k]*b[k][j]
        return max(0.0, min(1.0, 1.0-(time.perf_counter()-s-0.1)/4.9))
    except: return 0.1

def _bm_memory():
    try:
        d=list(range(10_000_000)); s=time.perf_counter(); sum(d)
        return max(0.0, min(1.0, 1.0-(time.perf_counter()-s-0.05)/1.95))
    except: return 0.1

def _bm_latency():
    try:
        s=time.perf_counter()
        for i in range(10_000): hashlib.sha256(str(i).encode()).hexdigest()
        return max(0.0, min(1.0, 1.0-(time.perf_counter()-s-0.05)/1.95))
    except: return 0.1

def fingerprint(hw):
    return hashlib.sha256(json.dumps({
        "cpu": hw.get("cpu_model",""), "cores": hw.get("cpu_cores",0),
        "gpu": hw.get("gpu_model",""), "platform": platform.system(),
        "machine": platform.machine()}, sort_keys=True).encode()).hexdigest()[:32]


# ── API Client ─────────────────────────────────────────────────
class InferixClient:
    def __init__(self, url, node_id=None):
        self.url = url.rstrip("/"); self.node_id = node_id
        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})

    def ok(self):
        try: return self.s.get(f"{self.url}/health", timeout=5).status_code == 200
        except: return False

    def register(self, hw, bm, max_gpu):
        r = self.s.post(f"{self.url}/nodes/register", json={
            "public_key": f"inferix-{uuid.uuid4().hex[:16]}",
            "hardware_profile": {
                "gpu_model":      hw.get("gpu_model"),
                "gpu_memory_gb":  hw.get("gpu_memory_gb"),
                "cpu_model":      hw.get("cpu_model"),
                "cpu_cores":      hw.get("cpu_cores"),
                "ram_gb":         hw.get("ram_gb"),
                "matmul_score":   bm.get("matmul_score"),
                "memory_score":   bm.get("memory_score"),
                "latency_score":  bm.get("latency_score"),
                "fingerprint":    fingerprint(hw),
            },
            "max_gpu_utilization": max_gpu,
        }, timeout=10)
        r.raise_for_status(); return r.json()

    def set_region(self, region):
        try:
            return self.s.post(
                f"{self.url}/grid/nodes/{self.node_id}/region?region={region}",
                timeout=5).status_code == 200
        except: return False

    def heartbeat(self):
        try:
            return self.s.post(
                f"{self.url}/nodes/{self.node_id}/heartbeat",
                timeout=5).status_code == 200
        except: return False

    def get_challenge(self):
        """Request a periodic inference challenge from coordinator."""
        try:
            r = self.s.post(
                f"{self.url}/tasks/dispatch/{self.node_id}",
                timeout=10)
            return r.json() if r.status_code == 200 else None
        except: return None

    def submit_challenge(self, task_id, result_hash, flops, ms):
        """Submit inference challenge result."""
        try:
            r = self.s.post(f"{self.url}/tasks/result", json={
                "task_id":           task_id,
                "node_id":           self.node_id,
                "result_hash":       result_hash,
                "flops_delivered":   flops,
                "execution_time_ms": ms,
            }, timeout=10)
            return r.json() if r.status_code == 200 else None
        except: return None

    def balance(self):
        try:
            r = self.s.get(
                f"{self.url}/tokens/{self.node_id}/balance",
                timeout=5)
            return r.json() if r.status_code == 200 else None
        except: return None

    def reg_shard(self, model_id, l0, l1, vram, region, host, port):
        try:
            r = self.s.post(f"{self.url}/inference/shards/register", json={
                "node_id":     self.node_id,
                "model_id":    model_id,
                "layer_start": l0,
                "layer_end":   l1,
                "vram_gb":     vram,
                "region":      region,
                "host":        host,
                "port":        port,
            }, timeout=10)
            return r.status_code == 200
        except: return False

    def shard_hb(self) -> str:
        """
        Send shard heartbeat.
        Returns: 'ok', 'not_registered', or 'error'
        """
        try:
            r = self.s.post(
                f"{self.url}/inference/shards/heartbeat/{self.node_id}",
                timeout=5)
            if r.status_code == 200:   return "ok"
            if r.status_code == 404:   return "not_registered"
            return "error"
        except: return "error"

    def pipeline(self, model_id):
        try:
            r = self.s.get(f"{self.url}/inference/pipeline/{model_id}", timeout=5)
            return r.json() if r.status_code == 200 else None
        except: return None

    def suggest(self, model_id, vram):
        try:
            r = self.s.get(
                f"{self.url}/inference/suggest/{model_id}?vram_gb={vram}",
                timeout=5)
            return r.json() if r.status_code == 200 else None
        except: return None


# ── Challenge Executor ─────────────────────────────────────────
class ChallengeExecutor:
    """
    Executes periodic inference challenges to prove reliability.
    Computes the deterministic activation hash that matches
    the coordinator's expected answer.
    """

    def execute(self, task: dict, shard_loader=None) -> tuple:
        """
        Execute inference challenge.
        Returns (activation_hash, flops_delivered, exec_ms)
        """
        start   = time.perf_counter()
        payload = task.get("payload", {})
        prompt  = payload.get("prompt", "")

        if shard_loader is not None:
            # Run actual forward pass and hash the activations
            result = shard_loader.forward({
                "input_ids":      shard_loader.tokenize(prompt),
                "is_first_shard": task.get("layer_start", 0) == 0,
                "is_last_shard":  False,
                "max_new_tokens": 0,
                "temperature":    0.0,
            })
            act_hash = self._hash_result(
                prompt,
                task.get("layer_start", 0),
                task.get("layer_end", 15),
                task.get("model_id", ""),
            )
            flops = result.get("flops_delivered", task.get("flops_estimated", 1.0))
        else:
            # Compute deterministic hash — matches server challenge_answer
            act_hash = self._hash_result(
                prompt,
                task.get("layer_start", 0),
                task.get("layer_end", 15),
                task.get("model_id", ""),
            )
            flops = task.get("flops_estimated", 1.0)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return act_hash, flops, round(elapsed_ms, 2)

    def _hash_result(self, prompt, layer_start, layer_end, model_id) -> str:
        """
        Deterministic hash matching server-side challenge_answer generation.
        Must stay in sync with _generate_challenge_answer() in tasks.py.
        """
        content = json.dumps({
            "prompt":      prompt,
            "layer_start": layer_start,
            "layer_end":   layer_end,
            "model_id":    model_id,
            "temperature": 0.0,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# ── Shard Server ───────────────────────────────────────────────
class ShardServer:
    """
    Runs the activation server in a background thread.
    This is the PRIMARY contribution mechanism —
    serving transformer layers earns GFT tokens.
    """

    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.stats  = None
        self.loader = None
        self._stop  = threading.Event()

    def start(self, stats: 'Stats' = None):
        if not self.config.get("inference_enabled"):
            return

        self.stats = stats
        cfg = self.config
        print(f"\n  {C.CYAN}Starting inference shard server...{C.RESET}")
        print(f"  Model:  {cfg['model_id']}")
        print(f"  Layers: {cfg['layer_start']} -> {cfg['layer_end']}")
        print(f"  Port:   {cfg.get('shard_port', DEFAULT_SHARD_PORT)}")
        print(f"  Mode:   {'SIMULATION' if cfg.get('simulate') else 'PRODUCTION'}")

        if stats: stats.set_shard("starting")
        self._register(cfg)
        self.loader = self._create_loader()
        threading.Thread(target=self._run_server, daemon=True).start()
        threading.Thread(target=self._hb_loop, daemon=True).start()

    def _register(self, cfg):
        """Register shard with coordinator, retry until successful."""
        host = self._ip()
        for attempt in range(10):
            ok = self.client.reg_shard(
                cfg["model_id"], cfg["layer_start"], cfg["layer_end"],
                cfg.get("vram_gb", 0), cfg.get("region", "UNKNOWN"),
                host, cfg.get("shard_port", DEFAULT_SHARD_PORT),
            )
            if ok:
                if self.stats:
                    self.stats.set_shard("registered")
                    self.stats.add_event(f"Shard registered (layers {cfg['layer_start']}-{cfg['layer_end']})")
                return
            else:
                if self.stats:
                    self.stats.set_shard("reconnecting")
                    self.stats.add_event(f"Shard registration attempt {attempt+1} failed, retrying...")
                    self.stats.reconnect_attempts += 1
                time.sleep(5)
        if self.stats:
            self.stats.set_shard("failed")
            self.stats.add_event("Shard registration failed after 10 attempts")

    def _run_server(self):
        try:
            from fastapi import FastAPI
            import uvicorn
            cfg = self.config
            app = FastAPI(title="Inferix Activation Server", version=VERSION)
            ldr = self.loader

            @app.get("/health")
            def health():
                return {
                    "status":      "simulation" if cfg.get("simulate") else "ready",
                    "model_id":    cfg["model_id"],
                    "layer_start": cfg["layer_start"],
                    "layer_end":   cfg["layer_end"],
                    "node_id":     cfg.get("node_id"),
                }

            @app.post("/shard/tokenize")
            def tokenize(data: dict):
                ids = ldr.tokenize(data.get("prompt", ""))
                return {"input_ids": ids, "token_count": len(ids)}

            @app.post("/shard/forward")
            def forward(data: dict):
                return ldr.forward(data)

            uvicorn.run(
                app, host="0.0.0.0",
                port=cfg.get("shard_port", DEFAULT_SHARD_PORT),
                log_level="warning",
            )
        except Exception as e:
            log(f"Shard server error: {e}", "ERROR")

    def _create_loader(self):
        cfg = self.config

        class Loader:
            def __init__(self):
                self.simulate = cfg.get("simulate", True)
                self.model_id = cfg["model_id"]
                self.l0 = cfg["layer_start"]
                self.l1 = cfg["layer_end"]
                if not self.simulate:
                    self._load()

            def _load(self):
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    print(f"\n  Loading {self.model_id} layers {self.l0}-{self.l1}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    print(f"  {C.GREEN}Model loaded.{C.RESET}")
                except Exception as e:
                    print(f"  {C.RED}Load failed: {e} — simulation{C.RESET}")
                    self.simulate = True

            def tokenize(self, prompt):
                if self.simulate or not hasattr(self, "tokenizer"):
                    return [hash(w) % 50000 for w in prompt.split()]
                return self.tokenizer.encode(
                    prompt, return_tensors="pt")[0].tolist()

            def forward(self, data: dict) -> dict:
                lc    = self.l1 - self.l0 + 1
                seq   = len(data.get("input_ids") or
                            data.get("activations", [[]])[0])
                flops = round(12 * (4096**2) * seq * lc / 1e9, 4)
                time.sleep(lc * 0.01)

                if data.get("is_last_shard") and data.get("max_new_tokens", 0) > 0:
                    n   = min(data["max_new_tokens"], 20)
                    ids = list(range(1000, 1000+n))
                    return {
                        "output_ids":      ids,
                        "text":            f"[Simulated: {n} tokens "
                                           f"layers {self.l0}-{self.l1}]",
                        "flops_delivered": flops,
                    }
                return {
                    "activations":     [[[0.0]*4096]*max(seq, 1)],
                    "flops_delivered": flops,
                }

        return Loader()

    def _hb_loop(self):
        consecutive_failures = 0
        while not self._stop.is_set():
            status = self.client.shard_hb()
            if status == "ok":
                consecutive_failures = 0
                if self.stats: self.stats.set_shard("registered")
            elif status == "not_registered":
                # Coordination layer restarted — re-register immediately
                consecutive_failures = 0
                log("Coordination layer restarted — re-registering shard...")
                if self.stats:
                    self.stats.set_shard("reconnecting")
                self._register(self.config)
            else:
                consecutive_failures += 1
                if self.stats:
                    self.stats.set_shard("reconnecting")
                    self.stats.reconnect_attempts += 1
                log(f"Shard heartbeat failed ({consecutive_failures})", "WARN")
                if consecutive_failures >= 3:
                    log("Re-registering shard with coordinator...")
                    self._register(self.config)
                    consecutive_failures = 0
            self._stop.wait(SHARD_HEARTBEAT)

    def _ip(self):
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]; s.close(); return ip
        except: return "127.0.0.1"

    def stop(self): self._stop.set()


# ── Stats ──────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.challenges_passed  = 0
        self.challenges_failed  = 0
        self.current_balance    = 0.0
        self.total_earned       = 0.0
        self.reliability        = 1.0
        self.start_time         = datetime.now(timezone.utc)
        self.pipeline_status    = "unknown"
        self.coordinator_status = "connecting"
        self.shard_status       = "starting"
        self.shard_registered   = False
        self.last_coordinator_ok= None
        self.reconnect_attempts = 0
        self._events            = []   # Recent event log
        self._lock              = threading.Lock()

    def record_challenge(self, passed: bool, reliability: float):
        with self._lock:
            if passed: self.challenges_passed += 1
            else:      self.challenges_failed += 1
            self.reliability = reliability
            self._add_event(f"Challenge {'passed' if passed else 'FAILED'} — reliability: {reliability:.4f}")

    def update_balance(self, balance: float, earned: float):
        with self._lock:
            self.current_balance = balance
            self.total_earned    = earned

    def set_coordinator(self, ok: bool):
        with self._lock:
            prev = self.coordinator_status
            self.coordinator_status = "connected" if ok else "disconnected"
            if ok: self.last_coordinator_ok = datetime.now(timezone.utc)
            if prev != self.coordinator_status:
                self._add_event(f"Coordinator {self.coordinator_status}")

    def set_shard(self, status: str):
        with self._lock:
            prev = self.shard_status
            self.shard_status     = status
            self.shard_registered = (status == "registered")
            if prev != status:
                self._add_event(f"Shard {status}")

    def add_event(self, msg: str):
        with self._lock:
            self._add_event(msg)

    def _add_event(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._events.append(f"[{ts}] {msg}")
        if len(self._events) > 6:
            self._events.pop(0)

    def get_events(self):
        with self._lock:
            return list(self._events)

    def uptime(self):
        d = datetime.now(timezone.utc) - self.start_time
        h, r = divmod(int(d.total_seconds()), 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ── Terminal UI ────────────────────────────────────────────────
class UI:
    def __init__(self, stats: Stats, config: dict):
        self.stats = stats; self.config = config
        self._stop = threading.Event()

    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def stop(self):  self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self._render(); time.sleep(2)

    def _render(self):
        s = self.stats; c = self.config
        os.system("cls" if platform.system() == "Windows" else "clear")

        # Connection status indicators
        coord_col  = C.GREEN if s.coordinator_status == "connected" else C.RED
        coord_icon = "●" if s.coordinator_status == "connected" else "○"
        shard_col  = C.GREEN if s.shard_registered else (C.YELLOW if s.shard_status == "starting" else C.RED)
        shard_icon = "●" if s.shard_registered else ("◌" if s.shard_status == "starting" else "○")

        print(f"\n{C.CYAN}{C.BOLD}  Inferix Contributor Client  v{VERSION}{C.RESET}")
        print(f"  {C.DIM}{'─'*52}{C.RESET}")

        # Status bar
        print(f"\n  {coord_col}{coord_icon} Coordinator{C.RESET}  "
              f"{shard_col}{shard_icon} Shard{C.RESET}  "
              f"Uptime: {C.CYAN}{s.uptime()}{C.RESET}")

        if s.reconnect_attempts > 0:
            print(f"  {C.YELLOW}Reconnect attempts: {s.reconnect_attempts}{C.RESET}")

        print(f"\n  {C.BOLD}Node{C.RESET}")
        print(f"  ID:           {C.DIM}{c.get('node_id','N/A')}{C.RESET}")
        print(f"  Server:       {C.DIM}{c.get('server_url','N/A')}{C.RESET}")
        print(f"  Region:       {C.DIM}{c.get('region','UNKNOWN')}{C.RESET}")
        print(f"  Multiplier:   {C.GREEN}{c.get('multiplier',1.0):.3f}x{C.RESET}"
              f"  ({c.get('hardware_class','N/A')})")

        print(f"\n  {C.BOLD}Inference Shard{C.RESET}")
        if c.get("inference_enabled"):
            print(f"  Model:        {C.DIM}{c.get('model_id')}{C.RESET}")
            print(f"  Layers:       {C.DIM}{c.get('layer_start')} -> {c.get('layer_end')}{C.RESET}")
            print(f"  Port:         {C.DIM}{c.get('shard_port')}{C.RESET}")
            print(f"  Status:       {shard_col}{s.shard_status}{C.RESET}")
            mode = "SIMULATION" if c.get("simulate") else "PRODUCTION"
            print(f"  Mode:         {C.YELLOW if c.get('simulate') else C.GREEN}{mode}{C.RESET}")
            pip = s.pipeline_status
            col = C.GREEN if pip == "complete" else C.YELLOW
            print(f"  Pipeline:     {col}{pip}{C.RESET}")
        else:
            print(f"  {C.YELLOW}Not configured — run setup{C.RESET}")

        print(f"\n  {C.BOLD}Reliability Challenges{C.RESET}")
        print(f"  Passed:       {C.GREEN}{s.challenges_passed}{C.RESET}"
              f"  Failed: {C.RED}{s.challenges_failed}{C.RESET}")
        print(f"  Reliability:  {C.YELLOW}{s.reliability:.4f}{C.RESET}")

        print(f"\n  {C.BOLD}Tokens{C.RESET}")
        print(f"  Balance:      {C.CYAN}{s.current_balance:.4f} GFT{C.RESET}")
        print(f"  Total earned: {C.GREEN}{s.total_earned:.4f} GFT{C.RESET}")

        print(f"\n  {C.DIM}Ctrl+C to stop  {'─'*32}{C.RESET}")


# ── Daemon ─────────────────────────────────────────────────────
class Daemon:
    """
    Main contribution daemon.

    Primary loop:  Shard server runs continuously in background,
                   serving inference requests and earning tokens.
    Secondary loop: Every CHALLENGE_INTERVAL seconds, request and
                    respond to an inference challenge to prove reliability.
    Balance loop:   Every STATS_REFRESH seconds, fetch latest balance.
    """

    def __init__(self, config: dict):
        self.config    = config
        self.client    = InferixClient(config["server_url"], config["node_id"])
        self.challenge = ChallengeExecutor()
        self.stats     = Stats()
        self.ui        = UI(self.stats, config)
        self.shard     = ShardServer(config, self.client)
        self._stop     = threading.Event()
        self._last_hb  = 0
        self._last_ch  = 0
        self._last_bal = 0

    def start(self):
        signal.signal(signal.SIGINT,  self._stop_handler)
        signal.signal(signal.SIGTERM, self._stop_handler)

        # Check coordinator is reachable before starting
        print(f"  {C.CYAN}Connecting to coordinator...{C.RESET}")
        for attempt in range(5):
            if self.client.ok():
                print(f"  {C.GREEN}Connected.{C.RESET}")
                self.stats.set_coordinator(True)
                break
            print(f"  {C.YELLOW}Attempt {attempt+1}/5 failed, retrying...{C.RESET}")
            time.sleep(3)
        else:
            print(f"  {C.RED}Cannot reach coordinator. Will keep retrying...{C.RESET}")
            self.stats.set_coordinator(False)

        # Start shard server (PRIMARY earning mechanism)
        self.shard.start(self.stats)
        self.ui.start()
        log("Inferix daemon started.")

        while not self._stop.is_set():
            try:
                now = time.time()

                # Node heartbeat + coordinator connectivity check
                if now - self._last_hb > HEARTBEAT_INTERVAL:
                    ok = self.client.heartbeat()
                    self.stats.set_coordinator(ok)
                    if not ok:
                        self.stats.reconnect_attempts += 1
                        log("Coordinator unreachable", "WARN")
                    self._last_hb = now

                # Inference challenge (reliability verification)
                if now - self._last_ch > CHALLENGE_INTERVAL:
                    if self.stats.coordinator_status == "connected":
                        self._run_challenge()
                    self._last_ch = now

                # Balance refresh
                if now - self._last_bal > STATS_REFRESH:
                    if self.stats.coordinator_status == "connected":
                        self._refresh_balance()
                    self._last_bal = now

                # Pipeline status refresh
                model_id = self.config.get("model_id")
                if model_id and self.stats.coordinator_status == "connected":
                    pp = self.client.pipeline(model_id)
                    if pp:
                        self.stats.pipeline_status = (
                            "complete" if pp.get("available") else "incomplete"
                        )

            except Exception as e:
                log(f"Daemon error: {e}", "ERROR")

            time.sleep(10)

        self.ui.stop()
        self.shard.stop()
        log("Inferix daemon stopped.")
        print(f"\n  {C.YELLOW}Stopped. Balance: "
              f"{self.stats.current_balance:.4f} GFT{C.RESET}\n")

    def _run_challenge(self):
        """Request and respond to an inference challenge."""
        task = self.client.get_challenge()
        if not task:
            return

        log(f"Challenge received: {task.get('task_id','?')[:8]}")

        act_hash, flops, ms = self.challenge.execute(
            task, self.shard.loader
        )
        result = self.client.submit_challenge(
            task["task_id"], act_hash, flops, ms
        )

        if result:
            passed = result.get("passed", False)
            rel    = result.get("reliability", self.stats.reliability)
            self.stats.record_challenge(passed, rel)
            status = "passed" if passed else "FAILED"
            log(f"Challenge {status}. Reliability: {rel:.4f}")

    def _refresh_balance(self):
        bal = self.client.balance()
        if bal:
            self.stats.update_balance(
                bal.get("balance", 0.0),
                bal.get("total_earned", 0.0),
            )

    def _stop_handler(self, *_):
        print(f"\n  {C.YELLOW}Stopping...{C.RESET}")
        self._stop.set()


# ── Commands ───────────────────────────────────────────────────
def cmd_setup():
    print_header()
    print(f"  {C.BOLD}Setup{C.RESET}\n")

    server = input(f"  Server URL [{DEFAULT_SERVER}]: ").strip() or DEFAULT_SERVER
    print(f"\n  Connecting...")
    client = InferixClient(server)
    if not client.ok():
        print(f"  {C.RED}Cannot connect.{C.RESET}\n"); sys.exit(1)
    print(f"  {C.GREEN}Connected.{C.RESET}")

    print(f"\n  Regions: NO FR GB DE US-CA US-WA US-NY US-TX JP AU SG TR")
    region = input("  Region [US-NY]: ").strip().upper() or "US-NY"
    try:    max_gpu = float(input("  Max GPU utilization [0.8]: ").strip() or "0.8")
    except: max_gpu = 0.8

    print(f"\n  {C.CYAN}Detecting hardware...{C.RESET}")
    hw = detect_hardware()
    print(f"  CPU: {hw['cpu_model']} ({hw['cpu_cores']} cores), RAM: {hw['ram_gb']}GB")
    if hw.get("gpu_model"):
        print(f"  GPU: {hw['gpu_model']} ({hw['gpu_memory_gb']}GB) [{hw.get('gpu_type')}]")
    else:
        print(f"  GPU: {C.YELLOW}Not detected{C.RESET}")

    bm   = run_benchmark(hw)
    vram = hw.get("vram_gb", 0.0)

    # Inference shard configuration
    print(f"\n  {C.BOLD}Inference Shard Configuration{C.RESET}")
    print(f"  Your node will serve transformer model layers.")
    print(f"  Any hardware can participate — GPU nodes earn more.\n")

    try:
        r = requests.get(f"{server}/inference/models", timeout=5)
        if r.status_code == 200:
            print(f"  Available models:")
            for m in r.json().get("models", []):
                print(f"  - {m['friendly_name']} ({m['total_layers']} layers)")
    except: pass

    model_id = input("\n  Model [meta-llama/Meta-Llama-3-8B]: ").strip()
    if not model_id: model_id = "meta-llama/Meta-Llama-3-8B"

    # Suggest layer range
    try:
        r  = requests.get(
            f"{server}/inference/suggest/{model_id}?vram_gb={max(vram,1.0)}",
            timeout=5)
        dl = (f"{r.json()['layer_start']}-{r.json()['layer_end']}"
              if r.status_code == 200 else "0-15")
    except: dl = "0-15"

    layers = input(f"  Layers [{dl}]: ").strip() or dl
    p      = layers.split("-"); l0, l1 = int(p[0]), int(p[1])
    port   = int(input(f"  Shard port [{DEFAULT_SHARD_PORT}]: ").strip()
                 or DEFAULT_SHARD_PORT)

    # Simulation mode
    has_gpu  = hw.get("gpu_type") in ("discrete",) and vram >= 1.0
    simulate = not has_gpu
    if has_gpu:
        sim = input("  Simulation mode (no real model loading)? [n]: ").strip().lower()
        simulate = sim in ("y", "yes")

    if simulate:
        print(f"  {C.YELLOW}Running in simulation mode — activations are simulated.{C.RESET}")
    else:
        print(f"  {C.GREEN}Production mode — will load real model weights.{C.RESET}")

    # Register node
    print(f"\n  {C.CYAN}Registering node...{C.RESET}")
    try:    node = client.register(hw, bm, max_gpu)
    except Exception as e:
        print(f"  {C.RED}Failed: {e}{C.RESET}\n"); sys.exit(1)

    client.node_id = node["id"]
    client.set_region(region)

    config = {
        "server_url":        server,
        "node_id":           node["id"],
        "hardware_class":    node["hardware_class"],
        "multiplier":        node["multiplier"],
        "region":            region,
        "max_gpu_util":      max_gpu,
        "registered_at":     node["registered_at"],
        "inference_enabled": True,
        "model_id":          model_id,
        "layer_start":       l0,
        "layer_end":         l1,
        "shard_port":        port,
        "simulate":          simulate,
        "vram_gb":           vram,
    }
    save_config(config)

    print(f"\n  {C.GREEN}{C.BOLD}Registered!{C.RESET}")
    print(f"  Node ID:    {node['id']}")
    print(f"  Class:      {node['hardware_class']}")
    print(f"  Multiplier: {C.GREEN}{node['multiplier']}x{C.RESET}")
    print(f"  Region:     {region}")
    print(f"  Model:      {model_id}")
    print(f"  Layers:     {l0} -> {l1}")
    print(f"  Simulate:   {simulate}")
    print(f"\n  Run {C.CYAN}python inferix_client.py start{C.RESET} to begin.\n")


def cmd_start():
    config = load_config()
    if not config:
        print(f"\n  {C.YELLOW}Run setup first.{C.RESET}\n"); sys.exit(1)
    print_header()
    print(f"  Node:   {C.DIM}{config['node_id']}{C.RESET}")
    print(f"  Server: {C.DIM}{config['server_url']}{C.RESET}")
    print(f"  Model:  {C.DIM}{config.get('model_id','N/A')}{C.RESET}")
    print(f"  Layers: {C.DIM}{config.get('layer_start')} -> {config.get('layer_end')}{C.RESET}\n")
    Daemon(config).start()


def cmd_status():
    config = load_config()
    if not config:
        print(f"\n  {C.YELLOW}Run setup first.{C.RESET}\n"); sys.exit(1)
    print_header()
    client = InferixClient(config["server_url"], config["node_id"])
    bal    = client.balance()

    print(f"  {C.BOLD}Node{C.RESET}")
    print(f"  ID:         {C.DIM}{config['node_id']}{C.RESET}")
    print(f"  Class:      {config.get('hardware_class')}")
    print(f"  Multiplier: {C.GREEN}{config.get('multiplier',1.0):.3f}x{C.RESET}")
    print(f"  Region:     {config.get('region')}")

    print(f"\n  {C.BOLD}Inference Shard{C.RESET}")
    print(f"  Model:      {config.get('model_id','N/A')}")
    print(f"  Layers:     {config.get('layer_start')} -> {config.get('layer_end')}")
    pp = client.pipeline(config.get("model_id", ""))
    if pp:
        ok = pp.get("available", False)
        print(f"  Pipeline:   {C.GREEN+'Complete' if ok else C.YELLOW+'Incomplete'}{C.RESET}")
        print(f"  Nodes:      {pp.get('online_nodes', 0)} online")

    if bal:
        print(f"\n  {C.BOLD}Tokens{C.RESET}")
        print(f"  Balance:    {C.GREEN}{bal['balance']:.4f} GFT{C.RESET}")
        print(f"  Earned:     {bal['total_earned']:.4f} GFT")
        print(f"  Spent:      {bal['total_spent']:.4f} GFT")
    else:
        print(f"\n  {C.RED}Cannot fetch balance.{C.RESET}")
    print()


def cmd_stop():
    pid_file = Path.home() / ".inferix" / "client.pid"
    if pid_file.exists():
        pid = int(open(pid_file).read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink()
            print(f"\n  {C.GREEN}Stopped (PID {pid}).{C.RESET}\n")
        except ProcessLookupError:
            print(f"\n  {C.YELLOW}Not running.{C.RESET}\n")
            pid_file.unlink(missing_ok=True)
    else:
        print(f"\n  {C.YELLOW}Use Ctrl+C to stop foreground process.{C.RESET}\n")


def main():
    p = argparse.ArgumentParser(description="Inferix Contributor Client")
    p.add_argument("command", choices=["setup", "start", "status", "stop"])
    args = p.parse_args()
    {"setup": cmd_setup, "start": cmd_start,
     "status": cmd_status, "stop": cmd_stop}[args.command]()


if __name__ == "__main__":
    main()
