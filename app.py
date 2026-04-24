from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
import uuid
import subprocess
import sys
import json
import os
import time

app = FastAPI(title="AML Detection API")

BASE_DIR = Path("./runs")
BASE_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


# 🚀 UPLOAD + START PIPELINE
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    run_id = str(uuid.uuid4())
    run_dir = BASE_DIR / run_id
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / file.filename

    # save file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logfile = run_dir / "pipeline.log"
    status_file = run_dir / ".status.json"

    cmd = [
        PYTHON,
        "-u",
        str(Path(__file__).parent / "pipeline.py"),
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_dir.resolve()),
    ]

    logfh = open(logfile, "ab")

    proc = subprocess.Popen(
        cmd,
        stdout=logfh,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent)
    )

    status = {
        "pid": proc.pid,
        "status": "running",
        "started_at": time.time(),
    }
    status_file.write_text(json.dumps(status))

    return {
        "run_id": run_id,
        "status": "started",
        "log_endpoint": f"/logs/{run_id}",
        "status_endpoint": f"/status/{run_id}"
    }


# 📊 STATUS ENDPOINT
@app.get("/status/{run_id}")
def status(run_id: str):
    run_dir = BASE_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")

    status_file = run_dir / ".status.json"
    logfile = run_dir / "pipeline.log"
    output_dir = run_dir / "output"

    status = {}
    if status_file.exists():
        status = json.loads(status_file.read_text())

    pid = status.get("pid")
    alive = _is_pid_alive(pid) if pid else False

    # 🔥 Detect completion properly
    if pid and not alive:
        if "finished_at" not in status:
            # check if predictions exist → success
            if (output_dir / "predictions.csv").exists():
                status["status"] = "success"
            else:
                status["status"] = "failed"

            status["finished_at"] = time.time()
            status_file.write_text(json.dumps(status))

    # Artefacts
    artefacts = {}
    for name, path in {
        "predictions": output_dir / "predictions.csv",
        "graph": output_dir / "graph",
        "visualisation": output_dir / "visualisation",
    }.items():
        if path.exists():
            artefacts[name] = str(path.resolve())

    return {
        "run_id": run_id,
        "status": status.get("status"),
        "alive": alive,
        "artefacts": artefacts
    }


# 📜 LOG STREAM (VERY USEFUL)
@app.get("/logs/{run_id}")
def get_logs(run_id: str):
    logfile = BASE_DIR / run_id / "pipeline.log"
    if not logfile.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    with open(logfile, "r", errors="ignore") as f:
        lines = f.readlines()[-200:]  # last 200 lines

    return {"logs": lines}


# 📥 DOWNLOAD RESULTS
@app.get("/download/{run_id}")
def download_predictions(run_id: str):
    file = BASE_DIR / run_id / "output" / "predictions.csv"

    if not file.exists():
        raise HTTPException(status_code=404, detail="Predictions not ready")

    return FileResponse(file, filename="predictions.csv")


# ❌ OPTIONAL: KILL JOB
@app.post("/kill/{run_id}")
def kill_job(run_id: str):
    status_file = BASE_DIR / run_id / ".status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="run not found")

    status = json.loads(status_file.read_text())
    pid = status.get("pid")

    if pid and _is_pid_alive(pid):
        os.kill(pid, 9)
        status["status"] = "killed"
        status_file.write_text(json.dumps(status))
        return {"message": "killed"}

    return {"message": "already stopped"}


# 🖥️ DASHBOARD — serves index.html at /dashboard
@app.get("/dashboard")
def serve_dashboard():
    dashboard_path = Path(__file__).parent / "index.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found next to app.py")
    return FileResponse(str(dashboard_path))