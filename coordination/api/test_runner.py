from fastapi import APIRouter
import subprocess
import sys

router = APIRouter(prefix="/tests", tags=["Tests"])


@router.get("/run")
def run_tests():
    """
    Run the test suite and return results.
    Only available in DEBUG mode.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "coordination/tests/",
         "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd="/app"
    )
    lines   = result.stdout.split("\n")
    summary = [l for l in lines if l.strip()]

    return {
        "exit_code": result.returncode,
        "passed":    result.returncode == 0,
        "output":    summary,
        "errors":    result.stderr.split("\n") if result.stderr else [],
    }
