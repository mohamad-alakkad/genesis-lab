"""OS-enforced resource guard for every heavy GENESIS process.

Windows: a Job Object with a job-wide committed-memory limit, BELOW_NORMAL priority and a CPU
affinity mask limited to half the logical cores. Children inherit the job. The job has no
KILL_ON_JOB_CLOSE, so a detached trainer outlives the launcher and keeps its limits.
Linux: systemd-run --scope with MemoryMax (if available), plus nice 10.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

LIMIT_MB = 1800

if os.name == "nt":
    from ctypes import wintypes as W

    _k = ctypes.WinDLL("kernel32", use_last_error=True)
    _nt = ctypes.WinDLL("ntdll")

    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

    class _IO(ctypes.Structure):
        _fields_ = [("c", ctypes.c_uint64 * 6)]

    class _EXT(ctypes.Structure):
        _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k.CreateJobObjectW.restype = W.HANDLE
    _k.OpenProcess.restype = W.HANDLE
    JOB_OBJECT_LIMIT_AFFINITY = 0x10
    JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
    BELOW_NORMAL = 0x4000
    CREATE_SUSPENDED = 0x4
    DETACHED_PROCESS = 0x8
    CREATE_NEW_PROCESS_GROUP = 0x200
    CREATE_NO_WINDOW = 0x08000000


def allowed_cores() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def mechanism() -> str:
    if os.name == "nt":
        return "windows-job-object"
    return "systemd-run-scope" if shutil.which("systemd-run") else "none(watchdog-only)"


def _make_job(limit_mb: int):
    job = _k.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
    n = os.cpu_count() or 2
    cores = allowed_cores()
    mask = ((1 << cores) - 1) << (n - cores)  # use the highest cores
    info = _EXT()
    info.Basic.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PRIORITY_CLASS | JOB_OBJECT_LIMIT_AFFINITY
    info.Basic.PriorityClass = BELOW_NORMAL
    info.Basic.Affinity = mask
    info.JobMemoryLimit = limit_mb * 1024 * 1024
    if not _k.SetInformationJobObject(W.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    return job


def thread_env(env: dict | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    c = str(allowed_cores())
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = c
    env["GENESIS_MEM_LIMIT_MB"] = str(LIMIT_MB)
    env["GENESIS_MEM_MECHANISM"] = mechanism()
    return env


def launch(cmd: list[str], *, limit_mb: int = LIMIT_MB, detach: bool = False, log_path: str | None = None,
           cwd: str | None = None) -> subprocess.Popen:
    """Start cmd inside the memory-limited container. Returns the Popen."""
    env = thread_env()
    out = open(log_path, "ab") if log_path else (subprocess.DEVNULL if detach else None)
    if os.name == "nt":
        flags = CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        if detach:
            flags = CREATE_SUSPENDED | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None,
                             stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        job = _make_job(limit_mb)
        ph = _k.OpenProcess(0x1FFFFF, False, p.pid)
        try:
            if not _k.AssignProcessToJobObject(W.HANDLE(job), W.HANDLE(ph)):
                err = ctypes.get_last_error()
                p.kill()
                raise OSError(err, "AssignProcessToJobObject failed")
            _nt.NtResumeProcess(W.HANDLE(ph))
        finally:
            _k.CloseHandle(W.HANDLE(ph))
        p._genesis_job = job  # keep handle alive while the launcher lives; the job survives via its process
        return p
    pre = ["nice", "-n", "10"]
    if shutil.which("systemd-run"):
        pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit_mb}M",
               "-p", "MemorySwapMax=0"] + pre
    return subprocess.Popen(pre + cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None,
                            stdin=subprocess.DEVNULL, start_new_session=detach)


if __name__ == "__main__":  # python -m genesis.jobguard -- <cmd...>  (foreground, returns child exit code)
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print("usage: python -m genesis.jobguard -- <command ...>")
        sys.exit(2)
    sys.exit(launch(args).wait())
