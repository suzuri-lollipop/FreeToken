"""Windows process helpers for the supervisor, on raw ctypes (kernel32/ntdll/shell32).

POSIX reaches these through ``/proc`` and process groups; the Win32 equivalents are toolhelp
snapshots (process tree), GetProcessTimes (a PID-reuse-stable identity), a PEB read (argv), and
TerminateProcess (stop: Windows has no SIGTERM, so the daemon's graceful path is the HTTP
prepare-stop and this module is the backstop). Every function is best-effort, mirroring
``osproc``: no raise on a vanished or protected pid, just a safe default.

Only meaningful on ``os.name == "nt"``; importing the module anywhere is fine, calling it off
Windows raises RuntimeError.
"""

from __future__ import annotations

import ctypes
import functools
import os

_SYNCHRONIZE = 0x00100000
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_WAIT_OBJECT_0 = 0
_ERROR_ACCESS_DENIED = 5
_INVALID_PARAMETER = 87
_MAX_PIDS = 4096  # sanity cap: a serve tree this large means the walk went wrong somewhere


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@functools.lru_cache(maxsize=1)
def _api():
    """The bound Win32 surface, or None off Windows (every helper degrades to its default)."""
    if os.name != "nt":
        return None
    import types

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    shell32 = ctypes.WinDLL("shell32")
    vp = ctypes.c_void_p
    pid32 = ctypes.c_uint32

    k32.OpenProcess.restype = vp
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, pid32]
    k32.CloseHandle.restype = ctypes.c_int
    k32.CloseHandle.argtypes = [vp]
    k32.WaitForSingleObject.restype = ctypes.c_uint32
    k32.WaitForSingleObject.argtypes = [vp, ctypes.c_uint32]
    k32.GetLastError.restype = ctypes.c_uint32
    k32.TerminateProcess.restype = ctypes.c_int
    k32.TerminateProcess.argtypes = [vp, ctypes.c_uint32]
    k32.GetProcessTimes.restype = ctypes.c_int
    k32.GetProcessTimes.argtypes = [
        vp,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    k32.CreateToolhelp32Snapshot.restype = vp
    k32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, pid32]
    k32.Process32FirstW.restype = ctypes.c_int
    k32.Process32FirstW.argtypes = [vp, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.restype = ctypes.c_int
    k32.Process32NextW.argtypes = [vp, ctypes.POINTER(PROCESSENTRY32W)]
    k32.ReadProcessMemory.restype = ctypes.c_int
    k32.ReadProcessMemory.argtypes = [
        vp, vp, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    ntdll.NtQueryInformationProcess.argtypes = [vp, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    # K32GetProcessMemoryInfo is exported from kernel32 since Vista; psapi.dll forwards to it.
    k32.K32GetProcessMemoryInfo.restype = ctypes.c_int
    k32.K32GetProcessMemoryInfo.argtypes = [vp, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_uint32]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell32.LocalFree.restype = vp
    shell32.LocalFree.argtypes = [vp]

    api = types.SimpleNamespace(
        k32=k32, ntdll=ntdll, shell32=shell32,
        ERROR_ACCESS_DENIED=_ERROR_ACCESS_DENIED,
        INVALID_PARAMETER=_INVALID_PARAMETER,
    )
    return api


# --------------------------------------------------------------------------- liveness / identity


def pid_alive(pid: int) -> bool:
    api = _api()
    if api is None or pid <= 0:
        return False
    k32 = api.k32
    # SYNCHRONIZE opens even for other users' processes (no admin), so an access-denied-free
    # handle + a zero-timeout wait distinguishes running / exited / gone.
    h = k32.OpenProcess(_SYNCHRONIZE, 0, pid)
    if not h:
        err = k32.GetLastError()
        if err == api.ERROR_ACCESS_DENIED:
            return True  # exists, we just cannot observe it
        if err == api.INVALID_PARAMETER:
            return False
        # A live process whose security descriptor forbids even SYNCHRONIZE: assume alive.
        return True
    try:
        return k32.WaitForSingleObject(h, 0) != _WAIT_OBJECT_0
    finally:
        k32.CloseHandle(h)


def read_starttime(pid: int) -> int | None:
    """Process creation time as FILETIME (100ns ticks since 1601). Units do not matter: callers
    only compare for equality, which is what makes it the PID-reuse guard off ``/proc``."""
    api = _api()
    if api is None:
        return None
    k32 = api.k32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not h:
        return None
    try:
        creation = ctypes.c_ulonglong(0)
        exit_t = ctypes.c_ulonglong(0)
        kernel = ctypes.c_ulonglong(0)
        user = ctypes.c_ulonglong(0)
        if not k32.GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(exit_t), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return creation.value
    finally:
        k32.CloseHandle(h)


def working_set_bytes(pid: int) -> int:
    """The process working set (the closest Windows analog of ``/proc`` PSS; not shared-page
    fractional, so slightly conservative upward). 0 if unavailable."""
    api = _api()
    if api is None:
        return 0
    k32 = api.k32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not h:
        return 0
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    finally:
        k32.CloseHandle(h)


# --------------------------------------------------------------------------- process tree


def snapshot_ppid_map() -> dict[int, int] | None:
    """Every live pid -> its parent pid, or None when the snapshot failed."""
    api = _api()
    if api is None:
        return None
    k32 = api.k32
    snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID_HANDLE:
        return None
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        out: dict[int, int] = {}
        if not k32.Process32FirstW(snap, ctypes.byref(entry)):
            return None
        while True:
            out[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not k32.Process32NextW(snap, ctypes.byref(entry)):
                break
        return out
    finally:
        k32.CloseHandle(snap)


def build_tree(root_pid: int, ppid_of: dict[int, int]) -> list[int]:
    """root_pid plus every transitive child, from a pid->parent map (pure; unit-testable).
    [] when the map has no entry for the root, so callers can tell "unknown pid" from "no
    children"."""
    if root_pid not in ppid_of:
        return []
    children: dict[int, list[int]] = {}
    for pid, ppid in ppid_of.items():
        children.setdefault(ppid, []).append(pid)
    out = [root_pid]
    queue = [root_pid]
    seen = {root_pid}
    while queue and len(out) < _MAX_PIDS:
        node = queue.pop()
        for child in children.get(node, ()):
            if child not in seen:
                seen.add(child)
                out.append(child)
                queue.append(child)
    return out


def tree_pids(root_pid: int) -> list[int]:
    """The serve and its mp-spawned workers: on Windows the spawn children are ordinary
    descendants (``spawn`` re-execs python), so the toolhelp parent-chain captures the tree."""
    ppid_of = snapshot_ppid_map()
    if ppid_of is None:
        return [root_pid] if pid_alive(root_pid) else []
    tree = build_tree(root_pid, ppid_of)
    if root_pid not in tree and pid_alive(root_pid):
        tree.insert(0, root_pid)
    return tree


def signal_tree(pid: int, exit_code: int) -> None:
    """Terminate ``pid`` and every descendant. Windows has no graceful POSIX-signal delivery;
    the daemon's orderly stop is the HTTP prepare-stop, this is the backstop the SIGTERM/SIGKILL
    escalation calls into. Never raises."""
    api = _api()
    if api is None:
        return
    k32 = api.k32
    ppid_of = snapshot_ppid_map()
    targets = (build_tree(pid, ppid_of) if ppid_of else []) or [pid]
    for target in targets:
        try:
            h = k32.OpenProcess(_PROCESS_TERMINATE, 0, target)
            if not h:
                continue
            try:
                k32.TerminateProcess(h, ctypes.c_uint32(int(exit_code) & 0xFFFFFFFF))
            finally:
                k32.CloseHandle(h)
        except OSError:  # pragma: no cover - defensive
            pass


# --------------------------------------------------------------------------- argv (best-effort)


def read_cmdline(pid: int) -> list[str]:
    """The process command line, split into argv exactly as Windows does it (CommandLineToArgvW).

    Reads the target's PEB -> RTL_USER_PROCESS_PARAMETERS -> CommandLine (UTF-16) through
    ReadProcessMemory. Same-bitness and offset assumptions only hold for children we spawn
    (same venv python), which is the caller that cares; anything else returns [] and the
    daemon degrades to a liveness-only check. Raises nothing."""
    api = _api()
    if api is None or ctypes.sizeof(ctypes.c_void_p) != 8:
        return []  # recipe below is the x64 one
    k32 = api.k32
    h = k32.OpenProcess(_PROCESS_VM_READ | _PROCESS_QUERY_INFORMATION, 0, pid)
    if not h:
        return []
    try:
        # PROCESS_BASIC_INFORMATION: four pointer-sized fields, PebBaseAddress is the third.
        pbi = (ctypes.c_size_t * 6)()
        ret_len = ctypes.c_uint32(0)
        if api.ntdll.NtQueryInformationProcess(h, 0, pbi, ctypes.sizeof(pbi), ctypes.byref(ret_len)) != 0:
            return []
        peb = pbi[2]
        if not peb:
            return []

        def _read(addr: int, size: int) -> bytes | None:
            buf = ctypes.create_string_buffer(size)
            got = ctypes.c_size_t(0)
            if not k32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, size, ctypes.byref(got)):
                return None
            if got.value != size:
                return None
            return buf.raw

        def _read_ptr(addr: int) -> int | None:
            raw = _read(addr, 8)
            return int.from_bytes(raw, "little") if raw else None

        # x64 PEB: ProcessParameters at 0x800; RTL_USER_PROCESS_PARAMETERS: CommandLine
        # UNICODE_STRING at 0x70 (Length u16 @0x70, Buffer ptr @0x78).
        params = _read_ptr(peb + 0x800)
        if not params:
            return []
        raw = _read(params + 0x70, 2)
        if not raw:
            return []
        length = int.from_bytes(raw, "little")
        if length == 0 or length > 32768:
            return []
        buf_addr = _read_ptr(params + 0x78)
        if not buf_addr:
            return []
        raw = _read(buf_addr, length)
        if not raw:
            return []
        line = raw.decode("utf-16-le", "replace")
        argc = ctypes.c_int(0)
        argv_ptr = api.shell32.CommandLineToArgvW(line, ctypes.byref(argc))
        if not argv_ptr:
            return [line]
        try:
            return [argv_ptr[i] for i in range(argc.value) if argv_ptr[i]]
        finally:
            api.shell32.LocalFree(argv_ptr)
    except Exception:  # noqa: BLE001 - a protected, exiting, or foreign-bitness process
        return []
    finally:
        k32.CloseHandle(h)
