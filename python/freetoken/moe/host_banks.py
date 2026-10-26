"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import mmap
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class PinFailed(RuntimeError):
    """cudaHostRegister refused a bank: the host is out of pinnable RAM or over its pin quota."""


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).

    DISK extends PAGEABLE to banks whose backing is a repacked bank file (an FTW
    sidecar): the layer never occupies RAM by construction, its pages fault from
    the file on first touch and the kernel reclaims them under pressure (the page
    cache is the RAM tier). A DISK layer can still be pinned at settle time when
    the budget allows -- the registration faults the file pages in and pins them.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"
    DISK = "disk"
    # filled but not settled: probe_pin settles the layer after the load (--moe-cpu-layers auto finds the pin limit)
    DEFERRED = "deferred"


# FREETOKEN_SIMULATE_PIN_CAP_GB fails pins past that many GiB, like a driver pin quota (WSL/WDDM), to exercise the pin probe on hosts without one
_sim_pinned = 0


def _sim_pin_cap() -> int | None:
    value = os.environ.get("FREETOKEN_SIMULATE_PIN_CAP_GB")
    return int(float(value) * 2**30) if value else None


def charge_pin(nbytes: int) -> None:
    """Count ``nbytes`` against the simulated pin cap; raise PinFailed past it (no-op without the cap)."""
    global _sim_pinned
    cap = _sim_pin_cap()
    if cap is None:
        return
    if _sim_pinned + nbytes > cap:
        raise PinFailed(f"simulated pin cap {cap / 2**30:.1f} GiB reached")
    _sim_pinned += nbytes


def refund_pin(nbytes: int) -> None:
    global _sim_pinned
    if _sim_pin_cap() is not None:
        _sim_pinned -= nbytes


_DEFAULT_CHUNK = 8 << 20

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []

# O_DIRECT's Windows counterpart is FILE_FLAG_NO_BUFFERING, which asks for the same three
# alignments (sector-aligned offset, length and buffer) that _BLK already enforces here. A
# positioned read is os.preadv on POSIX and ReadFile with an OVERLAPPED offset on Windows.
# Shared with models/weight.py's parallel shard reader and checkpoint/ftw.py, so these are
# public -- and since both spellings exist, an unbuffered read is not a POSIX-only option.
_POSIX_DIRECT = os.name != "nt"
# Whether this platform can open a file for unbuffered positioned reads at all. Callers gate
# their fast (page-cache-bypassing) read path on this, NOT on the presence of os.O_DIRECT.
DIRECT_READ_SUPPORTED = True

if _POSIX_DIRECT:

    def open_direct(path: str) -> int:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)

    def open_direct_ex(path: str) -> tuple[int, bool]:
        """``(handle, really unbuffered)`` -- what ``open_direct`` returns, plus the answer a
        caller needs on a platform that hands back a buffered handle when direct I/O is refused."""
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True

    def pread_into(fd: int, view: memoryview, offset: int) -> int:
        return os.preadv(fd, [view], offset)

    def close_direct(fd: int) -> None:
        os.close(fd)

    def drop_read_cache(path: str, offset: int = 0, length: int = 0) -> None:
        # never fail a load over a page-cache hint
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _avail_phys_bytes() -> int | None:
        # MemAvailable, not MemFree: it counts the reclaimable page cache a load could take
        # back, which is the figure that decides whether a fill OOMs (MemFree alone reads
        # "no room" on any box that has been reading files all day).
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return None

    def _working_set_bytes() -> int | None:
        return None  # there the lock ceiling is RLIMIT_MEMLOCK, which no resident byte count feeds

else:
    import ctypes.wintypes as _wt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GENERIC_READ = 0x80000000
    _SHARE_ALL = 0x1 | 0x2 | 0x4
    _OPEN_EXISTING = 3
    _NO_BUFFERING = 0x20000000  # not 0x40000000, which is FILE_FLAG_OVERLAPPED
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [_wt.LPCWSTR, _wt.DWORD, _wt.DWORD, ctypes.c_void_p,
                                      _wt.DWORD, _wt.DWORD, ctypes.c_void_p]
    _kernel32.ReadFile.restype = _wt.BOOL
    _kernel32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _wt.DWORD,
                                   ctypes.POINTER(_wt.DWORD), ctypes.c_void_p]
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    # the working-set queries take and return 64-bit sizes; GetCurrentProcess hands back the
    # (HANDLE)-1 pseudo-handle, which ctypes would truncate to an int
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.GetProcessWorkingSetSizeEx.restype = ctypes.c_int
    _kernel32.GetProcessWorkingSetSizeEx.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
    ]
    _kernel32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    _kernel32.SetProcessWorkingSetSizeEx.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32,
    ]
    _kernel32.VirtualLock.restype = ctypes.c_int
    _kernel32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _logged_buffer_fallback = False

    class _Overlapped(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_ulonglong), ("internal_high", ctypes.c_ulonglong),
                    ("offset", _wt.DWORD), ("offset_high", _wt.DWORD), ("key", ctypes.c_void_p)]

    def open_direct_ex(path: str):
        """``(handle, unbuffered)``: a NO_BUFFERING handle, or a buffered one plus ``False``
        where the volume refuses direct I/O."""
        global _logged_buffer_fallback
        for flags in (_NO_BUFFERING, _FILE_ATTRIBUTE_NORMAL):
            handle = _kernel32.CreateFileW(path, _GENERIC_READ, _SHARE_ALL, None,
                                           _OPEN_EXISTING, flags, None)
            if handle is not None and handle != _INVALID_HANDLE:
                direct = flags == _NO_BUFFERING
                if not direct and not _logged_buffer_fallback:
                    _logged_buffer_fallback = True
                    logger.warning("direct I/O unavailable on this volume; reading %s through "
                                   "the page cache (slower, and nothing evicts it afterwards)",
                                   path)
                return handle, direct
        raise ctypes.WinError(ctypes.get_last_error())

    def open_direct(path: str):
        return open_direct_ex(path)[0]

    def pread_into(handle, view: memoryview, offset: int) -> int:
        overlapped = _Overlapped()
        overlapped.offset = offset & 0xFFFFFFFF
        overlapped.offset_high = offset >> 32
        got = _wt.DWORD(0)
        buffer = ctypes.addressof(ctypes.c_char.from_buffer(view))
        if not _kernel32.ReadFile(handle, buffer, len(view), ctypes.byref(got),
                                  ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        return got.value

    def close_direct(handle) -> None:
        _kernel32.CloseHandle(handle)

    def drop_read_cache(path: str, offset: int = 0, length: int = 0) -> None:
        pass  # unbuffered reads never populate the cache, and there is no fadvise here

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dw_length", _wt.DWORD), ("dw_memory_load", _wt.DWORD),
            ("ull_total_phys", ctypes.c_ulonglong), ("ull_avail_phys", ctypes.c_ulonglong),
            ("ull_total_page_file", ctypes.c_ulonglong), ("ull_avail_page_file", ctypes.c_ulonglong),
            ("ull_total_virtual", ctypes.c_ulonglong), ("ull_avail_virtual", ctypes.c_ulonglong),
            ("ull_avail_extended_virtual", ctypes.c_ulonglong),
        ]

    _kernel32.GlobalMemoryStatusEx.restype = _wt.BOOL
    _kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]

    def _avail_phys_bytes() -> int | None:
        status = _MemoryStatusEx()
        status.dw_length = ctypes.sizeof(status)
        if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ull_avail_phys)

    class _ProcessMemoryCounters(ctypes.Structure):
        # the documented 10 fields; the tail is unread but part of the size the API checks
        _fields_ = [
            ("cb", _wt.DWORD), ("page_fault_count", _wt.DWORD),
            ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
            ("quota_peak_paged", ctypes.c_size_t), ("quota_peak_nonpaged", ctypes.c_size_t),
            ("quota_paged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    # kernel32 exports the working-set queries but not the memory counters on this build, and
    # psapi.dll is the documented forwarder for exactly that one call
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _psapi.GetProcessMemoryInfo.restype = _wt.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), _wt.DWORD,
    ]

    def _working_set_bytes() -> int | None:
        """Bytes currently in this process' working set -- what VirtualLock's quota is spent on.

        The page-lock quota is the working set itself, so a bank of 0.8 GiB is refused by a
        1 GiB maximum no matter that nothing is locked yet; sizing the raise without this
        number is what made that look like a missing privilege."""
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.working_set)


def mem_available_bytes() -> int | None:
    """Bytes of RAM a load could realistically take, or ``None`` where the platform cannot say.

    The two spellings are not the same number: Linux's ``MemAvailable`` credits the reclaimable
    page cache, Windows' ``ullAvailPhys`` does not credit the standby list, so the Windows figure
    is the conservative one -- it calls a box tight sooner than the Linux one would."""
    return _avail_phys_bytes()


def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


_UNSET = object()  # lazy direct-fd marker

_PREAD_POOL = None


def _pread_pool():
    """Shared pread workers for file-backed bank row gathers (lazy, daemon threads)."""
    global _PREAD_POOL
    if _PREAD_POOL is None:
        import concurrent.futures

        _PREAD_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="bank-pread"
        )
    return _PREAD_POOL


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.
    * ``"file"`` -- MAP_PRIVATE window over ``[offset, offset + nbytes)`` of ``file``; the
      bytes are never read at load, they fault from the file on first touch (the disk
      tier). ``pin()`` registers the window -- cudaHostRegister faults the file pages in
      and page-locks them, so a pinned file-backed bank is a RAM tier that still matches
      the file. ``release()`` drops the pages back to the file. Banks are read-only after
      conversion, and MAP_PRIVATE keeps any stray write private instead of corrupting the
      checkpoint.

    The buffer is rounded up to the O_DIRECT block; ``tensor`` views exactly ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "_locked",
                 "_fd", "_file_off", "_file_len", "_dfd", "_file_path")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 *, backing: str | None = None, file: str | None = None, offset: int = 0):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda", "file"), backing
        if backing == "file":
            assert file is not None, "file-backed banks need file="
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self._fd = None
        self._file_off = 0
        self._file_len = 0
        self._dfd = _UNSET
        self._file_path = file
        if backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._buf = raw.numpy()[off:off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
        elif backing == "file":
            # mmap requires an aligned mapping start; a per-layer FTW entry is ALIGN-
            # aligned by the writer, but map the enclosing window anyway so unaligned
            # offsets (head-padded flat regions) work too. The window covers exactly
            # [align_down(offset), align_up(offset + nbytes)) -- never the O_DIRECT
            # block rounding, which can run past the end of the file.
            win_off = (offset // _BLK) * _BLK
            win_len = ((offset + self.nbytes - win_off + _BLK - 1) // _BLK) * _BLK
            self._fd = os.open(file, os.O_RDONLY)
            # ACCESS_COPY = MAP_PRIVATE: reads fault from the file, writes stay private
            # (a bank must never write back into the checkpoint).
            self._buf = mmap.mmap(self._fd, win_len, access=mmap.ACCESS_COPY, offset=win_off)
            _LIVE_BUFFERS.append(self._buf)
            head = offset - win_off
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf)) + head
            self._pinned = False
            self._file_off = offset
            self._file_len = self.nbytes
            # a view with element size > 1 needs the head to be a whole number of
            # elements; the disk tier's per-layer entries are page-aligned (head 0),
            # so this only guards a hand-built unaligned window
            assert head % elsize == 0, (offset, head, elsize)
        else:
            self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
            _LIVE_BUFFERS.append(self._buf)
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            self._pinned = False
        # window heads can be unaligned (flat regions carved mid-page), so view through a
        # frombuffer of the whole buffer then slice the head off in bytes
        if backing == "file" and self.addr % _BLK:
            base = torch.frombuffer(self._buf, dtype=torch.uint8, count=len(self._buf))
            self.tensor = base[self.addr % _BLK: self.addr % _BLK + self.nbytes].view(dtype).view(*shape)
        else:
            self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._locked = False

    @property
    def residency(self) -> HostResidency:
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def _buffer_base(self) -> int:
        """Page-aligned base of the whole underlying buffer (the mapping window)."""
        return ctypes.addressof(ctypes.c_char.from_buffer(self._buf))

    def willneed(self) -> None:
        """Ask the kernel to fault this bank's pages in ahead of use (FADV_WILLNEED).

        File-backed banks only: readahead pulls the bytes from the backing file while
        the current layer computes, so the next layer's first touch is a page-cache
        hit instead of an NVMe read. No-op elsewhere (and on platforms without
        posix_fadvise)."""
        if self._fd is None:
            return
        try:
            os.posix_fadvise(self._fd, self._file_off, self._file_len, os.POSIX_FADV_WILLNEED)
        except (AttributeError, OSError):
            pass

    def read_rows(self, row_ids: list[int], row_bytes: int, out_rows) -> None:
        """Parallel read of the given expert rows into ``out_rows`` (a [len(row_ids),
        row_bytes] uint8 numpy view): position ``i`` receives row ``row_ids[i]``.

        Reading through an fd sidesteps the mmap fault storm -- a gather through the
        mapping faults one 4 KiB page at a time on the calling thread while a read
        request per expert keeps the NVMe queue deep. Row-sized requests on a
        block-multiple row go out O_DIRECT (past the page cache: 60 GiB of buffered
        row reads thrash a cache that the pinned banks already starve, and direct
        reads measured 4-5x the buffered rate on the sidecar volume); anything
        unaligned falls back to the buffered fd, where alignment is a perf choice."""
        assert self._fd is not None, "read_rows needs a file-backed bank"
        fd = self._direct_fd()
        if fd is not None:
            base = out_rows.ctypes.data
            if (base % _BLK or row_bytes % _BLK or self._file_off % _BLK
                    or any((e * row_bytes) % _BLK for e in row_ids)):
                fd = None  # an unaligned row set takes the buffered path
        if fd is None:
            fd = self._fd
        pool = _pread_pool()
        futs = [
            pool.submit(os.preadv, fd, [memoryview(out_rows[i])], self._file_off + e * row_bytes)
            for i, e in enumerate(row_ids)
        ]
        for i, f in enumerate(futs):
            got = f.result()
            if got != row_bytes:
                raise RuntimeError(
                    f"short read for expert row {row_ids[i]}: {got} of {row_bytes} bytes"
                )

    def _direct_fd(self):
        """Lazily opened unbuffered fd for read_rows (None where the volume refuses
        O_DIRECT or on platforms without it)."""
        if self._dfd is _UNSET:
            self._dfd = None
            if getattr(os, "O_DIRECT", 0):
                try:
                    self._dfd = os.open(self._file_path, os.O_RDONLY | os.O_DIRECT)
                except OSError:
                    self._dfd = None
        return self._dfd

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        Registration covers the whole mapping window (page-aligned), not ``addr``
        (a file-backed bank can carry an unaligned head inside its window).
        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        from freetoken.kernel.pinned import host_register

        charge_pin(len(self._buf))
        try:
            host_register(self._buffer_base(), len(self._buf))
        except RuntimeError as exc:
            refund_pin(len(self._buf))
            raise PinFailed(f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB") from exc
        self._pinned = True

    def unpin(self) -> None:
        """Undo ``pin()`` on an mmap bank; its pages stay resident (born-pinned banks cannot unpin)."""
        if not self._pinned or not isinstance(self._buf, mmap.mmap):
            return
        from freetoken.kernel.pinned import host_unregister

        host_unregister(self.addr)
        refund_pin(len(self._buf))
        self._pinned = False

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._pinned:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but with no device address, so the GPU cannot gather its rows.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK on POSIX, the working-set quota on Windows) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed, _os_lock_refusal_reason
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self._buffer_base(), len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            _os_lock_refusal_reason = str(exc)
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too
_os_lock_refusal_reason: str | None = None  # what the first refusal said, read by the residency echo


def os_lock_refusal() -> str | None:
    """The first page-lock refusal of this run, or ``None`` while nothing has been refused.

    Lets the log that echoes residency back distinguish 'this platform will not let us lock, and
    that was reported once already' from a bank that settled pageable for some other reason."""
    return _os_lock_refusal_reason


_lock_plan = 0  # bytes this run will keep host-resident; sizes the Windows working-set raise


def set_lock_plan(*, resident_bytes: int = 0) -> None:
    """Tell the lock path how much this run will hold in RAM.

    ``resident_bytes`` sizes the working-set raise: banks settle while later banks are still being read,
    so the working set at the first lock is a fraction of the footprint -- sized on the live number, the
    first bank locks and the second is refused once the model is in."""
    global _lock_plan
    _lock_plan = max(0, int(resident_bytes))


def _os_lock(addr: int, nbytes: int) -> None:
    """Page-lock a bank without spending CUDA pin quota: mlock on POSIX, VirtualLock on Windows.

    Both are bounded by a per-process ceiling the caller cannot see from the address alone --
    RLIMIT_MEMLOCK there, the process working set here -- so each branch tries its own raise
    first and the refusal surfaces from the lock call itself, named by its real ceiling.
    """
    global _os_locked_total
    if os.name == "nt":
        _nt_lock(addr, nbytes)
        return
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


_NT_LOCK_HEADROOM = 1 << 30  # the raise covers one more bank plus room to grow, not just its bytes
_WS_MAX_LIMITED = 0x2  # GetProcessWorkingSetSizeEx: the maximum is imposed (job, or an earlier Ex call)

_nt_quota_ceiling = 0  # working-set maximum the OS actually granted, read back after the last raise
_nt_quota_job_capped = False


def _nt_quota_request(nbytes: int, locked_total: int, working_set: int, planned: int = 0) -> int:
    """The working-set maximum that makes one more lock of ``nbytes`` grantable.

    VirtualLock charges its quota against the whole working set, not against the bytes locked so
    far, so a server holding 65 GiB resident needs the maximum above 65 GiB before its first bank
    locks. Sizing the raise on the locked bytes alone refuses that first bank and blames the
    privilege, which is the dead end this used to report. The live working set is itself only a
    snapshot -- banks settle while the loader is still reading, so ``planned``, what the run will
    hold once the experts are in, sets the floor the snapshot cannot go below."""
    return max(working_set, planned) + locked_total + nbytes + _NT_LOCK_HEADROOM


def _nt_working_set_limits() -> tuple[int, int, int] | None:
    """This process' working-set (minimum, maximum, flags), or ``None`` when the OS won't say."""
    process = _kernel32.GetCurrentProcess()
    minimum, maximum, flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_uint32()
    if not _kernel32.GetProcessWorkingSetSizeEx(
        process, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags), 0
    ):
        return None
    return int(minimum.value), int(maximum.value), int(flags.value)


def _nt_set_working_set_max(minimum: int, maximum: int) -> bool:
    """Ask for a new working-set range; only the maximum is ever moved by us."""
    return bool(_kernel32.SetProcessWorkingSetSizeEx(_kernel32.GetCurrentProcess(), minimum, maximum, 0))


def _nt_raise_working_set_quota(want: int) -> int:
    """Raise the working-set maximum toward ``want`` and return the ceiling the OS granted.

    Only the maximum moves (SeProfileSingleProcessPrivilege); the minimum is handed back
    unchanged, as raising that needs a privilege a normal server process lacks. The read-back
    value is the quota the lock then faces, and a job object can hold it below ``want``, so the
    request is only worth what the grant says."""
    global _nt_quota_ceiling, _nt_quota_job_capped
    limits = _nt_working_set_limits()
    if limits is None:
        return _nt_quota_ceiling
    minimum, maximum, flags = limits
    if maximum < want and _nt_set_working_set_max(minimum, want):
        reread = _nt_working_set_limits()
        if reread is not None:
            minimum, maximum, flags = reread
    _nt_quota_ceiling = maximum
    _nt_quota_job_capped = bool(flags & _WS_MAX_LIMITED)
    return _nt_quota_ceiling


def _nt_lock_refusal(err: int, nbytes: int, ceiling: int, want: int) -> str:
    """Why VirtualLock said no, in terms of the two ceilings that can be behind it.

    The ceiling is named whichever way it went: saying the raise succeeded is the only way a reader
    can tell a job-capped maximum apart from a refusal that is purely the missing right."""
    msg = f"VirtualLock({nbytes / 2**30:.1f} GiB): WinError {err}"
    if ceiling:
        if ceiling < want:
            capped = " (imposed by the job this process runs in)" if _nt_quota_job_capped else ""
            msg += (
                f": the page-lock quota is the process working set, and the OS granted "
                f"{ceiling / 2**30:.1f} GiB of the {want / 2**30:.1f} GiB the resident banks need"
                f"{capped}"
            )
        else:
            msg += (
                f": the working-set quota was raised to {ceiling / 2**30:.1f} GiB, at or past the "
                f"{want / 2**30:.1f} GiB this run will hold, so no quota refused this"
            )
    return msg + (
        " - the 'Lock pages in memory' right lifts that quota entirely (secpol.msc -> Local "
        "Policies -> User Rights Assignment, then a new login session); without it every "
        "host-locked layer stays pageable"
    )


def _nt_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    want = _nt_quota_request(nbytes, _os_locked_total, _working_set_bytes() or 0, _lock_plan)
    ceiling = _nt_raise_working_set_quota(want)
    if not _kernel32.VirtualLock(ctypes.c_void_p(addr), nbytes):
        err = ctypes.get_last_error()
        raise OSError(err, _nt_lock_refusal(err, nbytes, ceiling, want))
    _os_locked_total += nbytes


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = ("labels", "applied", "has_unpinned", "actual", "deferred")

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}
        # DEFERRED layers' filled banks, for probe_pin
        self.deferred: dict[int, list[HostBank]] = {}

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


@contextlib.contextmanager
def requested_residency(labels: list[str] | None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins)."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def _settle(bank: HostBank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap).

    DISK leaves the bank alone: file-backed banks fault their pages from the backing
    file on demand (the disk tier), and pinning them is a deliberate later step the
    budget-driven split performs, not the settle default."""
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()


def pin_banks(banks: dict[str, HostBank | list[HostBank] | dict[int, HostBank]]) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries (indexed by layer id) and dict-valued ones (layer id -> bank) are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin."""
    plan = _requested_residency
    for bank in banks.values():
        if isinstance(bank, (list, dict)):
            for layer_id, layer_bank in bank.items() if isinstance(bank, dict) else enumerate(bank):
                residency = (
                    HostResidency.PINNED.value if plan is None
                    else plan.residency_for(layer_id)
                )
                if residency == HostResidency.DEFERRED.value:
                    plan.deferred.setdefault(layer_id, []).append(layer_bank)
                    continue
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
        else:
            bank.pin()


def _wddm() -> bool:
    return sys.platform == "win32" or (hasattr(os, "uname") and "microsoft" in os.uname().release.lower())


def pin_capped() -> bool:
    """Whether CUDA pinning is known to be capped: WDDM (WSL and native Windows), or FREETOKEN_SIMULATE_PIN_CAP_GB."""
    return _wddm() or _sim_pin_cap() is not None


def _probe_margin() -> int:
    """FREETOKEN_PIN_PROBE_MARGIN_GB, else 3 GiB under WDDM (WSL and native Windows)."""
    return int(float(os.environ.get("FREETOKEN_PIN_PROBE_MARGIN_GB", "3" if _wddm() else "0")) * 2**30)


def probe_pin(plan: _ResidencyPlan, num_layers: int, margin_bytes: int | None = None) -> list[str]:
    """Settle the DEFERRED layers: pin middle layers first until a pin fails, OS-lock the rest; returns each layer's residency.

    After a refusal, the last pinned layers give back at least ``margin_bytes`` (default ``_probe_margin()``)."""
    if margin_bytes is None:
        margin_bytes = _probe_margin()
    # per-layer decode miss rates are U-shaped, so the head and tail layers are the ones left unpinned
    order = sorted(plan.deferred, key=lambda l: (abs(l - (num_layers - 1) / 2), l))
    pinned: list[int] = []
    refused = None
    for layer in order:
        banks = plan.deferred[layer]
        try:
            for bank in banks:
                bank.pin()
        except PinFailed as exc:
            for bank in banks:
                bank.unpin()
            refused = exc
            break
        pinned.append(layer)
    if refused is not None:
        freed = 0
        while pinned and freed < margin_bytes:
            layer = pinned.pop()
            for bank in plan.deferred[layer]:
                freed += len(bank._buf) if bank._pinned else 0
                bank.unpin()
    labels = [HostResidency.PINNED.value] * num_layers
    kept = set(pinned)
    for layer, banks in plan.deferred.items():
        if layer in kept:
            continue
        for bank in banks:
            bank.lock()
        labels[layer] = (
            HostResidency.LOCKED.value if all(b.residency is HostResidency.LOCKED for b in banks) else HostResidency.PAGEABLE.value
        )
    return labels


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            bank, residency, plan, layer_id = item
            if residency == HostResidency.DEFERRED.value:
                plan.deferred.setdefault(layer_id, []).append(bank)
                continue
            try:
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank, residency: str = HostResidency.PINNED.value,
               plan=None, layer_id: int | None = None) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        drop_read_cache(path)
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = open_direct(path)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        pread_into(fd, mv[o:o + want], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        close_direct(fd)
    return size


def _preadv_all(fd, dst: memoryview, offset: int, need: int) -> None:
    """Positioned read into ``dst`` until ``need`` bytes have landed; direct I/O may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = pread_into(fd, dst[done:], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.
    """
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        drop_read_cache(path, file_offset, nbytes)
    fd = open_direct(path)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        close_direct(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "DIRECT_READ_SUPPORTED",
    "alloc_banks",
    "born_pinned_default",
    "close_direct",
    "drop_read_cache",
    "mem_available_bytes",
    "open_direct",
    "open_direct_ex",
    "pin_banks",
    "pread_into",
    "read_file_into",
    "read_range_into",
    "requested_residency",
]
