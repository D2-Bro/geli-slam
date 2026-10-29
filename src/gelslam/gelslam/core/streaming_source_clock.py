"""Pair source timing topics during reception without opening a bag file."""

from collections import defaultdict, deque
import json
from threading import RLock

import numpy as np

from gelslam.core.recorded_source_clock import DeviceMillisecondClock, _stamp_ns


class _StartupClock:
    """Freeze a robust host offset after warmup; retain the source clock rate.

    Both source clocks report SI seconds. A short online startup cannot estimate
    their rate reliably from receipt jitter. Do not refit already-used times.
    """

    def __init__(self, warmup_s):
        self.warmup_s = warmup_s
        self.anchor = None
        self.previous = None
        self.offsets = []
        self.origin = None

    def observe(self, source, host):
        if not np.isfinite(source) or not np.isfinite(host):
            raise ValueError("source/host times must be finite")
        if self.previous is not None and source <= self.previous:
            raise ValueError("source times are not increasing")
        self.previous = source
        if self.anchor is None:
            self.anchor = source
        if self.origin is None:
            self.offsets.append(host - (source - self.anchor))
            if len(self.offsets) >= 3 and source - self.anchor >= self.warmup_s - 1e-9:
                self.origin = float(np.median(self.offsets))
                self.offsets.clear()

    def at(self, source):
        return self.origin + (source - self.anchor)


class StreamingSourceClock:
    """Pair camera timing by header and headerless IMU timing by FIFO order.

    Start subscriptions before publishing/replaying both IMU topics. They must
    deliver the same measurements in the same order without independent loss.
    A bare UInt32 cannot identify a dropped IMU partner. Callers use reliable
    queues and serialize the two IMU callbacks. Camera drops remain matchable
    through host_wall_ns. No ROS-header fallback is used for integration.
    """

    def __init__(self, warmup_s=1.0, max_pending=4096):
        if not np.isfinite(warmup_s) or warmup_s <= 0:
            raise ValueError("recorded_source_warmup_s must be positive")
        self.max_pending = int(max_pending)
        if self.max_pending < 3:
            raise ValueError("source timing queue must hold at least three samples")
        self._lock = RLock()
        self._camera_clock = _StartupClock(warmup_s)
        self._imu_clock = _StartupClock(warmup_s)
        self._device_clock = DeviceMillisecondClock()
        self._images = defaultdict(deque)
        self._camera_times = defaultdict(deque)
        self._imus, self._device_times = deque(), deque()
        self._warm_images, self._warm_imus = deque(), deque()
        self._ready_images, self._ready_imus = deque(), deque()
        self._latest_image_ns = None
        self._latest_camera_ns = None
        self.error = None
        self.image_count = self.imu_count = 0

    @property
    def ready(self):
        return (self.error is None and self._camera_clock.origin is not None
                and self._imu_clock.origin is not None)

    def _check(self):
        if self.error is not None:
            raise ValueError(self.error)

    def _bound(self):
        pending = (sum(map(len, self._images.values()))
                   + sum(map(len, self._camera_times.values()))
                   + len(self._imus) + len(self._device_times)
                   + len(self._warm_images) + len(self._warm_imus)
                   + len(self._ready_images) + len(self._ready_imus))
        if pending > self.max_pending:
            self.error = "source timing queue overflow; check both timing topics and sensor delivery"
            raise ValueError(self.error)

    def _release(self):
        if self.ready:
            for message, source in self._warm_imus:
                self._ready_imus.append((message, self._imu_clock.at(source)))
            for message, source in self._warm_images:
                self._ready_images.append((message, self._camera_clock.at(source)))
            self._warm_imus.clear()
            self._warm_images.clear()
        self._bound()

    def _pair_camera(self, stamp):
        images, times = self._images.get(stamp), self._camera_times.get(stamp)
        while images and times:
            image, source = images.popleft(), times.popleft()
            self._camera_clock.observe(source, stamp * 1e-9)
            self._warm_images.append((image, source))
            self.image_count += 1
        if not images:
            self._images.pop(stamp, None)
        if not times:
            self._camera_times.pop(stamp, None)

    def add_image(self, message):
        with self._lock:
            self._check()
            stamp = _stamp_ns(message.header)
            self._latest_image_ns = stamp
            # The image subscription may discard older images while tracking.
            for old in list(self._camera_times):
                if old < stamp:
                    del self._camera_times[old]
            if (self._latest_camera_ns is not None and stamp < self._latest_camera_ns
                    and stamp not in self._camera_times):
                return  # Its timing message was already skipped.
            self._images[stamp].append(message)
            try:
                self._pair_camera(stamp)
                self._release()
            except ValueError as error:
                self.error = str(error)
                raise

    def add_camera_timing(self, message):
        with self._lock:
            self._check()
            try:
                timing = json.loads(message.data)
                stamp = int(timing['host_wall_ns'])
                source = int(timing['source_pts_ns']) * 1e-9
                self._latest_camera_ns = stamp
                for old in list(self._images):
                    if old < stamp:
                        del self._images[old]
                if (self._latest_image_ns is not None and stamp < self._latest_image_ns
                        and stamp not in self._images):
                    return
                self._camera_times[stamp].append(source)
                self._pair_camera(stamp)
                self._release()
            except (ValueError, KeyError, TypeError) as error:
                self.error = "invalid camera source timing: %s" % error
                raise ValueError(self.error) from error

    def _pair_imu(self):
        try:
            while self._imus and self._device_times:
                message = self._imus.popleft()
                source = self._device_clock.unwrap(self._device_times.popleft()) * 1e-3
                self._imu_clock.observe(source, _stamp_ns(message.header) * 1e-9)
                self._warm_imus.append((message, source))
                self.imu_count += 1
            self._release()
        except ValueError as error:
            self.error = str(error)
            raise

    def add_imu(self, message):
        with self._lock:
            self._check()
            self._imus.append(message)
            self._pair_imu()

    def add_imu_timing(self, message):
        with self._lock:
            self._check()
            self._device_times.append(int(message.data))
            self._pair_imu()

    def take_imus(self):
        with self._lock:
            result = list(self._ready_imus) if self.ready else []
            self._ready_imus.clear()
            return result

    def take_images(self):
        with self._lock:
            result = list(self._ready_images) if self.ready else []
            self._ready_images.clear()
            return result
