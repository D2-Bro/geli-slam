"""One process, three explicitly owned estimator workers.

Python callbacks and CustomFactors may hold the GIL. Threads separate ownership
and scheduling; they do not promise simultaneous CPU execution or real time.
"""
from collections import deque
from threading import Condition, Thread
from time import monotonic
from traceback import format_exc
from gelslam.core.shared_map import SharedMap


class SlamSystem:
    def __init__(self, tracking, local_mapping, loop_search=None, on_solution=None,
                 shared_map=None, max_pending_inputs=100000):
        self.shared_map = shared_map if shared_map is not None else SharedMap()
        self.tracking = tracking
        self.local_mapping = local_mapping
        self.loop_search = loop_search
        self.on_solution = on_solution
        if max_pending_inputs < 1:
            raise ValueError('max_pending_inputs must be positive')
        self.max_pending_inputs = max_pending_inputs
        self._condition = Condition()
        self._inputs = deque()
        self._image = None
        self._loops = deque()
        self._accepting = True
        self._tracking_done = False
        self._loop_done = loop_search is None
        self._failure = None
        self._threads = []
        self.metrics = dict(images_replaced=0, images_processed=0, input_peak=0,
                            tracking_seconds=[], ba_seconds=[], loop_seconds=[],
                            image_queue_age_seconds=[], feedback_delay_seconds=[])

    def start(self):
        if self._threads:
            raise RuntimeError('SLAM workers already started')
        for name, target in (('tracking', self._track), ('local-ba', self._map),
                             ('loop-search', self._loop)):
            if name == 'loop-search' and self.loop_search is None:
                continue
            thread = Thread(name='gelslam-' + name, target=self._guard,
                            args=(target,), daemon=False)
            self._threads.append(thread)
            thread.start()

    def _guard(self, operation):
        try:
            operation()
        except BaseException as error:
            with self._condition:
                if self._failure is None:
                    self._failure = (error, format_exc())
                self._accepting = False
                self._condition.notify_all()

    def raise_if_failed(self):
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeError('SLAM worker failed:\n' + failure[1]) from failure[0]

    def submit_input(self, callback, *args):
        with self._condition:
            if not self._accepting:
                return False
            if len(self._inputs) >= self.max_pending_inputs:
                raise RuntimeError('SLAM input queue exceeded configured retention bound')
            self._inputs.append((callback, args))
            self.metrics['input_peak'] = max(self.metrics['input_peak'], len(self._inputs))
            self._condition.notify_all()
            return True

    def submit_image(self, image):
        with self._condition:
            if not self._accepting:
                return False
            if self._image is not None:
                self.metrics['images_replaced'] += 1
            self._image = (monotonic(), image)
            self._condition.notify_all()
            return True

    def accept_keyframes(self, packets):
        # Called only by tracking. All keyframes are retained, in order.
        for packet in packets:
            self.shared_map.append_keyframe(packet)
        with self._condition:
            self._condition.notify_all()

    def _track(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._failure is not None or
                        self._inputs or self._image is not None or not self._accepting)
                    if self._failure is not None:
                        return
                    inputs, self._inputs = self._inputs, deque()
                    image, self._image = self._image, None
                    done = not self._accepting
                for callback, args in inputs:
                    callback(*args)
                if image is not None:
                    queued, payload = image
                    self.metrics['image_queue_age_seconds'].append(monotonic() - queued)
                    begin = monotonic()
                    self.tracking(payload)
                    self.metrics['tracking_seconds'].append(monotonic() - begin)
                    self.metrics['images_processed'] += 1
                if done:
                    return
        finally:
            with self._condition:
                self._tracking_done = True
                self._condition.notify_all()

    def _loop(self):
        last = (-1, -1)
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._failure is not None or self._tracking_done or
                        self.shared_map.work_token() != last)
                    if self._failure is not None:
                        return
                    snapshot = self.shared_map.snapshot_for_optimization()
                    token = (snapshot.epoch, len(snapshot.observations))
                    if token == last and self._tracking_done:
                        return
                begin = monotonic()
                candidates = self.loop_search.detect(snapshot)
                self.metrics['loop_seconds'].append(monotonic() - begin)
                with self._condition:
                    self._loops.extend(candidates)
                    last = token
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._loop_done = True
                self._condition.notify_all()

    def _map(self):
        last = (-1, -1)
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._failure is not None or self._loops or
                    (self._tracking_done and self._loop_done) or
                    self.shared_map.work_token() != last)
                if self._failure is not None:
                    return
                snapshot = self.shared_map.snapshot_for_optimization()
                token = (snapshot.epoch, len(snapshot.observations))
                loops, self._loops = list(self._loops), deque()
                if token == last and not loops and self._tracking_done and self._loop_done:
                    return
            begin = monotonic()
            changed = self.local_mapping.process(snapshot, loops, self.shared_map)
            self.metrics['ba_seconds'].append(monotonic() - begin)
            last = token
            if changed and self.on_solution is not None:
                self.on_solution(self.local_mapping)

    def stop(self):
        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()
        self.raise_if_failed()
