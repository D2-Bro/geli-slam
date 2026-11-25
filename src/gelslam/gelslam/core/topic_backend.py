"""Backend process service with the two worker roles of the original GelSLAM.

K thread (keyframe insertion): ingest() stores one received keyframe and
extracts its loop-search features. L thread (loop closure): optimize() runs
loop detection and the pose graph on every keyframe ingested so far.

Ownership: the lock protects the stored keyframes, the loop-search feature
database and the session switch. The pose graph, the Local BA keyframe database
and the loop-search cursor are used only by the thread that calls optimize().
process() runs both steps in the caller's thread (offline replay, tests).
"""
from threading import Lock

from gelslam.core.local_mapping import LocalMapping, LoopSearch
from gelslam.core.shared_map import SharedMap


class TopicBackend:
    def __init__(self, config, logger, enabled=True):
        self.config, self.logger, self.enabled = config, logger, enabled
        self.session_id = None
        self.store = SharedMap()
        self.mapper = LocalMapping(config, logger, enabled)
        self.search = LoopSearch(config, logger) if enabled else None
        self._lock = Lock()
        # A new tracking session replaces the store and the feature database in
        # ingest(); optimize() then replaces the mapper it owns.
        self._generation = self._mapper_generation = 0
        self._optimized_count = 0
        # Store and session of the last optimize() call (feedback publication).
        self.optimized_store, self.optimized_session_id = self.store, None

    def ingest(self, packet):
        """K thread: store one keyframe and extract its loop-search features."""
        with self._lock:
            if packet.session_id != self.session_id:
                if self.session_id is not None and packet.kid != 0:
                    self.logger.warning('Ignoring noninitial keyframe from another tracking session')
                    return False
                self.store = SharedMap()
                self.search = LoopSearch(self.config, self.logger) if self.enabled else None
                self.session_id = packet.session_id
                self._generation += 1
            self.store.append_keyframe(packet)
            if self.search is not None:
                self.search.insert(packet)
            return True

    def optimize(self):
        """L thread: loop detection and pose graph update for the ingested keyframes.

        Returns False when no keyframe arrived since the last call.
        """
        with self._lock:
            store, search, generation = self.store, self.search, self._generation
            session_id = self.session_id
            snapshot = store.snapshot_for_optimization()
        if generation != self._mapper_generation:
            self.mapper = LocalMapping(self.config, self.logger, self.enabled)
            self._mapper_generation, self._optimized_count = generation, 0
        if len(snapshot.observations) == self._optimized_count:
            return False
        candidates = search.detect(snapshot) if search is not None else ()
        changed = self.mapper.process(snapshot, candidates, store)
        self.optimized_store, self.optimized_session_id = store, session_id
        self._optimized_count = len(snapshot.observations)
        return changed

    def process(self, packets):
        """Ingest the packets, then optimize once, in the caller's thread."""
        appended = False
        for packet in packets:
            appended = self.ingest(packet) or appended
        if not appended:
            return False
        return self.optimize()
