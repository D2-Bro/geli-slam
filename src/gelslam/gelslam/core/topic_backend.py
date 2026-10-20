"""Backend process service. One worker owns all mutable graph/feature state."""
from gelslam.core.local_mapping import LocalMapping, LoopSearch
from gelslam.core.shared_map import SharedMap


class TopicBackend:
    def __init__(self, config, logger, enabled=True):
        self.config, self.logger, self.enabled = config, logger, enabled
        self.session_id = None
        self.store = SharedMap()
        self.mapper = LocalMapping(config, logger, enabled)
        self.search = LoopSearch(config, logger) if enabled else None

    def process(self, packets):
        appended = False
        for packet in packets:
            if packet.session_id != self.session_id:
                if self.session_id is not None and packet.kid != 0:
                    self.logger.warning('Ignoring noninitial keyframe from another tracking session')
                    continue
                self.store = SharedMap()
                self.mapper = LocalMapping(self.config, self.logger, self.enabled)
                self.search = LoopSearch(self.config, self.logger) if self.enabled else None
                self.session_id = packet.session_id
            self.store.append_keyframe(packet)
            appended = True
        if not appended:
            return False
        snapshot = self.store.snapshot_for_optimization()
        candidates = self.search.detect(snapshot) if self.search is not None else ()
        return self.mapper.process(snapshot, candidates, self.store)
