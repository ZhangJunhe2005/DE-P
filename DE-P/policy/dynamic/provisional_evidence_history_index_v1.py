"""Bounded read-only index for evidence history; reads never refresh time."""

from __future__ import annotations

from collections import OrderedDict


CONTRACT_VERSION = "provisional_evidence_history_index_v1"


class ProvisionalEvidenceHistoryIndexV1:
    def __init__(self, maximum_entries=256):
        maximum_entries = int(maximum_entries)
        if maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._entries = OrderedDict()
        self.eviction_count = 0

    def put(self, identity, generation, timestamp, value):
        key = (identity, generation)
        self._entries[key] = (float(timestamp), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maximum_entries:
            self._entries.popitem(last=False)
            self.eviction_count += 1

    def get(self, identity, generation):
        row = self._entries.get((identity, generation))
        return None if row is None else row[1]

    def timestamp(self, identity, generation):
        row = self._entries.get((identity, generation))
        return None if row is None else row[0]

    def __len__(self):
        return len(self._entries)


__all__ = ["CONTRACT_VERSION", "ProvisionalEvidenceHistoryIndexV1"]
