from dataclasses import dataclass

from lhotse import CutSet

import numpy as np


@dataclass  # (hash=True)
class Activity:
    garbage_class: bool = False
    cuts: "CutSet" = None

    def __post_init__(self):
        self.activity = {}
        self.speaker_to_idx_map = {}
        for cut in self.cuts:
            self.speaker_to_idx_map[cut.recording_id] = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in cut.supervisions))
                )
            }

    def get_activity(self, session_id, start_time, duration):
        cut = self.cuts[session_id].truncate(
            offset=start_time, duration=duration
        )
        activity_mask = cut.speakers_audio_mask(
            speaker_to_idx_map=self.speaker_to_idx_map[session_id]
        )
        if self.garbage_class is False:
            activity_mask = np.r_[activity_mask, [np.zeros_like(activity_mask[0])]]
        elif self.garbage_class is True:
            activity_mask = np.r_[activity_mask, [np.ones_like(activity_mask[0])]]
        return activity_mask, self.speaker_to_idx_map[session_id]
