from cached_property import cached_property

from lhotse import Recording

from gss.utils.data_utils import backup_orig_start_end, add_context


class RTTMDataset:
    def __init__(self, recording, rttm_path):
        """
        A database that is generated from a rttm file and a Lhotse Recording.

        Args:
            recording: Lhotse Recording type
            rttm_path:
                str, path or rttm file for the recording.

        """
        super().__init__()
        self._recording = recording
        self._rttm_path = rttm_path
        self._sample_rate = recording.sampling_rate
        assert isinstance(recording, Recording)

    @cached_property
    def _rttm(self):
        from paderbox.array import interval as array_interval

        # NOTE: The following takes care of the case if the RTTM file has more than
        # one session, although ideally this shouldn't happen.
        rttm = array_interval.from_rttm(self._rttm_path)[self._recording.id]

        return rttm

    def _example_id(self, speaker_id, start, end):
        start_time = start / self._sample_rate
        end_time = end / self._sample_rate
        return f"{self._recording.id}-{speaker_id}-{int(start_time*100):06d}-{int(end_time*100):06d}"

    def _load_audio(self, example):
        min_num_samples = example.get("end_orig", example["end"]) - example["start"]
        offset = example["start"] / self._sample_rate
        duration = (
            max(min_num_samples, example["end"] - example["start"]) / self._sample_rate
        )
        # Make sure duration does not exceed the recording length
        duration = min(duration, self._recording.duration)
        example["audio_data"] = self._recording.load_audio(
            offset=offset, duration=duration
        )
        return example

    @property
    def data(self):
        for speaker_id, speaker in self._rttm.items():
            for interval in speaker.intervals:
                start, end = interval
                example_id = self._example_id(speaker_id, start, end)
                data_item = {
                    "example_id": example_id,
                    "start": start,
                    "end": end,
                    "num_samples": end - start,
                    "session_id": self._recording.id,
                    "speaker_id": speaker_id,
                    "recording": self._recording,
                }
                yield data_item

    def get_examples(
        self,
        audio_read=False,
        context_samples=0,
        equal_start_context=False,
    ):
        for ex in self.data:
            if context_samples > 0:
                ex = backup_orig_start_end(ex)
                ex = add_context(
                    ex, context_samples, equal_start_context=equal_start_context
                )

            if audio_read:
                ex = self._load_audio(ex)

            yield ex
