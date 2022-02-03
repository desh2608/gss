from dataclasses import dataclass
from cached_property import cached_property

from paderbox.array import interval as array_interval


@dataclass  # (hash=True)
class Activity:
    garbage_class: bool = False
    rttm: str = None

    @cached_property
    def _data(self):
        data = array_interval.from_rttm(self.rttm)
        return data

    def __getitem__(self, session_id):
        # todo: garbage class
        data = self._data

        data = data[session_id]

        if self.garbage_class is False:
            data["Noise"] = array_interval.zeros()
        elif self.garbage_class is True:
            data["Noise"] = array_interval.ones()
        elif self.garbage_class is None:
            pass
        else:
            raise ValueError(self.garbage_class)

        return data
