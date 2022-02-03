from dataclasses import dataclass
from cached_property import cached_property


@dataclass  # (hash=True)
class Activity:
    garbage_class: bool = False
    rttm: str = None

    @cached_property
    def _data(self):
        from paderbox.array import intervall as array_intervall

        data = array_intervall.from_rttm(self.rttm)
        return data

    def __getitem__(self, session_id):
        from paderbox.array import intervall as array_intervall

        # todo: garbage class
        data = self._data

        data = data[session_id]

        if self.garbage_class is False:
            data["Noise"] = array_intervall.zeros()
        elif self.garbage_class is True:
            data["Noise"] = array_intervall.ones()
        elif self.garbage_class is None:
            pass
        else:
            raise ValueError(self.garbage_class)

        return data
