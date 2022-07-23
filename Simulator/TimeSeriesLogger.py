import numpy as np


class EventType:
    switch_bw = "switch_bw"
    controller_bw = "controller_bw"


class TimeSeriesLogger(object):
    def __init__(self, time_slice):
        self.event_logger = {EventType.switch_bw: {},
                             EventType.controller_bw: {}}
        self.time_slice = time_slice

    def log_event(self, event_type, event_time, event_value):
        """
        :param event_type:
        :param event_time:
        :param event_value:
        Log event in accumulative way add events to current time slots
        :return:
        """
        time_series_index = int(np.floor(event_time / self.time_slice[event_type]))
        self.event_logger[event_type][time_series_index] = event_value + self.event_logger[event_type].get(
            time_series_index, 0)

    def record_one_event(self, event_type, event_time, event_value):
        """
        :param event_type:
        :param event_time:
        :param event_value:
        :return:
        Record event one time in time slot for example threshold that does not change every epoch
        """
        time_series_index = int(np.floor(event_time / self.time_slice[event_type]))
        if time_series_index in self.event_logger[event_type]:
            return False
        else:
            self.event_logger[event_type][time_series_index] = event_value
            return True

    def sum_all_events(self, event_type):
        return sum(self.event_logger[event_type].values())

    def clear_logs(self):
        for event_list in self.event_logger.values():
            event_list.clear()

    @staticmethod
    def init_time_series_time_slice(time_slice):
        return TimeSeriesLogger({EventType.switch_bw: time_slice,
                                 EventType.controller_bw: time_slice})
