from methods_types import DashboardMethod


class DumbDashboardMethod(DashboardMethod):
    def __init__(self):
        pass

    def process(self, base, gt, fn):
        return {}
