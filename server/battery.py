
class Battery:
    # capacity in Ah
    def __init__(self, total_capacity, R0, R1, R2, C1, C2):
        # capacity in As
        self.total_capacity = total_capacity * 3600
        self.actual_capacity = self.total_capacity

        # Thevenin model : OCV + R0 + R1//C1
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.C1 = C1
        self.C2 = C2

        self._current = 0
        self._RC_voltage = 0

