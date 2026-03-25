class ExcitonRangeError:
    def __init__(self, message="n_exc out of range from what VASP calculated"):
        self.message = message
        super().__init__(self.message)


class FBZKPTsMismatchError:
    def __init__(self, message="The FBZKPTs from BSE dataset is different from the GW dataset"):
        self.message = message
        super().__init__(self.message)