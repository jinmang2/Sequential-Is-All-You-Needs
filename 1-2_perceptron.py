import numpy as np

class SingleGate:

    def __init__(self, gatename='AND'):
        if gatename.upper() == 'AND':
            W, b = (0.5, 0.5), -0.7
        elif gatename.upper() == 'NAND':
            W, b = (-0.5, -0.5), 0.7
        elif gatename.upper() == 'OR':
            W, b = (-0.2, -0.2), 0.1
        else:
            raise NotImplementedError(
            "Only have to input 'AND', 'NAND', 'OR' as gatename keyword argument.")
        self.W = np.array(W)
        self.b = np.array(b)

    def _getSignal(self, X):
        return np.sum(X * self.W) + self.b

    def _step(self, X):
        signal = self._getSignal(X)
        return 1 if signal > 0 else 0

    def run(self, X=None, mode='input'):
        Err_msg = "Change the mode to a 'truth_table' or type X input."
        Err_msg2 = "'mode' argument should only use one of 'input' and 'truth_table'."
        assert ~((mode == 'input') and (X is None)), Err_msg
        if mode is not 'input':
            if mode.lower() == 'truth_table':
                return {
                    (0, 0): self._step([0, 0]),
                    (0, 1): self._step([0, 1]),
                    (1, 0): self._step([1, 0]),
                    (1, 1): self._step([1, 1]),
                }
            else:
                raise NameError(Err_msg2)
        else:
            X = np.array(X)
            return self._step(X)

class XORGate:

    def __init__(self):
        self.NAND = SingleGate('NAND')
        self.OR   = SingleGate('OR')
        self.AND  = SingleGate('AND')

    def _step(self, X):
        return self.AND.run(
                [self.NAND.run(X), self.OR.run(X)]
            )

    def run(self, X=None, mode='input'):
        Err_msg = "Change the mode to a 'truth_table' or type X input."
        Err_msg2 = "'mode' argument should only use one of 'input' and 'truth_table'."
        assert ~((mode == 'input') and (X is None)), Err_msg
        if mode is not 'input':
            if mode.lower() == 'truth_table':
                return {
                    (0, 0): self._step([0, 0]),
                    (0, 1): self._step([0, 1]),
                    (1, 0): self._step([1, 0]),
                    (1, 1): self._step([1, 1]),
                }
            else:
                raise NameError(Err_msg2)
        else:
            X = np.array(X)
            return self._step(X)


def main():
    NAND = SingleGate('NAND')
    OR = SingleGate('OR')
    AND = SingleGate('AND')
    XOR = XORGate()

    print("NAND:", NAND.run(mode='truth_table'))
    print(" AND:", AND.run(mode='truth_table'))
    print("  OR:", OR.run(mode='truth_table'))
    print(" XOR:", XOR.run(mode='truth_table'))

if __name__ == '__main__':
    main()
    # NAND: {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    #  AND: {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    #   OR: {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    #  XOR: {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    # [Finished in 0.189s]
