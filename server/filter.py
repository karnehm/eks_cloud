import numpy as np


class ExtendedKalmanFilter(object):

    def __init__(self, x, A, B, C, D, P, Q, R):

        self._x = x
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._P = P
        self._Q = Q
        self._R = R
        self._y = None
        self._K = None

    def update(self, z):

        P = self._P
        R = self._R
        x = self._x

        S = self._C * P * self._C.T + R # Step 2a
        K = P * self._C.T * S.I # Step 2a
        self._K = K

        y = np.subtract(z, self._y) # Step 2b
        self._x = x + K * y # Step 2b

        self._P = self._P - K * (self._C * self._P * self._C.T + R)*K.T



    def predict(self, u=0, u_before=0): # u = u_{k-1}
        self._x = self._A * self._x + self._B * u_before # Step 1a
        self._P = self._A * self._P * self._A.T + self._Q # Step 1b
        self._y = self._C * self._x + self._D * u  # Step 1c
        return self._y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
