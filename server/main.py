import pandas as pd
from flask import Flask, request, abort
import numpy as np

from battery import Battery
from filter import ExtendedKalmanFilter
from utils import Polynomial
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

ekf = {}

# https://github.com/Murtaza-097/Soc_Estimation/blob/main/BatteryModel.mat
battery_parameters = pd.read_csv('battery_parameters.csv')
bp_index = -1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_EKF(soc: float, battery: Battery, std_dev, time_step):
    # initial state (SoC is intentionally set to a wrong value)
    # x = [[SoC], [RC1 voltage]]
    x = np.matrix([[soc],
                   [0.0],
                   [0.0]])

    exp_coeff1 = np.exp(-time_step / (battery.C1 * battery.R1))
    exp_coef2 = np.exp(-time_step / (battery.C2 * battery.R2))

    # state transition model
    A = np.matrix([[1, 0, 0],
                   [0, exp_coeff1, 0],
                   [0, 0, exp_coef2]])

    # control-input model
    B = np.matrix([[-time_step / battery.total_capacity],
                   [battery.R1 * (1 - exp_coeff1)],
                   [battery.R2 * (1 - exp_coef2)]])

    C = np.matrix([0, -battery.R1, -battery.R2])

    D = -battery.R0

    # variance from std_dev
    var = std_dev ** 2

    # measurement noise
    R = var

    # state covariance
    P = np.matrix([[var, 0, 0],
                   [0, var, 0],
                   [0, 0, var]])

    # process noise covariance matrix
    Q = np.matrix([[var / 50, 0, 0],
                   [0, var / 50, 0],
                   [0, 0, var / 50]])

    return ExtendedKalmanFilter(x, A, B, C, D, P, Q, R)


'''
input of request: 
- experiment_id: string
- timestamp: number|string
- soc_<location>: number [1;0]
'''
@app.route('/', methods=["PUT"])
def update():
    data = request.json

    if 'experiment_id' not in data:
        abort(400, 'experiment_id is required')

    if 'timestamp' not in data:
        abort(400, 'timestamp is required')

    if len([e for e in data.keys() if e.startswith('soc_')]) < 1:
        abort(400, 'There is no soc included')

    ekf[data['experiment_id']] = {}


    # Capacity of the cell
    Q = 5 # Ah

    # Thevenin model values
    R0 = 0.001734
    R1 = 0.00051514
    C1 = 19576
    R2 = 0.00051545
    C2 = 1.9916e+05

    battery_parameters_temperature = find_nearest(battery_parameters.Temp, data['cel_c1'])
    battery_parameters_soc = find_nearest(battery_parameters[battery_parameters['Temp'] == battery_parameters_temperature]['SOC'], data['soc_c1'])
    bp = battery_parameters[(battery_parameters['Temp'] == battery_parameters_temperature) & (battery_parameters['SOC'] == battery_parameters_soc)]
    ekf[data['experiment_id']]['bp_index'] = bp.index[0]

    battery = Battery(Q, bp['R0'].values[0], bp['R1'].values[0], bp['R2'].values[0],
                      bp['C1'].values[0], bp['C2'].values[0])

    time_step = 0.1
    std_dev = 0.015
    ekf[data['experiment_id']]['kf'] = get_EKF(soc=data['soc_c1'], battery=battery, time_step=time_step, std_dev=std_dev)
    ekf[data['experiment_id']]['amp_c1_t-1'] = 0

    return {'status': 'OK'}, 200


@app.route('/', methods=["DELETE"])
def delete():
    data = request.json
    ekf[data['experiment_id']] = {}


@app.route('/', methods=["POST"])
def kalman():
    data = request.json

    if data['experiment_id'] not in ekf:
        ekf[data['experiment_id']] = ExtendedKalmanFilter()

    if 'amp_c1_t-1' not in ekf[data['experiment_id']]:
        ekf[data['experiment_id']]['amp_c1_t-1'] = 0

    current_soc = ekf[data['experiment_id']]['kf'].x[0, 0]
    # Check if Battery Parameters have to be updated
    battery_parameters_temperature = find_nearest(battery_parameters.Temp, data['cel_c1'])
    battery_parameters_soc = find_nearest(
        battery_parameters[battery_parameters['Temp'] == battery_parameters_temperature]['SOC'], current_soc)
    bp = battery_parameters[(battery_parameters['Temp'] == battery_parameters_temperature) & (
                battery_parameters['SOC'] == battery_parameters_soc)]
    if not ekf[data['experiment_id']]['bp_index'] == bp.index[0]:
        Q = 5  # Ah
        battery = Battery(Q, bp['R0'].values[0], bp['R1'].values[0], bp['R2'].values[0],
                          bp['C1'].values[0], bp['C2'].values[0])

        time_step = 0.1
        std_dev = 0.015
        x = ekf[data['experiment_id']]['kf'].x
        ekf[data['experiment_id']]['kf'] = get_EKF(soc=x[0, 0], battery=battery, time_step=time_step,
                                                   std_dev=std_dev)
        ekf[data['experiment_id']]['kf'].x = x

    ekf[data['experiment_id']]['kf'].predict(u=data['amp_c1'], u_before=ekf[data['experiment_id']]['amp_c1_t-1'])
    ekf[data['experiment_id']]['kf'].update(data['vol_c1'])
    ekf[data['experiment_id']]['amp_c1_t-1'] = data['amp_c1']

    return {'soc_c1': ekf[data['experiment_id']]['kf'].x[0, 0]}, 200


if __name__ == '__main__':
    app.run(debug=True)
