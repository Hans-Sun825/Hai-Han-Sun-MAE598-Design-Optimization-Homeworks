import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import math


n_state = 5
n_action = 2
T = 100
dt = 0.1
x_0 = np.array([-2.,2.,0.,0.,0.])
total_time_step = 100

action_trajectory = []
state_trajectory = []

# environment parameters
FRAME_TIME = 0.1  # time interval
BACKWARD_ACCEL_Y = 0.005  # gravity constant in Y direction
BACKWARD_ACCEL_X = 0.01  # gravity constant in X direction
BOOST_ACCEL = 0.1  # thrust constant
delta = 60 # steering angle
L = 0.5 # wheelbase
OMEGA_RATE = math.tan(delta)/L  # max rotation rate
cos_value = math.cos(delta)
Delta_OMEGA_RATE = 1 / (L * (cos_value) ** 2)  # max rotation rate 
#print(Delta_OMEGA_RATE)

def mpc(x_0, T):

    x = cp.Variable((n_state, T + 1))
    u = cp.Variable((n_action, T))
    theta0 = x_0[4]
    A = np.array([[1, 0, dt, 0, 0],
                [0, 1, 0, dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, BOOST_ACCEL * dt * OMEGA_RATE]])
    B = np.array([[0, 0],
                [0, 0],
                [np.sin(90 + theta0) * dt, 0],
                [np.cos(90 + theta0) * dt, 0],
                [0, BOOST_ACCEL * dt * Delta_OMEGA_RATE]])
    c = np.array([0,0,-BACKWARD_ACCEL_X * dt, -BACKWARD_ACCEL_Y * dt, 0])
    cost = 0
    constr = []
    for t in range(T):
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + c, 
                   cp.abs(u[0, t]) <= 2, 
                   cp.abs(u[1, t]) <= 1,
                   x[1, t] >= 0]
    # cost = cp.sum_squares(x[:, T])
    cost = 10 * cp.square(x[0, T]) + cp.sum_squares(x[:, T])

    # sums problem objectives and concatenates constraints.
    constr += [x[:, 0] == x_0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()
    #problem.solve(verbose=True)
    return x, u

def visualize(x,u):
    data = x
    action_data = u
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]
    theta = data[:, 4]
    thrust = action_data[:,0]
    frame = range(T)

    fig, ax = plt.subplots(1, 4, tight_layout = 1, figsize = (15, 5))

    ax[0].plot(x, y, c = 'b')
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set(title=f'Displacement plot(x-y)')

    ax[1].plot(frame, vx, c = 'c', label = "Velocity in x")
    ax[1].plot(frame, vy, c = 'r', label = "Velocity in y")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend(frameon=0)
    ax[1].set(title =f'Velocity plot')

    ax[2].plot(frame, theta, c = 'g', label = "theta")
    ax[2].set_xlabel("Time interval")
    ax[2].set_ylabel("Theta")
    ax[2].legend(frameon=0)
    ax[2].set(title=f'Theta plot')

    ax[3].plot(frame, thrust, c = 'y', label = "thrust")
    ax[3].set_xlabel("Time interval")
    ax[3].set_ylabel("Thrust")
    ax[3].legend(frameon=0)
    ax[3].set(title=f'Thrust plot')
    plt.show()

def simulate(state, action):
    delta_state_backward = np.array([0., 0., -BACKWARD_ACCEL_X * FRAME_TIME, -BACKWARD_ACCEL_Y * FRAME_TIME, 0.])
    state_tensor = np.zeros(5)
    state_tensor[2] = np.sin(state[4] + 90)
    state_tensor[3] = np.cos(state[4] + 90)
    state_tensor[4] = (state[4]) * OMEGA_RATE

    delta_state = BOOST_ACCEL * FRAME_TIME * state_tensor * action[0]

    # Theta
    #state_tensor_delta = np.array([0., 0., 0., 0., 1.])
    delta_state_theta = BOOST_ACCEL * FRAME_TIME * OMEGA_RATE * action[1] 

    # Update state
    step_mat = np.array([[1., 0.,FRAME_TIME, 0., 0.],
                                [0., 1., 0., FRAME_TIME, 0.],
                                [0., 0., 1., 0., 0.],
                                [0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 1.]])
    
    shift_mat = np.array([[0., 0.,FRAME_TIME, 0., 0.],
                                [0., 0., 0., FRAME_TIME, 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.]])

    state = step_mat @ state + shift_mat @ delta_state * 0.5 + shift_mat @ delta_state_backward * 0.5
    state = state + delta_state + delta_state_backward
    state[4] += delta_state_theta
    print(state)
    print(action)
    return state

def control(x_0, total_time_step):
    x_current = x_0
    for i in range(total_time_step):
        x, u = mpc(x_current, T-i)
        # visualize(np.array(x[1:,:].value).T, np.array(u.value).T)
        action = u[:,0].value
        x_current = simulate(x_current, action)
        action_trajectory.append(action)
        state_trajectory.append(x_current)

control(x_0, total_time_step)

from ipywidgets import IntProgress
from IPython.display import display
from matplotlib import pyplot as plt, rc
from matplotlib.animation import FuncAnimation, PillowWriter
rc('animation', html='jshtml')

def animation(state_trajectory, action_trajectory):
        length = 0.3          # m
        width = 0.3          # m

        v_exhaust = 1     
        print("Generating Animation")
        steps = min(len(state_trajectory), len(action_trajectory))
        final_time_step = round(1/steps,2)
        f = IntProgress(min = 0, max = steps)
        display(f)

        data = np.array(state_trajectory)
        action_data = np.array(action_trajectory)
        x_t = data
        u_t = action_data
        print(x_t.shape, u_t.shape)

        fig = plt.figure(figsize = (5,8), constrained_layout=False)
        ax1 = fig.add_subplot(111)
        plt.axhline(y=0., color='b', linestyle='--', lw=0.8)

        ln1, = ax1.plot([], [], linewidth = 20, color = 'lightblue') # rocket body
        ln6, = ax1.plot([], [], '--', linewidth = 2, color = 'orange') # trajectory line
        ln2, = ax1.plot([], [], linewidth = 4, color = 'tomato') # thrust line

        plt.tight_layout()

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-2, 5)
        ax1.set_aspect(1)  # aspect of the axis scaling, i.e. the ratio of y-unit to x-unit

        def update(i):
            rocket_theta = x_t[i, 4]

            rocket_x = x_t[i, 0]
            # length/1 is just to make rocket bigger in animation
            rocket_x_points = [rocket_x + length/1 * np.sin(rocket_theta), rocket_x - length/1 * np.sin(rocket_theta)]

            rocket_y = x_t[i, 1]
            rocket_y_points = [rocket_y + length/1 * np.cos(rocket_theta), rocket_y - length/1 * np.cos(rocket_theta)]

            ln1.set_data(rocket_x_points, rocket_y_points)

            thrust_mag = u_t[i, 0]
            thrust_angle = -u_t[i, 1]

            flame_length = (thrust_mag) * (0.4/v_exhaust)
            # flame_x_points = [rocket_x_points[1], rocket_x_points[1] + flame_length * np.sin(thrust_angle - rocket_theta)]
            # flame_y_points = [rocket_y_points[1], rocket_y_points[1] - flame_length * np.cos(thrust_angle - rocket_theta)]
            flame_x_points = [rocket_x_points[1], rocket_x_points[1] - flame_length * np.sin(rocket_theta)]
            flame_y_points = [rocket_y_points[1], rocket_y_points[1] - flame_length * np.cos(rocket_theta)]

            ln2.set_data(flame_x_points, flame_y_points)
            ln6.set_data(x_t[:i, 0], x_t[:i, 1])
            f.value += 1

        playback_speed = 5000 # the higher the slower 
        anim = FuncAnimation(fig, update, np.arange(0, steps-1, 1), interval= final_time_step * playback_speed)

        # Save as GIF
        writer = PillowWriter(fps=20)
        anim.save("Automatic_Parking_mpc.gif", writer=writer)

animation(state_trajectory, action_trajectory)
