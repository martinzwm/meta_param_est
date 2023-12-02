import pdb
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# seed
seed_value = 42
import random
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SpringMassSystem(nn.Module):
    def __init__(self, W):
        super(SpringMassSystem, self).__init__()
        
        self.k1, self.k2, self.m1, self.m2 = W
    
    def dynamics(self, t, state):
        """
        Define the system dynamics using Newton's second law.
        state = [x1, x2, x1_dot, x2_dot]
        """
        x1, x2, x1_dot, x2_dot = state
        
        # Dynamics for mass m1
        dx1_dt = x1_dot
        dx1_dot_dt = (self.k2*(x2-x1) - self.k1*x1) / self.m1
        
        # Dynamics for mass m2
        dx2_dt = x2_dot
        dx2_dot_dt = (-self.k2*(x2-x1)) / self.m2
        
        return torch.tensor([dx1_dt, dx2_dt, dx1_dot_dt, dx2_dot_dt])
    
    def trajectory(self, initial_state, t_span, dt):
        """
        Generate the trajectory of the system over a specified time span using Euler's method.
        """
        times = torch.arange(t_span[0], t_span[1], dt)
        states = [initial_state]
        
        for t in times[:-1]:
            derivatives = self.dynamics(t, states[-1])
            new_state = states[-1] + derivatives * dt
            states.append(new_state)
        
        return times, torch.stack(states)
    
    def trajectories(self, initial_states, t_span, dt):
        """
        Generate multiple trajectories, one for each initial condition.
        Uses trajectory() internally.
        """
        times, trajectories = [], []

        for initial_state in initial_states:
            times_i, trajectory_i = self.trajectory(initial_state, t_span, dt)
            times.append(times_i)
            trajectories.append(trajectory_i)
        
        times, trajectories = torch.stack(times), torch.stack(trajectories)
        return times, trajectories
    

def generate_data(num_parameter_sets=6, num_trajectories_per_set=1, t_span=[0, 10], dt=0.1, save_path=None):
    """
    Generate random trajectories based on random parameters and initial states.

    :param num_parameter_sets: Number of random spring-mass parameter sets to generate.
    :param num_trajectories_per_set: Number of random initial states (trajectories) to generate per parameter set.
    :param t_span: Time span for which the trajectory is generated.
    :param dt: Time step for Euler's method.
    :return: List of (W, times, trajectories) tuples where:
             - W is a tensor of the spring-mass parameters.
             - times is a tensor of the time points.
             - trajectories is a tensor of the shape (num_trajectories, len(times), 4).
    """
    
    data = []

    # Set m to be linspace bewteen 1 and 2 inclusive
    ms_linspace = torch.linspace(1, 2, num_parameter_sets)

    for m1 in ms_linspace:
        for m2 in ms_linspace:
            # Define system
            W = torch.tensor([1.0, 1.0, m1, m2])
            system = SpringMassSystem(W)

            # Randomly generate initial states (x1, x2 between -1.0 to 1.0 and velocities as 0)
            initial_states = torch.cat([
                (2 * torch.rand(num_trajectories_per_set, 2) - 1.0),
                torch.zeros(num_trajectories_per_set, 2)
            ], dim=1)
            # initial_states = torch.cat([
            #     (0.1 * torch.ones(num_trajectories_per_set, 2)),
            #     torch.zeros(num_trajectories_per_set, 2)
            # ], dim=1)

            times, trajectories = system.trajectories(initial_states, t_span, dt)

            data.append((W, times, trajectories))

    if save_path:
        with open(save_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data


class TrajectoryDataset(Dataset):
    """Load and return a dataset of saved trajectories."""

    def __init__(self, data_path=None):
        if data_path is None:
            raise ValueError("data_path cannot be None")
        else:
            with open(data_path, 'rb') as handle:
                self.data = pickle.load(handle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        W, times, trajectories = self.data[idx]
        return W, times, trajectories
    

def get_dataloader(batch_size=32, data_path=None, num_workers=0, shuffle=True):
    dataset = TrajectoryDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


class TestDynamics:

    def __init__(self):
        pass

    def test_dynamics(self, plot_figure=False):
        """Example usage"""
        # Define dynamic system
        W = torch.tensor([1.0, 2.0, 1.0, 2.0])  # k1, k2, m1, m2
        system = SpringMassSystem(W)

        # Define initial state
        initial_state = torch.tensor([0.1, 0.2, 0.0, 0.0])  # Initial displacements and velocities
        t_span = [0, 10]  # From t=0 to t=10
        dt = 0.1  # Time step

        # Generate trajectories
        times, trajectory = system.trajectory(initial_state, t_span, dt)

        if plot_figure:
            # Plot the trajectories
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            # Displacement subplot
            axs[0].plot(times, trajectory[:, 0], label="x1 (m1 displacement)")
            axs[0].plot(times, trajectory[:, 1], label="x2 (m2 displacement)")
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("Displacement")
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_title("k1 = {}, k2 = {}, m1 = {}, m2 = {}".format(*W.tolist()))

            # Velocity subplot
            axs[1].plot(times, trajectory[:, 2], label="x1_dot (m1 velocity)")
            axs[1].plot(times, trajectory[:, 3], label="x2_dot (m2 velocity)")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Velocity")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.savefig('./test_dynamics.png')

        return times, trajectory
    
    def generate_gif(self, variable="m1"):
        """Generate a .gif of trajectories to visualize the effect of m1 and m2 on the system"""
        images = []
        for v in np.linspace(1, 2, 100):
            # Define dynamic system
            if variable == "m1":
                m1 = v
                m2 = 1.0
            elif variable == "m2":
                m1 = 1.0
                m2 = v
            W = torch.tensor([1.0, 1.0, m1, m2]) # k1, k2, m1, m2
            system = SpringMassSystem(W)

            # Define initial state
            initial_state = torch.tensor([0.1, 0.2, 0.0, 0.0])  # Initial displacements and velocities
            t_span = [0, 10]  # From t=0 to t=10
            dt = 0.1  # Time step

            # Generate trajectories
            times, trajectory = system.trajectory(initial_state, t_span, dt)

            # Save figure
            fig, ax = plt.subplots()
            ax.plot(times, trajectory[:, 0], label="x1 (m1 displacement)")
            ax.plot(times, trajectory[:, 1], label="x2 (m2 displacement)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Displacement")
            ax.legend()
            ax.grid(True)
            ax.set_title("k1 = {}, k2 = {}, m1 = {:.2f}, m2 = {:.2f}".format(*W.tolist()))
            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close()
        
        # Create a .gif of the trajectories
        import imageio
        imageio.mimsave(f'./trajectories_{variable}.gif', images, duration=100)


    def test_trajectories(self):
        # Define dynamic system
        W = torch.tensor([1.0, 2.0, 1.0, 2.0])
        system = SpringMassSystem(W)

        # Define initial state
        t_span = [0, 10]  # From t=0 to t=10
        dt = 0.01  # Time step
        initial_states = torch.rand(10, 4) * 10 # Generate random inital states

        # Generate trajectories
        times, trajectories = system.trajectories(initial_states, t_span, dt)
        print(times.shape)
        print(trajectories.shape)

    def test_dataset(self):
        dataset = TrajectoryDataset(data_path="train_data.pickle")
        print(len(dataset))
        print(dataset[0])

    def test_data_generation(self):
        data = generate_data(num_parameter_sets=5, num_trajectories_per_set=100, save_path="train_data.pickle")
        data = generate_data(num_parameter_sets=5, num_trajectories_per_set=2, save_path="val_data.pickle")
        

if __name__ == "__main__":
    test = TestDynamics()
    # test.test_dynamics(plot_figure=True)
    # test.generate_gif(variable="m2")
    test.test_data_generation()