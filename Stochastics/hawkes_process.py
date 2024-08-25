import numpy as np
import matplotlib.pyplot as plt

class HawkesProcess:
    def __init__(self,baseline_intensity,decay,excitation_strenght,max_time, seed = None):
        """
        Initalize the Hawkes process class

        Parameters: 
        ----------------------------
            baseline_intensity: float, the baseline intensity of the process. 
            decay: float, the decay rate of the excitation effect
            excitation_strenght: float, the strength of exciation caused by each event
            max_time: float, the maximum time (T) to simulate the process
            seed: int, random seed for reproducibility
        """
        self.baseline_intensity = baseline_intensity
        self.decay = decay
        self.excitation_strenght = excitation_strenght
        self.max_time = max_time
        self.seed = seed
        np.random.seed(self.seed)
        self.event_times = []

    def simulate(self):
        """
        Simulate the Hawkes process over the time interval [0,max_time]
        """

        t = 0
        self.event_times = []

        while t< self.max_time:
            # Calculate the current intensity
            current_intensity = self.baseline_intensity + self.calculate_excitement()

            # Draw the inter-arrival times
            u = np.random.uniform(0,1)
            w = -np.log(u)/current_intensity

            # Move to next time
            t += w

            # Acceptance rejection to determine if an event occurs
            if np.random.uniform(0,1) <= self.intensity(t)/current_intensity:
                self.event_times.append()

    def calculate_excitement(self,current_time):
        """
        Calculate the excitement at a given time based on passed events
        """

        excitement = 0
        for event_time in self.event_times:
            excitement += self.excitation_strenght*np.exp(-self.decay*(current_time-event_time))
        
        return excitement
    
    def intensity(self,current_time):
        """Calculate the intensity of the Hawkes process"""
        return self.baseline_intensity + self.calculate_excitement(current_time)
    
    def plot_events(self):
        """Plots the events of the process"""
        plt.figure(figsize = (10,2))
        plt.eventplot(self.event_times,orientation='horizontal',colors='black')
        plt.xlim(0,self.max_time)
        plt.xlabel('Time')
        plt.title("Hawkes process")
        plt.show()
        