class BaseAgent:
    def __init__(self, name):
        self.name = name

    def act(self, observation):
        """
        Return an action given the current observation.
        """
        raise NotImplementedError
    
    def setOpponent(self, opponent_name):
        """
        Set the name of the opponent agent.
        """
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done):
        """
        Optional: update internal state for learning agents.
        """
        pass

    def save(self, path):
        """
        Optional: Save model parameters to disk.
        """
        pass

    def load(self, path):
        """
        Optional: Load model parameters from disk.
        """
        pass
