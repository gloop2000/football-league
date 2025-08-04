from agents.dqn import DQNAgent
from agents.minimax_q import MinimaxQAgent
from agents.nash_q import NashQAgent
from agents.phc import PHCAgent
from agents.ppo import PPOAgent
from agents.random_agent import RandomAgent
from league.league_manager import LeagueManager
from train.train_agents import train_agents
from utils.constants import NUM_ACTIONS, ACTIONS, DQN, MINIMAX_Q, NASH_Q, PHC, PPO, RANDOM


def train_agents():
    # Initialize agents
    agent_Random = RandomAgent(RANDOM)
    agent_DQN = DQNAgent(DQN)
    agent_Minimax = MinimaxQAgent(MINIMAX_Q)
    agent_NashQ = NashQAgent(NASH_Q)
    agent_PHCAgent = PHCAgent(PHC)
    agent_PPO = PPOAgent(PPO)
    
    # Set opponent for each agent
    agent_DQN.set_opponent(RANDOM)
    agent_Minimax.set_opponent(RANDOM)
    agent_NashQ.set_opponent(RANDOM)
    agent_PHCAgent.set_opponent(RANDOM)
    agent_PPO.set_opponent(RANDOM)

    # Train agents against RandomAgent
    results_DQN = train_agents(agent_DQN, agent_Random, episodes=10000, save_dir="models/DQN_vs_Random", verbose=True, log_to_csv=True, plot_results=True)
    results_Minimax = train_agents(agent_Minimax, agent_Random, episodes=10000, save_dir="models/Minimax_vs_Random", verbose=True, log_to_csv=True, plot_results=True)
    results_NashQ = train_agents(agent_NashQ, agent_Random, episodes=10000, save_dir="models/NashQ_vs_Random", verbose=True, log_to_csv=True, plot_results=True)
    results_PHCAgent = train_agents(agent_PHCAgent, agent_Random, episodes=10000, save_dir="models/PHC_vs_Random", verbose=True, log_to_csv=True, plot_results=True)
    results_PPO = train_agents(agent_PPO, agent_Random, episodes=10000, save_dir="models/PPO_vs_Random", verbose=True, log_to_csv=True, plot_results=True)

    print("Final win stats for DQN vs Random:", results_DQN)
    print("Final win stats for Minimax Q vs Random:", results_Minimax)
    print("Final win stats for NashQ vs Random:", results_NashQ)
    print("Final win stats for PHC vs Random:", results_PHCAgent)
    print("Final win stats for PPO vs Random:", results_PPO)

def play_league():
    # Initialize agents
    agent_DQN = DQNAgent(DQN)
    agent_Minimax = MinimaxQAgent(MINIMAX_Q)
    agent_NashQ = NashQAgent(NASH_Q)
    agent_PHCAgent = PHCAgent(PHC)
    agent_PPO = PPOAgent(PPO)

    # Load agents
    agent_DQN.load("models/DQN_vs_Random/DQN_10000_model.pkl")
    agent_Minimax.load("models/MinimaxQ_vs_Random/MinimaxQ_10000_model.pkl")
    agent_NashQ.load("models/NashQ_vs_Random/NashQ_10000_model.pkl")
    agent_PHCAgent.load("models/PHC_vs_Random/PHC_10000_model.pkl")
    agent_PPO.load("models/PPO_vs_Random/PPO_10000_model.pkl")

    agents = [
        agent_DQN,
        agent_Minimax,
        agent_NashQ,
        agent_PHCAgent,
        agent_PPO
    ]

    league = LeagueManager(agents, episodes_per_match=100, train_during_league=False)
    league.run_round_robin()
    league.print_leaderboard()
    league.export_leaderboard("league_results.csv")
    league.export_match_results("match_results.csv")

if __name__ == "__main__":
    # Uncomment to train agents
    # train_agents()

    # Uncomment to play league
    play_league()