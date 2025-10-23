from agents.dqn import DQNAgent
from agents.minimax_q import MinimaxQAgent
from agents.nash_q import NashQAgent
from agents.phc import PHCAgent
from agents.ppo import PPOAgent
from agents.random_agent import RandomAgent
from league.league_manager import LeagueManager
from train.train_agents import train_agents
from utils.constants import DQN, MINIMAX_Q, NASH_Q, PHC, PPO, RANDOM
from utils.plot import plot_win_rates_from_csv as plot_win_rates
from datetime import datetime

def train_agents_for_league():
    # Initialize agents
    agent_Random = RandomAgent(RANDOM)
    agent_DQN = DQNAgent(DQN)
    agent_Minimax = MinimaxQAgent(MINIMAX_Q)
    agent_PHCAgent = PHCAgent(PHC)
    agent_NashQ = NashQAgent(NASH_Q)
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
    results_NashQ = train_agents(agent_NashQ, agent_Random, episodes=150000, save_dir="models/NashQ_vs_Random", verbose=True, log_to_csv=True, plot_results=True)
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
    agent_NashQ.load("models/NashQ_vs_Random/NashQ_150000_model.pkl")
    agent_PHCAgent.load("models/PHC_vs_PHC/PHC_10000_model.pkl")
    agent_PPO.load("models/PPO_vs_PPO/PPO_2_10000_model.pkl")

    agents = [
        agent_DQN,
        agent_Minimax,
        agent_NashQ,
        agent_PHCAgent,
        agent_PPO
    ]

    league = LeagueManager(agents, episodes_per_match=1, train_during_league=False)
    league.run_round_robin()
    league.print_leaderboard()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    league.export_leaderboard("league_results_" + timestamp_str + ".csv")

if __name__ == "__main__":
    # Train agents
    train_agents_for_league()

    # Play league
    play_league()

    # plot graphs for previously trained agents
    plot_win_rates("./models/NashQ_vs_Random/training_log.csv", "NashQ", "Random", "./graphs/NashQ_vs_Random", smooth=True)
    plot_win_rates("./models/DQN_vs_Random/training_log.csv", "DQN", "Random", "./graphs/DQN_vs_Random", smooth=True)
    plot_win_rates("./models/DQN_vs_DQN/training_log.csv", "DQN", "DQN_2", "./graphs/DQN_vs_DQN", smooth=True)
    plot_win_rates("./models/MinimaxQ_vs_Random/training_log.csv", "MinimaxQ", "Random", "./graphs/MinimaxQ_vs_Random", smooth=True)
    plot_win_rates("./models/Minimax_vs_Minimax/training_log.csv", "MinimaxQ", "MinimaxQ_2", "./graphs/MinimaxQ_vs_MinimaxQ", smooth=True)
    plot_win_rates("./models/PHC_vs_Random/training_log.csv", "PHC", "Random", "./graphs/PHC_vs_Random", smooth=True)
    plot_win_rates("./models/PHC_vs_PHC/training_log.csv", "PHC", "PHC_2", "./graphs/PHC_vs_PHC", smooth=True)
    plot_win_rates("./models/PPO_vs_PPO/training_log.csv", "PPO", "PPO_2", "./graphs/PPO_vs_PPO", smooth=True)
    plot_win_rates("./models/PPO_vs_Random/training_log.csv", "PPO", "Random", "./graphs/PPO_vs_Random", smooth=True)