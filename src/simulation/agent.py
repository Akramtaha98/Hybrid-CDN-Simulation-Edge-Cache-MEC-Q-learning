import numpy as np

class QLearningAgent:
    """Tabular Q-learning for routing decisions.

    States: (type, key)
      - type: 'content' or 'compute'
      - key:  for content: 'hit' or 'miss'
              for compute: 'idle' or 'busy'

    Actions:
      - content: 0=Edge, 1=Neighbor, 2=Cloud
      - compute: 0=MEC,  1=Cloud
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.Q_content = {}
        self.Q_compute = {}

    def _table_for(self, state):
        return (self.Q_content, 3) if state[0] == 'content' else (self.Q_compute, 2)

    def choose_action(self, state, valid_actions):
        table, n_actions = self._table_for(state)
        key = state[1]
        if key not in table:
            table[key] = np.zeros(n_actions, dtype=float)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))
        qvals = table[key].copy()
        best = max(valid_actions, key=lambda a: qvals[a])
        return int(best)

    def update(self, state, action, reward, next_state):
        table, n_actions = self._table_for(state)
        key = state[1]
        if key not in table:
            table[key] = np.zeros(n_actions, dtype=float)
        next_table, n_next = self._table_for(next_state)
        next_key = next_state[1]
        if next_key not in next_table:
            next_table[next_key] = np.zeros(n_next, dtype=float)
        best_next = float(np.max(next_table[next_key]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - table[key][action]
        table[key][action] += self.lr * td_error

    def set_epsilon(self, epsilon):
        self.epsilon = float(epsilon)
