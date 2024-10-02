import numpy as np

def build_transition_matrix(sequence):
  """Builds a transition matrix from a given sequence."""
  states = set(sequence)
  num_states = len(states)
  transition_matrix = np.zeros((num_states, num_states))

  for i in range(len(sequence) - 1):
    current_state_index = list(states).index(sequence[i])
    next_state_index = list(states).index(sequence[i + 1])
    transition_matrix[current_state_index][next_state_index] += 1

  # Normalize the matrix
  transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

  return transition_matrix

def simulate_next_event(transition_matrix, current_state):
  """Simulates the next event based on the transition matrix and current state."""
  probabilities = transition_matrix[current_state]
  next_state = np.random.choice(len(probabilities), p=probabilities)
  return next_state

# Get user input
sequence = input("Enter a sequence of events (separated by spaces): ").split()

#States: Up, down, unchanged
#Sequence: Up down unchanged up up down


# Build the transition matrix
transition_matrix = build_transition_matrix(sequence)

# Simulate future events
current_state = np.random.randint(len(transition_matrix))
for i in range(10):
  next_state = simulate_next_event(transition_matrix, current_state)
  print(f"Predicted next event: {list(transition_matrix.index)[next_state]}")