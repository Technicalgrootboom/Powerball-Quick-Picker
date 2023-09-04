import random

# Initialize Q-learning table
Q = {}

# Hyperparameters for Q-learning
learning_rate = 0.3
discount_factor = 0.8
exploration_prob = 0.2

def get_next_action(state):
    # Epsilon-greedy policy for action selection
    if random.uniform(0, 1) < exploration_prob:
        # Random exploration
        return tuple(random.choices(range(1, 70), k=6) + [random.randint(1, 26)])
    else:
        # Greedy exploitation
        if state not in Q or len(Q[state]["actions"]) == 0:
            # If state is not explored or no valid actions, make a random move
            return tuple(random.choices(range(1, 70), k=6) + [random.randint(1, 26)])
        else:
            # Choose the action with the highest Q-value
            return Q[state]["actions"]

def update_q_value(prev_state, action, next_state, reward):
    if prev_state not in Q:
        Q[prev_state] = {"actions": action, "value": 0}
    if next_state not in Q:
        Q[next_state] = {"actions": [], "value": 0}

    prev_value = Q[prev_state]["value"]
    next_value = Q[next_state]["value"]
    best_next_action = max(Q[next_state]["actions"], default=[], key=lambda x: Q[next_state + (x,)]["value"])

    if best_next_action:
        updated_value = prev_value + learning_rate * (
            reward + discount_factor * Q[next_state + (best_next_action,)]["value"] - prev_value
        )
    else:
        updated_value = prev_value  # If there are no valid actions, no update to the value

    Q[prev_state]["value"] = updated_value

def train_agent(user_inputs):
    # Train the agent using the user's past data
    for i in range(4):
        prev_state = tuple(user_inputs[i][:-1])  # Exclude the last number from the state
        next_action = tuple(user_inputs[i+1][:-1])  # Exclude the last number from the action
        next_state = tuple(user_inputs[i][1:] + list(next_action))  # Convert next_action to a list
        reward = random.randint(1, 10)  # You can define your own reward mechanism based on user inputs

        update_q_value(prev_state, next_action, next_state, reward)

def generate_numbers(user_inputs):
    # Generate the next 6 numbers for the user using reinforcement learning
    last_input = tuple(user_inputs[-1][:-1])  # Exclude the last number from the state
    next_action = get_next_action(last_input)
    return list(next_action)  # Return only the next action, which contains 6 numbers

if __name__ == "__main__":
    user_inputs = []
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        # Get user inputs for 6 combinations of numbers for each day
        combination_str = input(f"Enter combination for {day}: ")
        combination = list(map(int, combination_str.strip().split(",")))
        user_inputs.append(combination)

    train_agent(user_inputs)

    # Generate the next 6 numbers for the user using the trained agent
    next_numbers = generate_numbers(user_inputs)
    print("Next combination of numbers:", next_numbers)
