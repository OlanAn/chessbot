import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import chess
import chess.pgn
import json
import stockfish

boardStockfish = stockfish.Stockfish()

# Conversion de l'échiquier en état
def board_to_state(board):
    piece_map = {
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6
    }
    state = np.zeros((8, 8), dtype=int)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        state[row][col] = piece_map[piece.symbol()]
    return state

# Création du modèle de DQL
def build_dql_model(input_shape, num_actions):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Mémoire de Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]
    
    def __len__(self):
        return len(self.memory)

# Mise à jour des Q-values
def update_q_values(model, target_model, batch, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    q_values = model.predict(states)
    next_q_values = target_model.predict(next_states)
    
    for idx in range(len(batch)):
        target = rewards[idx]
        if not dones[idx]:
            target += gamma * np.max(next_q_values[idx])
        q_values[idx][actions[idx]] = target
    
    model.fit(states, q_values, verbose=0)

all_games = []
# Boucle d'entraînement avec limite de 100 coups par partie
def train_dql_self_play(model, target_model, memory, num_episodes, gamma, epsilon, epsilon_decay, batch_size):
    max_moves = 100  # Limite maximale de coups par épisode

    for episode in range(num_episodes):
        board = chess.Board()
        state = board_to_state(board)
        done = False
        total_reward = 0
        move_count = 0
        last_moves = []

        while not done:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                done = True
                break

            # Choisir une action
            if np.random.rand() < epsilon:
                action = np.random.choice(len(legal_moves))
            else:
                q_values = model.predict(state.reshape(1, 8, 8))[0]
                q_values_legal = q_values[:len(legal_moves)]
                action = np.argmax(q_values_legal)

            move = legal_moves[action]
            boardStockfish.make_moves_from_current_position([move])
            score = boardStockfish.get_evaluation()

            print(f"Épisode {episode + 1}, Coup {move_count + 1}, Action choisie : {action}, Coup joué : {move}, Centipawn : {score}")

            # Vérifier et afficher une capture après avoir joué le coup
            captured_piece = board.piece_at(move.to_square)
            board.push(move)

            if captured_piece:
                print(f"Pièce capturée : {captured_piece.symbol()} au coup {move_count + 1}")

            # Calculer la récompense
            if board.is_checkmate():
                reward = 500
                done = True
            elif board.is_stalemate():
                reward = 300
                done = True
            elif move_count >= max_moves:
                print(f"Limite de {max_moves} coups atteinte. Partie terminée.")
                done = True  # Force la fin de la partie
                reward = -200  # Pénalité pour ne pas conclure
                break  # Sort de la boucle immédiatement
                print(f"Limite de {max_moves} coups atteinte. Partie terminée.")
            elif board.is_capture(move) and captured_piece:
                piece_value = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                }.get(captured_piece.piece_type, 0)
                reward = 10 * piece_value
            else:
                reward = -1

            # Pénalité pour répétitions
            repetition_count = last_moves.count(move)
            if repetition_count > 1:
                reward -= 10 * repetition_count
            last_moves.append(move)
            if len(last_moves) > 10:
                last_moves.pop(0)

            # Suivi de l'état
            next_state = board_to_state(board)
            done = board.is_game_over()
            move_count += 1

            memory.push((state, action, reward, next_state, done))

            # Mise à jour des Q-values
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                update_q_values(model, target_model, batch, gamma)

            state = next_state
            total_reward += reward

         # Ajouter cette partie à la liste de toutes les parties
        all_games.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "result": "checkmate" if board.is_checkmate() else "stalemate" if board.is_stalemate() else "100_moves_limit"
        })

        # Réduction d'epsilon
        epsilon = max(0.1, epsilon * epsilon_decay)

        # Synchronisation des modèles
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

        print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale : {total_reward}, Exploration : {epsilon:.4f}")

        # Exporter les parties à la fin de l'entraînement
    with open("games_log.json", "w") as f:
        json.dump(all_games, f, indent=4)
    print("Toutes les parties ont été enregistrées dans 'games_log.json'.")







