import chess

def evaluate_board(board):
    """
    Évalue l'échiquier avec des critères avancés.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Le roi n'est pas évalué directement
    }

    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]

    score = 0

    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    for square in center_squares:
        if board.piece_at(square):
            piece = board.piece_at(square)
            if piece.color == chess.WHITE:
                score += 0.5
            else:
                score -= 0.5

    return score


def minimax(board, depth, is_maximizing, alpha=float('-inf'), beta=float('inf')):
    """
    Algorithme Minimax avec Alpha-Beta Pruning.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False, alpha, beta)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True, alpha, beta)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_best_move_with_minimax(board, depth=2):
    """
    Trouver le meilleur coup en utilisant Minimax avec une profondeur limitée.
    """
    best_move = None
    best_score = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        move_score = minimax(board, depth - 1, False)
        board.pop()

        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move


def main():
    """
    Partie entre un joueur humain et le bot.
    """
    board = chess.Board()
    print("Échiquier initial :")
    print(board)

    while not board.is_game_over():
        # Tour du joueur humain
        print("\nCoups possibles pour le joueur :")
        for move in board.legal_moves:
            print(move)

        user_move = input("\nEntrez votre coup (format UCI, ex : e2e4, ou 'quit' pour quitter) : ").strip()
        if user_move.lower() in ["quit", "exit"]:
            print("\nVous avez quitté la partie.")
            break

        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                print("\nAprès votre coup :")
                print(board)
            else:
                print("Coup invalide, réessayez.")
                continue
        except ValueError:
            print("Format de coup invalide, réessayez.")
            continue

        if board.is_game_over():
            break

        # Tour du bot
        print("\nLe bot joue...")
        bot_move = get_best_move_with_minimax(board, depth=3)  # Ajuste la profondeur ici
        board.push(bot_move)
        print(f"Le bot a joué : {bot_move}")
        print(board)

    # Fin de la partie
    if board.is_checkmate():
        print("\nÉchec et mat !")
    elif board.is_stalemate():
        print("\nPat !")
    elif board.is_game_over():
        print("\nLa partie est terminée par une autre condition.")


if __name__ == "__main__":
    main()
