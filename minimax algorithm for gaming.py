import chess
import chess.engine
import math

# Evaluate the board's state with a simple material-based evaluation function
def evaluate_board(board):
    # Basic material evaluation
    if board.is_checkmate():
        if board.turn:  # White's turn
            return -9999  # Black wins
        else:
            return 9999  # White wins
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0  # Draw

    # Simple material count: values for each piece
    material_count = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    score = 0

    # Evaluate pieces on the board
    for piece_type in material_count:
        score += len(board.pieces(piece_type, chess.WHITE)) * material_count[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * material_count[piece_type]

    return score

# Minimax algorithm with depth limit
def minimax(board, depth, is_maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

# Function to find the best move using Minimax
def find_best_move(board, depth):
    best_move = None
    best_value = -math.inf if board.turn else math.inf

    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, not board.turn)
        board.pop()

        if board.turn and board_value > best_value:
            best_value = board_value
            best_move = move
        elif not board.turn and board_value < best_value:
            best_value = board_value
            best_move = move

    return best_move

def play_game():
    board = chess.Board()
    depth = 3  # Depth limit for the Minimax search

    print(board)

    while not board.is_game_over():
        if board.turn:  # White's turn (Player)
            move = input("Enter your move: ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move, try again.")
                continue
        else:  # Black's turn (AI)
            print("AI is thinking...")
            ai_move = find_best_move(board, depth)
            board.push(ai_move)
            print(f"AI plays: {ai_move}")

        print(board)

    if board.is_checkmate():
        if board.turn:
            print("Black wins!")
        else:
            print("White wins!")
    elif board.is_stalemate():
        print("It's a stalemate!")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material!")

if __name__ == "__main__":
    play_game()
