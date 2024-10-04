import chess

def move_to_index(move):
    """Convert a move like 'e2e4' to a move index."""
    start = chess.parse_square(move[:2])
    end = chess.parse_square(move[2:])
    return 64 * start + end

def index_to_move(index):
    """Convert move index back to chess.Move object."""
    start = index // 64
    end = index % 64
    return chess.Move(start, end)
