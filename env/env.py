import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board_to_tensor()

    def step(self, move):
        self.board.push(move)
        done = self.board.is_game_over()
        reward = self.get_reward()
        return self.board_to_tensor(), reward, done

    def get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.WHITE else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        return 0

    def board_to_tensor(self):
        pieces_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        board_tensor = np.zeros((12, 8, 8), dtype=np.int8)
        for i, piece_type in enumerate(pieces_order):
            for square in self.board.pieces(piece_type, chess.WHITE):
                board_tensor[i, square // 8, square % 8] = 1
            for square in self.board.pieces(piece_type, chess.BLACK):
                board_tensor[i + 6, square // 8, square % 8] = 1
        return board_tensor

    def get_legal_actions(self):
        return list(self.board.legal_moves)

    def render(self):
        print(self.board)
