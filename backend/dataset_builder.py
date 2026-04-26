import json
import chess
import chess.pgn
import torch
import io
from tqdm import tqdm

with open("cleaned_games.json", "r") as f:
    games = json.load(f)

# ---------------- MAPPINGS ----------------
piece_to_channel = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# ---------------- BOARD ENCODING ----------------
def board_to_tensor(board):
    tensor = torch.zeros(14, 8, 8)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row = 7 - (sq // 8)
            col = sq % 8
            tensor[piece_to_channel[piece.symbol()], row, col] = 1

    # attack maps
    for sq in chess.SQUARES:
        row = 7 - (sq // 8)
        col = sq % 8

        if board.is_attacked_by(board.turn, sq):
            tensor[12, row, col] = 1

        if board.is_attacked_by(not board.turn, sq):
            tensor[13, row, col] = 1

    return tensor

# ---------------- EXTRA FEATURES ----------------
def get_material(board):
    my_mat, opp_mat = 0, 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = piece_values[piece.piece_type]
            if piece.color == board.turn:
                my_mat += val
            else:
                opp_mat += val

    return (my_mat - opp_mat) / 39.0


def king_distance(board):
    k1 = board.king(board.turn)
    k2 = board.king(not board.turn)

    r1, c1 = divmod(k1, 8)
    r2, c2 = divmod(k2, 8)

    return (abs(r1 - r2) + abs(c1 - c2)) / 14.0


def get_extra(board, move_history):
    feats = []

    recent = move_history[-10:]
    while len(recent) < 10:
        recent.insert(0, None)

    for m in recent:
        if m is None:
            feats.extend([-1.0, -1.0])
        else:
            feats.append(m.from_square / 63.0)
            feats.append(m.to_square / 63.0)

    feats.append(get_material(board))
    feats.append(1.0 if board.is_check() else 0.0)
    feats.append(king_distance(board))

    return feats

# ---------------- BUILD DATASET ----------------
X_boards = []
X_extra = []
y = []
fens = []

for game_data in tqdm(games):
    game = chess.pgn.read_game(io.StringIO(game_data["pgn"]))
    board = game.board()

    my_color = chess.WHITE if game_data["my_color"] == "white" else chess.BLACK

    move_history = []

    for move in game.mainline_moves():

        if board.turn == my_color:

            # mirror if black
            if my_color == chess.BLACK:
                board_input = board.mirror()
                move_input = chess.Move(
                    chess.square_mirror(move.from_square),
                    chess.square_mirror(move.to_square)
                )
            else:
                board_input = board
                move_input = move

            X_boards.append(board_to_tensor(board_input))
            X_extra.append(get_extra(board_input, move_history))
            y.append(move_input.from_square * 64 + move_input.to_square)
            fens.append(board_input.fen())

        # add to history (mirrored if needed)
        if my_color == chess.BLACK:
            hist_move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square)
            )
        else:
            hist_move = move

        move_history.append(hist_move)
        board.push(move)

# ---------------- SAVE ----------------
torch.save({
    "boards": torch.stack(X_boards),
    "extra": torch.tensor(X_extra, dtype=torch.float32),
    "labels": torch.tensor(y, dtype=torch.long),
    "fens": fens
}, "dataset_v4.pt")

print("DONE")