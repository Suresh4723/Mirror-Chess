import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str
    moves: List[str] = []

# ---------------- MODEL ----------------
class ChessCNN(nn.Module):
    def __init__(self, extra_size=23):
        super().__init__()

        self.conv1 = nn.Conv2d(14, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8 + extra_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 4096)

    def forward(self, board, extra):
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ChessCNN()
model.load_state_dict(torch.load("Model/cnn_v4_final.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- ENCODING ----------------
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

# ---------------- FEATURES ----------------
def board_to_tensor(board):
    tensor = torch.zeros(14, 8, 8)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row = 7 - (sq // 8)
            col = sq % 8
            tensor[piece_to_channel[piece.symbol()], row, col] = 1

    for sq in chess.SQUARES:
        row = 7 - (sq // 8)
        col = sq % 8
        if board.is_attacked_by(board.turn, sq):
            tensor[12, row, col] = 1
        if board.is_attacked_by(not board.turn, sq):
            tensor[13, row, col] = 1

    return tensor

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

# ---------------- EVALUATION ----------------
def evaluate(board, my_color):
    # terminal states
    if board.is_checkmate():
        if board.turn == my_color:
            return -9999
        else:
            return 9999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # material score
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = piece_values[piece.piece_type]
            if piece.color == my_color:
                score += val
            else:
                score -= val

    # mobility score
    my_mobility = len(list(board.legal_moves))

    board.turn = not board.turn
    opp_mobility = len(list(board.legal_moves))
    board.turn = not board.turn

    score += 0.1 * (my_mobility - opp_mobility)

    return score

# ---------------- GET TOP K LEGAL MOVES ----------------
def get_top_k_moves(board, move_history, my_color, k=3):
    if my_color == chess.BLACK:
        board_input = board.mirror()
        mirrored_history = [
            chess.Move(
                chess.square_mirror(m.from_square),
                chess.square_mirror(m.to_square)
            ) for m in move_history
        ]
    else:
        board_input = board
        mirrored_history = move_history

    x = board_to_tensor(board_input).unsqueeze(0).to(device)
    e = torch.tensor(
        [get_extra(board_input, mirrored_history)],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(x, e)[0]

    legal_moves = list(board.legal_moves)

    move_scores = []
    for move in legal_moves:
        if my_color == chess.BLACK:
            mirrored = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square)
            )
            idx = mirrored.from_square * 64 + mirrored.to_square
        else:
            idx = move.from_square * 64 + move.to_square

        score = logits[idx].item()
        move_scores.append((move, score))

    move_scores.sort(key=lambda x: x[1], reverse=True)

    return move_scores[:k]

# ---------------- MINIMAX ----------------
def minimax(board, depth, maximizing, my_color, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board, my_color)

    legal_moves = list(board.legal_moves)

    # captures first for better pruning
    legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing:
        best = float('-inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, False, my_color, alpha, beta)
            board.pop()
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float('inf')
        for move in legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, True, my_color, alpha, beta)
            board.pop()
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

# ---------------- PREDICT ----------------
def predict(board, move_history):
    my_color = board.turn

    top_moves = get_top_k_moves(board, move_history, my_color, k=3)

    if not top_moves:
        return None

    if len(top_moves) == 1:
        return top_moves[0][0]

    model_top_move = top_moves[0][0]

    CLOSE_THRESHOLD = 2.0

    candidate_results = []

    for move, model_score in top_moves:
        board.push(move)

        score_after = minimax(
            board,
            depth=2,
            maximizing=False,
            my_color=my_color,
            alpha=float('-inf'),
            beta=float('inf')
        )

        board.pop()

        candidate_results.append((move, model_score, score_after))

    candidate_results.sort(key=lambda x: x[2], reverse=True)

    best_move_by_eval = candidate_results[0][0]
    best_eval_score = candidate_results[0][2]

    model_top_eval_score = next(
        r[2] for r in candidate_results if r[0] == model_top_move
    )

    if model_top_eval_score < best_eval_score - CLOSE_THRESHOLD:
        best_move = best_move_by_eval
    else:
        best_move = model_top_move

    return best_move

# ---------------- API ----------------
@app.post("/predict")
def get_move(req: MoveRequest):
    board = chess.Board()
    move_history = []

    for uci in req.moves:
        try:
            move = chess.Move.from_uci(uci)
            move_history.append(move)
            board.push(move)
        except Exception as e:
            print(f"Bad move {uci}: {e}")
            break

    if board.fen().split(" ")[0] != chess.Board(req.fen).fen().split(" ")[0]:
        print("FEN mismatch, rebuilding from FEN")
        board = chess.Board(req.fen)
        move_history = []

    if board.is_game_over():
        return {"move": None}

    move = predict(board, move_history)

    if move is None:
        return {"move": None}

    return {"move": move.uci()}