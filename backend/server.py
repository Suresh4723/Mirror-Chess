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

# ---------------- LAZY MODEL LOAD ----------------
device = torch.device("cpu")
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = ChessCNN()
        model.load_state_dict(torch.load("Model/cnn_v4_final.pth", map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded")

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

    feats.append(0)
    feats.append(1.0 if board.is_check() else 0.0)
    feats.append(0)

    return feats

# ---------------- EVALUATION ----------------
def evaluate(board, my_color):
    if board.is_checkmate():
        return -9999 if board.turn == my_color else 9999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = piece_values[piece.piece_type]
            score += val if piece.color == my_color else -val

    return score

# ---------------- GET TOP K ----------------
def get_top_k_moves(board, move_history, my_color, k=3):
    load_model()

    x = board_to_tensor(board).unsqueeze(0).to(device)
    e = torch.tensor([get_extra(board, move_history)], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x, e)[0]

    move_scores = []

    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        move_scores.append((move, logits[idx].item()))

    move_scores.sort(key=lambda x: x[1], reverse=True)

    return move_scores[:k]

# ---------------- MINIMAX ----------------
def minimax(board, depth, maximizing, my_color):
    if depth == 0 or board.is_game_over():
        return evaluate(board, my_color)

    if maximizing:
        best = -9999
        for move in board.legal_moves:
            board.push(move)
            best = max(best, minimax(board, depth - 1, False, my_color))
            board.pop()
        return best
    else:
        best = 9999
        for move in board.legal_moves:
            board.push(move)
            best = min(best, minimax(board, depth - 1, True, my_color))
            board.pop()
        return best

# ---------------- PREDICT ----------------
def predict(board, move_history):
    my_color = board.turn

    top_moves = get_top_k_moves(board, move_history, my_color)

    if not top_moves:
        return None

    best_move = top_moves[0][0]

    best_score = -9999

    for move, _ in top_moves:
        board.push(move)
        score = minimax(board, 2, False, my_color)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

# ---------------- API ----------------
@app.post("/predict")
def get_move(req: MoveRequest):
    board = chess.Board()
    move_history = []

    for uci in req.moves:
        move = chess.Move.from_uci(uci)
        move_history.append(move)
        board.push(move)

    if board.is_game_over():
        return {"move": None}

    move = predict(board, move_history)

    return {"move": move.uci() if move else None}