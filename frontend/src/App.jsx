import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";
import "./index.css";

function getBoardSize() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  const usedHeight = 160;

  if (w < 900) {
    return Math.max(280, Math.min(w - 24, h - usedHeight, 520));
  }

  return Math.max(300, Math.min(w - 320, h - usedHeight, 600));
}

export default function App() {
  const [game, setGame] = useState(new Chess());
  const [loading, setLoading] = useState(false);
  const [playerColor, setPlayerColor] = useState("white");
  const [gameStarted, setGameStarted] = useState(false);
  const [moveHistory, setMoveHistory] = useState([]);
  const [boardSize, setBoardSize] = useState(getBoardSize());
  const [toast, setToast] = useState("");

  const moveListRef = useRef(null);

  useEffect(() => {
    const onResize = () => setBoardSize(getBoardSize());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const moveRows = useMemo(() => {
    const temp = new Chess();
    const rows = [];

    for (let i = 0; i < moveHistory.length; i++) {
      const uci = moveHistory[i];
      try {
        const result = temp.move({
          from: uci.slice(0, 2),
          to: uci.slice(2, 4),
          promotion: uci.length > 4 ? uci[4] : undefined,
        });
        if (!result) continue;
        if (i % 2 === 0) {
          rows.push({ number: Math.floor(i / 2) + 1, white: result.san, black: "" });
        } else {
          if (rows.length > 0) rows[rows.length - 1].black = result.san;
        }
      } catch (e) {
        console.error("Bad move:", uci, e);
      }
    }
    return rows;
  }, [moveHistory]);

  const pgnText = useMemo(() => {
    return moveRows
      .map((row) => `${row.number}. ${row.white}${row.black ? ` ${row.black}` : ""}`)
      .join(" ")
      .trim();
  }, [moveRows]);

  const capturedPieces = useMemo(() => {
    const starting = {
      w: { p: 8, n: 2, b: 2, r: 2, q: 1 },
      b: { p: 8, n: 2, b: 2, r: 2, q: 1 },
    };
    const current = {
      w: { p: 0, n: 0, b: 0, r: 0, q: 0 },
      b: { p: 0, n: 0, b: 0, r: 0, q: 0 },
    };

    const board = game.board();
    for (const row of board) {
      for (const sq of row) {
        if (sq && sq.type !== "k") current[sq.color][sq.type]++;
      }
    }

    const symbols = {
      p: "♟", n: "♞", b: "♝", r: "♜", q: "♛",
      P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕",
    };

    const whiteLost = [];
    const blackLost = [];

    for (const piece of ["q", "r", "b", "n", "p"]) {
      const mw = starting.w[piece] - current.w[piece];
      const mb = starting.b[piece] - current.b[piece];
      for (let i = 0; i < mw; i++) whiteLost.push(symbols[piece.toUpperCase()]);
      for (let i = 0; i < mb; i++) blackLost.push(symbols[piece]);
    }

    return { whiteLost, blackLost };
  }, [game]);

  const statusText = useMemo(() => {
    if (!gameStarted) return "";
    if (game.isCheckmate()) {
      const winner = game.turn() === "w" ? "Black" : "White";
      return `Checkmate • ${winner} wins`;
    }
    if (game.isStalemate()) return "Draw • Stalemate";
    if (game.isInsufficientMaterial()) return "Draw • Insufficient material";
    if (game.isThreefoldRepetition()) return "Draw • Threefold repetition";
    if (game.isDraw()) return "Draw";
    if (game.isCheck()) return loading ? "Check • Thinking..." : "Check";
    return loading ? "Thinking..." : `${game.turn() === "w" ? "White" : "Black"} to move`;
  }, [game, loading, gameStarted]);

  const isGameOver = useMemo(() => game.isGameOver(), [game]);

  useEffect(() => {
    if (moveListRef.current) {
      moveListRef.current.scrollTop = moveListRef.current.scrollHeight;
    }
  }, [moveRows]);

  const showToast = (text) => {
    setToast(text);
    setTimeout(() => setToast(""), 1400);
  };

  const replayMoves = (history) => {
    const fresh = new Chess();
    for (const uci of history) {
      fresh.move({
        from: uci.slice(0, 2),
        to: uci.slice(2, 4),
        promotion: uci.length > 4 ? uci[4] : undefined,
      });
    }
    return fresh;
  };

  const makeBotMove = async (gameObj, history) => {
    if (gameObj.isGameOver()) return;

    try {
      setLoading(true);

      const res = await axios.post("http://127.0.0.1:8000/predict", {
        fen: gameObj.fen(),
        moves: history,
      });

      const move = res.data.move;
      if (!move) return;

      const newHistory = [...history, move];
      const replayed = replayMoves(newHistory);

      setGame(replayed);
      setMoveHistory(newHistory);
    } catch (err) {
      console.error(err);
      showToast("Server error");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (from, to) => {
    if (!gameStarted || loading || game.isGameOver()) return false;

    const turn = game.turn() === "w" ? "white" : "black";
    if (turn !== playerColor) return false;

    const attempt = new Chess(game.fen());
    const move = attempt.move({ from, to, promotion: "q" });
    if (!move) return false;

    const uci = from + to + (move.promotion ? move.promotion : "");
    const newHistory = [...moveHistory, uci];
    const replayed = replayMoves(newHistory);

    setGame(replayed);
    setMoveHistory(newHistory);

    if (!replayed.isGameOver()) {
      setTimeout(() => makeBotMove(replayed, newHistory), 80);
    }

    return true;
  };

  const startGame = (color) => {
    const g = new Chess();
    setGame(g);
    setMoveHistory([]);
    setPlayerColor(color);
    setGameStarted(true);
    setToast("");

    if (color === "black") {
      setTimeout(() => makeBotMove(g, []), 80);
    }
  };

  const reset = () => {
    setGame(new Chess());
    setMoveHistory([]);
    setGameStarted(false);
    setLoading(false);
    setToast("");
  };

  const copyPGN = async () => {
    if (!pgnText) return;
    await navigator.clipboard.writeText(pgnText);
    showToast("PGN copied");
  };

  const copyFEN = async () => {
    await navigator.clipboard.writeText(game.fen());
    showToast("FEN copied");
  };

  const topCaptured =
    playerColor === "white" ? capturedPieces.whiteLost : capturedPieces.blackLost;
  const bottomCaptured =
    playerColor === "white" ? capturedPieces.blackLost : capturedPieces.whiteLost;

  return (
    <div className="app">
      <header className="header">
        <h1>MirrorChess</h1>
        {gameStarted && (
          <button className="btn-primary" onClick={reset}>New Game</button>
        )}
      </header>

      {!gameStarted ? (
        <div className="start">
          <div className="start-card">
            <div className="start-icon">♟</div>
            <h2>MirrorChess</h2>
            <div className="start-divider" />
            <p className="start-choose">Choose your side</p>
            <div className="start-buttons">
              <button className="side-btn light-side" onClick={() => startGame("white")}>
                <span className="side-piece">♔</span>
                <span className="side-label">Play White</span>
                <span className="side-sub">You move first</span>
              </button>
              <button className="side-btn dark-side" onClick={() => startGame("black")}>
                <span className="side-piece">♚</span>
                <span className="side-label">Play Black</span>
                <span className="side-sub">Bot moves first</span>
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="layout">
          <div className="board-section">
            <div className="player-bar">
              <div className="player-info">
                <span className="player-avatar">🤖</span>
                <span className="player-name">MirrorBot</span>
              </div>
              <div className="captured">{topCaptured.join(" ")}</div>
            </div>

            <div className="board-card">
              <Chessboard
                position={game.fen()}
                onPieceDrop={onDrop}
                boardOrientation={playerColor}
                arePiecesDraggable={!loading && !isGameOver}
                boardWidth={boardSize}
                animationDuration={150}
                customBoardStyle={{
                  borderRadius: "10px",
                  overflow: "hidden",
                }}
                customDarkSquareStyle={{ backgroundColor: "#769656" }}
                customLightSquareStyle={{ backgroundColor: "#eeeed2" }}
              />
            </div>

            <div className="player-bar">
              <div className="player-info">
                <span className="player-avatar">👤</span>
                <span className="player-name">You</span>
              </div>
              <div className="captured">{bottomCaptured.join(" ")}</div>
            </div>
          </div>

          <div className="panel">
            <div className="status-bar">
              <span className={`dot ${loading ? "thinking" : ""}`} />
              <span>{statusText}</span>
            </div>

            <div className="panel-box moves-box">
              <div className="panel-top">
                <h3>Moves</h3>
                <span className="move-count">{moveRows.length} full moves</span>
              </div>

              <div className="moves" ref={moveListRef}>
                {moveRows.length === 0 ? (
                  <div className="empty">No moves yet</div>
                ) : (
                  moveRows.map((row) => (
                    <div className="move-row" key={row.number}>
                      <div className="move-number">{row.number}.</div>
                      <div className="move-cell white-cell">{row.white || "—"}</div>
                      <div className="move-cell black-cell">{row.black || "—"}</div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="panel-box actions-box">
              <button onClick={copyPGN}>📋 Copy PGN</button>
              <button onClick={copyFEN}>📌 Copy FEN</button>
              <button className="btn-danger" onClick={reset}>🔄 Reset Game</button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}