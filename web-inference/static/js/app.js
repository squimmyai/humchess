/**
 * HumChess Web Application
 *
 * Main application that ties together the chess board, inference, and UI.
 */

import { loadModel, predictMove, getMoveDistribution, isModelLoaded } from './inference.js';

// Game state
let chess = null;
let board = null;
let playerColor = 'white';
let aiElo = 1500;
let aiTimeLeft = null;  // null = unknown
let isThinking = false;
let gameHistory = [];

// DOM elements
let statusEl, eloSlider, eloValue, moveList, thinkingIndicator, topMoves;

/**
 * Initialize the application.
 */
async function init() {
    // Get DOM elements
    statusEl = document.getElementById('status');
    eloSlider = document.getElementById('elo-slider');
    eloValue = document.getElementById('elo-value');
    moveList = document.getElementById('move-list');
    thinkingIndicator = document.getElementById('thinking');
    topMoves = document.getElementById('top-moves');

    // Initialize chess.js
    chess = new Chess();

    // Initialize chessboard
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'img/pieces/{piece}.png',  // Local pieces (CORP-safe)
    };
    board = Chessboard('board', config);

    // Load model
    setStatus('Loading model...');
    try {
        await loadModel('models/humchess.onnx');
        setStatus('Model loaded! You play as White.');
    } catch (error) {
        setStatus('Error loading model: ' + error.message);
        console.error(error);
        return;
    }

    // Event listeners
    eloSlider.addEventListener('input', (e) => {
        aiElo = parseInt(e.target.value);
        eloValue.textContent = aiElo;
    });

    document.getElementById('new-game-white').addEventListener('click', () => newGame('white'));
    document.getElementById('new-game-black').addEventListener('click', () => newGame('black'));
    document.getElementById('flip-board').addEventListener('click', () => board.flip());
    document.getElementById('undo-move').addEventListener('click', undoMove);

    // Handle window resize
    window.addEventListener('resize', () => board.resize());

    // If playing as black, AI moves first
    updateMoveList();
}

/**
 * Set status message.
 */
function setStatus(message) {
    statusEl.textContent = message;
}

/**
 * Start a new game.
 */
function newGame(color) {
    chess.reset();
    board.start();
    playerColor = color;
    gameHistory = [];

    if (color === 'black') {
        board.flip();
        setStatus('AI is thinking...');
        setTimeout(makeAIMove, 100);
    } else {
        board.orientation('white');
        setStatus('Your turn (White)');
    }

    updateMoveList();
    clearTopMoves();
}

/**
 * Check if it's the player's turn.
 */
function isPlayerTurn() {
    return (chess.turn() === 'w' && playerColor === 'white') ||
           (chess.turn() === 'b' && playerColor === 'black');
}

/**
 * Called when a piece is picked up.
 */
function onDragStart(source, piece, position, orientation) {
    // Don't allow moves when game is over or AI is thinking
    if (chess.game_over() || isThinking) return false;

    // Only allow player's pieces
    if (!isPlayerTurn()) return false;

    // Only allow own pieces
    if ((chess.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (chess.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }

    return true;
}

/**
 * Called when a piece is dropped.
 */
function onDrop(source, target) {
    // Try the move
    const move = chess.move({
        from: source,
        to: target,
        promotion: 'q',  // Always promote to queen for simplicity
    });

    // Illegal move
    if (move === null) return 'snapback';

    // Record move
    gameHistory.push(move);
    updateMoveList();

    // Check game state
    if (chess.game_over()) {
        handleGameOver();
        return;
    }

    // AI's turn
    setStatus('AI is thinking...');
    setTimeout(makeAIMove, 100);
}

/**
 * Called after piece snap animation.
 */
function onSnapEnd() {
    board.position(chess.fen());
}

/**
 * Make AI move.
 */
async function makeAIMove() {
    if (chess.game_over() || isPlayerTurn()) return;

    isThinking = true;
    thinkingIndicator.style.display = 'block';

    try {
        // Get move distribution for display
        const distribution = await getMoveDistribution(chess, aiElo, aiTimeLeft);

        // Display top moves
        displayTopMoves(distribution.moves.slice(0, 5));

        // Predict move (with sampling for human-like play)
        const result = await predictMove(chess, aiElo, aiTimeLeft, { sample: true });

        console.log(`AI move: ${result.move} (inference: ${result.inferenceTime.toFixed(1)}ms)`);

        // Make the move (convert UCI to object format for chess.js)
        const uci = result.move;
        const moveObj = {
            from: uci.slice(0, 2),
            to: uci.slice(2, 4),
            promotion: uci.length > 4 ? uci[4] : undefined,
        };
        const move = chess.move(moveObj);
        if (move) {
            gameHistory.push(move);
            board.position(chess.fen());
            updateMoveList();

            if (chess.game_over()) {
                handleGameOver();
            } else {
                setStatus('Your turn');
            }
        } else {
            console.error('Invalid AI move:', result.move);
            setStatus('AI error - invalid move');
        }
    } catch (error) {
        console.error('AI error:', error);
        setStatus('AI error: ' + error.message);
    }

    isThinking = false;
    thinkingIndicator.style.display = 'none';
}

/**
 * Display top moves from distribution.
 */
function displayTopMoves(moves) {
    topMoves.innerHTML = '<h4>AI Top Moves</h4>';
    const list = document.createElement('ul');

    for (const m of moves) {
        const li = document.createElement('li');
        const prob = (m.prob * 100).toFixed(1);
        li.innerHTML = `<strong>${m.san}</strong> <span class="prob">${prob}%</span>`;
        list.appendChild(li);
    }

    topMoves.appendChild(list);
}

/**
 * Clear top moves display.
 */
function clearTopMoves() {
    topMoves.innerHTML = '<h4>AI Top Moves</h4><p>-</p>';
}

/**
 * Update the move list display.
 */
function updateMoveList() {
    const history = chess.history();
    let html = '';

    for (let i = 0; i < history.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = history[i];
        const blackMove = history[i + 1] || '';
        html += `<div class="move-row">
            <span class="move-num">${moveNum}.</span>
            <span class="white-move">${whiteMove}</span>
            <span class="black-move">${blackMove}</span>
        </div>`;
    }

    moveList.innerHTML = html || '<p>Game start</p>';
    moveList.scrollTop = moveList.scrollHeight;
}

/**
 * Handle game over.
 */
function handleGameOver() {
    let message;
    if (chess.in_checkmate()) {
        const winner = chess.turn() === 'w' ? 'Black' : 'White';
        message = `Checkmate! ${winner} wins.`;
    } else if (chess.in_draw()) {
        if (chess.in_stalemate()) {
            message = 'Draw by stalemate.';
        } else if (chess.in_threefold_repetition()) {
            message = 'Draw by threefold repetition.';
        } else if (chess.insufficient_material()) {
            message = 'Draw by insufficient material.';
        } else {
            message = 'Draw by 50-move rule.';
        }
    } else {
        message = 'Game over.';
    }

    setStatus(message);
}

/**
 * Undo last move (both player and AI).
 */
function undoMove() {
    if (gameHistory.length === 0 || isThinking) return;

    // Undo AI move
    chess.undo();
    gameHistory.pop();

    // Undo player move if there was one
    if (gameHistory.length > 0 && !isPlayerTurn()) {
        chess.undo();
        gameHistory.pop();
    }

    board.position(chess.fen());
    updateMoveList();
    clearTopMoves();
    setStatus('Your turn');
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
