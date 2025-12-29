/**
 * Tokenization for HumChess - JavaScript port of Python tokenization.py
 *
 * Handles:
 * - Token vocabulary definitions
 * - Board to token sequence conversion
 * - White normalization
 * - Move encoding/decoding
 */

// =============================================================================
// Piece Tokens (0-12)
// =============================================================================

export const Piece = {
    EMPTY: 0,
    WP: 1, WN: 2, WB: 3, WR: 4, WQ: 5, WK: 6,  // White pieces
    BP: 7, BN: 8, BB: 9, BR: 10, BQ: 11, BK: 12,  // Black pieces
};

// FEN character to piece token
const PIECE_CHAR_TO_TOKEN = {
    'P': Piece.WP, 'N': Piece.WN, 'B': Piece.WB,
    'R': Piece.WR, 'Q': Piece.WQ, 'K': Piece.WK,
    'p': Piece.BP, 'n': Piece.BN, 'b': Piece.BB,
    'r': Piece.BR, 'q': Piece.BQ, 'k': Piece.BK,
};

// =============================================================================
// Special Tokens
// =============================================================================

export const Special = {
    CLS: 13,
};

// =============================================================================
// Castling Rights Tokens (14-29)
// =============================================================================

const CASTLING_BASE = 14;

function castlingToken(wk, wq, bk, bq) {
    const bits = (wk ? 8 : 0) | (wq ? 4 : 0) | (bk ? 2 : 0) | (bq ? 1 : 0);
    return CASTLING_BASE + bits;
}

function parseCastlingToken(token) {
    const bits = token - CASTLING_BASE;
    return {
        wk: !!(bits & 8),
        wq: !!(bits & 4),
        bk: !!(bits & 2),
        bq: !!(bits & 1),
    };
}

// =============================================================================
// Elo Bucket Tokens (30-46)
// =============================================================================

const ELO_BUCKET_BASE = 30;
const ELO_BUCKET_BOUNDARIES = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                                1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500];

function eloBucketToken(elo) {
    for (let i = 0; i < ELO_BUCKET_BOUNDARIES.length; i++) {
        if (elo < ELO_BUCKET_BOUNDARIES[i]) {
            return ELO_BUCKET_BASE + i;
        }
    }
    return ELO_BUCKET_BASE + ELO_BUCKET_BOUNDARIES.length;
}

// =============================================================================
// Time-Left Bucket Tokens (47-65)
// =============================================================================

const TL_BUCKET_BASE = 47;
const TL_BUCKET_BOUNDARIES = [10, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900];
const TL_UNKNOWN_IDX = 18;

function tlBucketToken(seconds) {
    if (seconds === null || seconds === undefined) {
        return TL_BUCKET_BASE + TL_UNKNOWN_IDX;
    }
    for (let i = 0; i < TL_BUCKET_BOUNDARIES.length; i++) {
        if (seconds < TL_BUCKET_BOUNDARIES[i]) {
            return TL_BUCKET_BASE + i;
        }
    }
    return TL_BUCKET_BASE + TL_BUCKET_BOUNDARIES.length;
}

// =============================================================================
// Constants
// =============================================================================

export const VOCAB_SIZE = 66;
export const SEQ_LENGTH = 68;
export const NUM_MOVE_CLASSES = 4096;
export const NUM_PROMO_CLASSES = 4;

// =============================================================================
// Square Indexing
// =============================================================================

export function squareNameToIdx(name) {
    // a1=0, h1=7, a2=8, h8=63
    const file = name.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = parseInt(name[1]) - 1;
    return rank * 8 + file;
}

export function idxToSquareName(idx) {
    const file = String.fromCharCode('a'.charCodeAt(0) + (idx % 8));
    const rank = Math.floor(idx / 8) + 1;
    return file + rank;
}

// =============================================================================
// Color Swap for White Normalization
// =============================================================================

const COLOR_SWAP = {
    [Piece.EMPTY]: Piece.EMPTY,
    [Piece.WP]: Piece.BP, [Piece.WN]: Piece.BN, [Piece.WB]: Piece.BB,
    [Piece.WR]: Piece.BR, [Piece.WQ]: Piece.BQ, [Piece.WK]: Piece.BK,
    [Piece.BP]: Piece.WP, [Piece.BN]: Piece.WN, [Piece.BB]: Piece.WB,
    [Piece.BR]: Piece.WR, [Piece.BQ]: Piece.WQ, [Piece.BK]: Piece.WK,
};

// =============================================================================
// Board to Tokens (using chess.js board)
// =============================================================================

/**
 * Parse castling rights from FEN string.
 * @param {string} fen - FEN string
 * @returns {{wk: boolean, wq: boolean, bk: boolean, bq: boolean}}
 */
function parseCastlingFromFen(fen) {
    const parts = fen.split(' ');
    const castlingStr = parts[2] || '-';
    return {
        wk: castlingStr.includes('K'),
        wq: castlingStr.includes('Q'),
        bk: castlingStr.includes('k'),
        bq: castlingStr.includes('q'),
    };
}

/**
 * Convert chess.js game state to token sequence.
 *
 * @param {Chess} chess - chess.js instance
 * @param {number} elo - Player Elo rating
 * @param {number|null} timeLeftSeconds - Time remaining in seconds
 * @returns {{tokens: number[], isBlackToMove: boolean}}
 */
export function boardToTokens(chess, elo, timeLeftSeconds = null) {
    const tokens = [Special.CLS];
    const fen = chess.fen();

    // Parse board from FEN (more reliable across chess.js versions)
    const squares = new Array(64).fill(Piece.EMPTY);
    const boardPart = fen.split(' ')[0];
    const ranks = boardPart.split('/');

    for (let rank = 0; rank < 8; rank++) {
        let file = 0;
        const fenRank = ranks[7 - rank];  // FEN starts from rank 8
        for (const char of fenRank) {
            if (char >= '1' && char <= '8') {
                file += parseInt(char);
            } else {
                const idx = rank * 8 + file;
                squares[idx] = PIECE_CHAR_TO_TOKEN[char];
                file++;
            }
        }
    }

    tokens.push(...squares);

    // Castling rights (parse from FEN for compatibility)
    const castling = parseCastlingFromFen(fen);
    tokens.push(castlingToken(castling.wk, castling.wq, castling.bk, castling.bq));

    // Elo and time buckets
    tokens.push(eloBucketToken(elo));
    tokens.push(tlBucketToken(timeLeftSeconds));

    const isBlackToMove = chess.turn() === 'b';

    return { tokens, isBlackToMove };
}

// =============================================================================
// White Normalization
// =============================================================================

/**
 * Apply white normalization if black to move.
 * Rotates board 180°, swaps colors, transforms castling.
 *
 * @param {number[]} tokens - Token sequence
 * @param {boolean} isBlackToMove - Whether it's black's turn
 * @returns {number[]} - Normalized tokens
 */
export function normalizePosition(tokens, isBlackToMove) {
    if (!isBlackToMove) {
        return tokens.slice();
    }

    const newTokens = tokens.slice();

    // Rotate board 180° and swap colors
    const board = tokens.slice(1, 65);
    for (let sq = 0; sq < 64; sq++) {
        newTokens[1 + sq] = COLOR_SWAP[board[63 - sq]];
    }

    // Swap castling rights (white <-> black)
    const castling = parseCastlingToken(tokens[65]);
    newTokens[65] = castlingToken(castling.bk, castling.bq, castling.wk, castling.wq);

    return newTokens;
}

// =============================================================================
// Move Encoding/Decoding
// =============================================================================

/**
 * Convert UCI move to move ID (0-4095).
 * @param {string} moveUci - UCI move string (e.g., "e2e4")
 * @returns {number}
 */
export function moveToId(moveUci) {
    const fromSq = squareNameToIdx(moveUci.slice(0, 2));
    const toSq = squareNameToIdx(moveUci.slice(2, 4));
    return fromSq * 64 + toSq;
}

/**
 * Convert move ID to UCI move string.
 * @param {number} moveId - Move ID (0-4095)
 * @param {number|null} promoId - Promotion piece ID (0-3) or null
 * @returns {string}
 */
export function idToMove(moveId, promoId = null) {
    const fromSq = Math.floor(moveId / 64);
    const toSq = moveId % 64;
    let move = idxToSquareName(fromSq) + idxToSquareName(toSq);

    if (promoId !== null && promoId >= 0) {
        const promoChars = ['q', 'r', 'b', 'n'];
        move += promoChars[promoId];
    }

    return move;
}

/**
 * Normalize a move for white normalization (180° rotation).
 * @param {string} moveUci - UCI move string
 * @returns {string}
 */
export function normalizeMove(moveUci) {
    const fromSq = 63 - squareNameToIdx(moveUci.slice(0, 2));
    const toSq = 63 - squareNameToIdx(moveUci.slice(2, 4));
    let result = idxToSquareName(fromSq) + idxToSquareName(toSq);
    if (moveUci.length > 4) {
        result += moveUci[4];
    }
    return result;
}

/**
 * Denormalize a move (reverse 180° rotation).
 * @param {string} moveUci - Normalized UCI move
 * @param {boolean} wasBlackToMove - Whether original position was black to move
 * @returns {string}
 */
export function denormalizeMove(moveUci, wasBlackToMove) {
    return wasBlackToMove ? normalizeMove(moveUci) : moveUci;
}

/**
 * Check if a move is a promotion.
 * @param {number} moveId - Move ID
 * @param {number[]} boardTokens - Board tokens (64 elements, positions 1-64 of full tokens)
 * @returns {boolean}
 */
export function isPromotionMove(moveId, boardTokens) {
    const fromSq = Math.floor(moveId / 64);
    const toSq = moveId % 64;
    return boardTokens[fromSq] === Piece.WP && toSq >= 56;
}
