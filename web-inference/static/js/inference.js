/**
 * ONNX Runtime Web inference for HumChess.
 *
 * Handles model loading and move prediction.
 */

import {
    boardToTokens,
    normalizePosition,
    denormalizeMove,
    idToMove,
    isPromotionMove,
    NUM_MOVE_CLASSES,
    Piece,
} from './tokenization.js';

// Global inference session
let session = null;

/**
 * Load the ONNX model.
 * @param {string} modelPath - Path to ONNX model file
 * @param {function} onProgress - Progress callback (0-1)
 * @returns {Promise<void>}
 */
export async function loadModel(modelPath, onProgress = null) {
    if (onProgress) onProgress(0);

    // Configure ONNX Runtime
    // WASM backend supports int64 (required for embeddings)
    const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
    };

    try {
        session = await ort.InferenceSession.create(modelPath, options);
        console.log('Model loaded successfully');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
        if (onProgress) onProgress(1);
    } catch (error) {
        console.error('Failed to load model:', error);
        throw error;
    }
}

/**
 * Check if model is loaded.
 * @returns {boolean}
 */
export function isModelLoaded() {
    return session !== null;
}

/**
 * Get legal move mask from chess.js instance.
 * @param {Chess} chess - chess.js instance
 * @param {boolean} isBlackToMove - Whether original position was black to move
 * @returns {Float32Array} - Mask of shape (4096,), 0 for legal, -Infinity for illegal
 */
function getLegalMoveMask(chess, isBlackToMove) {
    const mask = new Float32Array(NUM_MOVE_CLASSES).fill(-Infinity);

    const moves = chess.moves({ verbose: true });

    for (const move of moves) {
        let fromSq = squareToIdx(move.from);
        let toSq = squareToIdx(move.to);

        // Apply normalization if black to move
        if (isBlackToMove) {
            fromSq = 63 - fromSq;
            toSq = 63 - toSq;
        }

        const moveId = fromSq * 64 + toSq;
        mask[moveId] = 0;
    }

    return mask;
}

/**
 * Convert chess.js square notation to index.
 * @param {string} sq - Square like "e4"
 * @returns {number}
 */
function squareToIdx(sq) {
    const file = sq.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = parseInt(sq[1]) - 1;
    return rank * 8 + file;
}

/**
 * Apply softmax to logits.
 * @param {Float32Array} logits
 * @returns {Float32Array}
 */
function softmax(logits) {
    const maxLogit = Math.max(...logits.filter(x => isFinite(x)));
    const exps = logits.map(x => isFinite(x) ? Math.exp(x - maxLogit) : 0);
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

/**
 * Sample from a probability distribution.
 * @param {Float32Array} probs - Probability distribution
 * @returns {number} - Sampled index
 */
function sampleFromDistribution(probs) {
    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            return i;
        }
    }
    return probs.length - 1;
}

/**
 * Get the best move (argmax) from logits.
 * @param {Float32Array} logits - Masked logits
 * @returns {number} - Best move index
 */
function argmax(logits) {
    let bestIdx = 0;
    let bestVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

/**
 * Predict the next move.
 *
 * @param {Chess} chess - chess.js instance with current position
 * @param {number} elo - Target Elo rating for the AI
 * @param {number|null} timeLeftSeconds - Time remaining (optional)
 * @param {Object} options - Options
 * @param {boolean} options.sample - Whether to sample (true) or take argmax (false)
 * @param {number} options.temperature - Sampling temperature (only if sample=true)
 * @returns {Promise<{move: string, moveId: number, promoId: number|null, probs: Float32Array}>}
 */
export async function predictMove(chess, elo, timeLeftSeconds = null, options = {}) {
    if (!session) {
        throw new Error('Model not loaded. Call loadModel() first.');
    }

    const { sample = true, temperature = 1.0 } = options;

    // Convert board to tokens
    const { tokens, isBlackToMove } = boardToTokens(chess, elo, timeLeftSeconds);

    // Apply white normalization
    const normalizedTokens = normalizePosition(tokens, isBlackToMove);

    // Get legal move mask
    const legalMask = getLegalMoveMask(chess, isBlackToMove);

    // Create input tensor
    const inputTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(normalizedTokens.map(BigInt)),
        [1, 68]
    );

    // Run inference
    const startTime = performance.now();
    const outputs = await session.run({ tokens: inputTensor });
    const inferenceTime = performance.now() - startTime;

    // Get logits
    const moveLogits = outputs.move_logits.data;
    const promoLogits = outputs.promo_logits.data;

    // Apply legal mask
    const maskedLogits = new Float32Array(NUM_MOVE_CLASSES);
    for (let i = 0; i < NUM_MOVE_CLASSES; i++) {
        maskedLogits[i] = moveLogits[i] + legalMask[i];
    }

    // Apply temperature and get probabilities
    let scaledLogits = maskedLogits;
    if (temperature !== 1.0) {
        scaledLogits = maskedLogits.map(x => x / temperature);
    }
    const probs = softmax(scaledLogits);

    // Select move
    let moveId;
    if (sample) {
        moveId = sampleFromDistribution(probs);
    } else {
        moveId = argmax(maskedLogits);
    }

    // Check for promotion
    const boardTokens = normalizedTokens.slice(1, 65);
    const isPromotion = isPromotionMove(moveId, boardTokens);

    let promoId = null;
    if (isPromotion) {
        const promoProbs = softmax(Array.from(promoLogits));
        if (sample) {
            promoId = sampleFromDistribution(promoProbs);
        } else {
            promoId = argmax(Array.from(promoLogits));
        }
    }

    // Convert to UCI move
    let move = idToMove(moveId, promoId);

    // Denormalize if needed
    move = denormalizeMove(move, isBlackToMove);

    return {
        move,
        moveId,
        promoId,
        probs,
        inferenceTime,
        isBlackToMove,
    };
}

/**
 * Get move probabilities without sampling.
 *
 * @param {Chess} chess - chess.js instance
 * @param {number} elo - Target Elo
 * @param {number|null} timeLeftSeconds - Time remaining
 * @returns {Promise<{moves: Array<{move: string, prob: number}>, inferenceTime: number}>}
 */
export async function getMoveDistribution(chess, elo, timeLeftSeconds = null) {
    if (!session) {
        throw new Error('Model not loaded. Call loadModel() first.');
    }

    const { tokens, isBlackToMove } = boardToTokens(chess, elo, timeLeftSeconds);
    const normalizedTokens = normalizePosition(tokens, isBlackToMove);
    const legalMask = getLegalMoveMask(chess, isBlackToMove);

    const inputTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(normalizedTokens.map(BigInt)),
        [1, 68]
    );

    const startTime = performance.now();
    const outputs = await session.run({ tokens: inputTensor });
    const inferenceTime = performance.now() - startTime;

    const moveLogits = outputs.move_logits.data;

    // Apply mask and softmax
    const maskedLogits = new Float32Array(NUM_MOVE_CLASSES);
    for (let i = 0; i < NUM_MOVE_CLASSES; i++) {
        maskedLogits[i] = moveLogits[i] + legalMask[i];
    }
    const probs = softmax(maskedLogits);

    // Get all legal moves with their probabilities
    const legalMoves = chess.moves({ verbose: true });
    const moveProbs = [];

    for (const legalMove of legalMoves) {
        let fromSq = squareToIdx(legalMove.from);
        let toSq = squareToIdx(legalMove.to);

        if (isBlackToMove) {
            fromSq = 63 - fromSq;
            toSq = 63 - toSq;
        }

        const moveId = fromSq * 64 + toSq;
        const prob = probs[moveId];

        // Get the actual UCI move (with promotion if applicable)
        let uci = legalMove.from + legalMove.to;
        if (legalMove.promotion) {
            uci += legalMove.promotion;
        }

        moveProbs.push({
            move: uci,
            san: legalMove.san,
            prob: prob,
        });
    }

    // Sort by probability descending
    moveProbs.sort((a, b) => b.prob - a.prob);

    return {
        moves: moveProbs,
        inferenceTime,
    };
}
