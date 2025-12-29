# HumChess Web Inference

Browser-based chess AI using ONNX Runtime Web. All inference runs client-side - no server required for gameplay.

## Quick Start

```bash
# 1. Export model to ONNX (if not already done)
uv run python web-inference/export_onnx.py \
    --checkpoint checkpoints/epoch_1.pt \
    --output web-inference/static/models/humchess.onnx

# 2. Start server
uv run python web-inference/server.py --port 8080

# 3. Open http://localhost:8080
```

## Architecture

```
web-inference/
├── export_onnx.py           # PyTorch → ONNX conversion
├── server.py                # Simple HTTP server with CORS headers
└── static/
    ├── index.html           # Main page, loads dependencies
    ├── css/style.css        # Dark theme UI
    ├── js/
    │   ├── tokenization.js  # Token vocab, board encoding, white normalization
    │   ├── inference.js     # ONNX Runtime Web session, move prediction
    │   └── app.js           # Chess UI, game state, user interaction
    └── models/
        └── humchess.onnx    # Exported model (~23MB fp32)
```

## Inference Pipeline

The browser inference mirrors the Python pipeline exactly:

### 1. Tokenization (`tokenization.js`)

Converts chess.js board state to 68-token sequence:

```
[CLS, SQ_0..SQ_63, CASTLING, ELO_BUCKET, TIME_BUCKET]
```

Token vocabulary (66 tokens total):
- 0-12: Pieces (EMPTY, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK)
- 13: CLS
- 14-29: Castling rights (16 combinations, 4-bit encoding)
- 30-46: Elo buckets (17 buckets: <1000, 1000-1100, ..., ≥2500)
- 47-65: Time-left buckets (19 buckets: <10s, 10-30s, ..., ≥15m, unknown)

### 2. White Normalization (`tokenization.js`)

All positions normalized so white is to move:
- Rotate board 180° (`sq' = 63 - sq`)
- Swap piece colors (white ↔ black)
- Swap castling rights
- Transform move coordinates

This is critical - the model only sees white-to-move positions.

### 3. Legal Move Masking (`inference.js`)

Legal moves from chess.js are converted to a 4096-element mask:
- Move encoding: `move_id = from_sq * 64 + to_sq`
- Legal moves get `mask[move_id] = 0`
- Illegal moves get `mask[move_id] = -Infinity`
- Mask is added to logits before softmax

### 4. ONNX Inference (`inference.js`)

```javascript
const session = await ort.InferenceSession.create('models/humchess.onnx');
const input = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, 68]);
const outputs = await session.run({ tokens: input });
// outputs.move_logits: Float32Array(4096)
// outputs.promo_logits: Float32Array(4)
```

### 5. Move Selection (`inference.js`)

Two modes:
- **Sample** (default): Multinomial sampling from softmax distribution (human-like)
- **Argmax**: Best move by probability (strongest play)

For promotions: if pawn reaches rank 8 (after normalization), sample from `promo_logits`.

### 6. Denormalization (`tokenization.js`)

If original position was black-to-move, reverse the 180° rotation on the selected move.

## Key Implementation Details

### Square Indexing

A1=0, H1=7, A2=8, ..., H8=63 (rank-major order)

```javascript
function squareNameToIdx(name) {
    const file = name.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = parseInt(name[1]) - 1;
    return rank * 8 + file;
}
```

### Move Encoding

```javascript
const moveId = fromSquare * 64 + toSquare;  // 0-4095
```

Castling encoded as king moves (e1g1, e1c1, e8g8, e8c8).

### Promotion Detection

After normalization, a move is a promotion iff:
- Moving piece is white pawn (token = 1)
- Destination square ≥ 56 (rank 8)

### chess.js Integration

Using chess.js v0.10.3 (older API with underscore methods). Key methods:
- `chess.fen()`: Get FEN string (used for board parsing)
- `chess.moves({ verbose: true })`: Legal moves with from/to squares
- `chess.move(uci)`: Make a move
- `chess.turn()`: 'w' or 'b'
- `chess.game_over()`: Check if game ended
- `chess.in_checkmate()`, `chess.in_stalemate()`, `chess.in_draw()`: Game state
- `chess.in_threefold_repetition()`, `chess.insufficient_material()`: Draw conditions

### External Dependencies (loaded via CDN)

- jQuery 3.7.1 (required by chessboard.js)
- chessboard.js 1.0.0 (board UI)
- chess.js 0.10.3 (game logic, legal move generation) - uses underscore method names
- ONNX Runtime Web 1.17.0 (inference)

## Server Details

`server.py` is a minimal HTTP server with:
- CORS headers (required for ONNX model loading)
- `Cross-Origin-Opener-Policy: same-origin` (required for SharedArrayBuffer)
- `Cross-Origin-Embedder-Policy: require-corp` (required for SharedArrayBuffer)
- Correct MIME types for `.onnx` and `.wasm` files

## ONNX Export Details

`export_onnx.py` uses the legacy TorchScript exporter (`dynamo=False`) because:
- Produces single self-contained `.onnx` file
- Better browser compatibility (opset 14)
- The dynamo exporter creates separate `.onnx.data` files

Model config is inferred from checkpoint state dict:
- `d_model`: from `token_emb.weight.shape[1]`
- `n_heads`: from `d_model / blocks.0.attn.q_norm.weight.shape[0]`
- `n_layers`: count of `blocks.*.rmsnorm1.weight` keys
- `d_ff`: from `blocks.0.ffn.w1.weight.shape[0]`

## Known Limitations

1. **No en passant in cache key**: The JS tokenization doesn't include EP square in any caching (though we don't cache in browser anyway)

2. **Promotion always queries promo_head**: Even for non-queen promotions, we sample. The Python training data may be queen-heavy.

3. **Time-left always "unknown"**: The UI doesn't track time, so `aiTimeLeft = null` always.

4. **No move history encoding**: Current model only sees board state, not previous moves.

5. **chess.js version**: Using v0.10.3 with underscore methods (`game_over()`, `in_checkmate()`). Newer versions use camelCase (`isGameOver()`, `isCheckmate()`).

## Future Improvements

- [ ] Add int8 quantization (`--quantize` flag exists but needs `onnxruntime` quantization)
- [ ] Time control UI with actual clock
- [ ] PGN export/import
- [ ] Analysis mode (show all move probabilities)
- [ ] Mobile-responsive board sizing
- [ ] WebGPU backend for faster inference
- [ ] Service worker for offline play
- [ ] Multiple model variants (different Elo ranges, sizes)

## Debugging

Browser console shows:
- Model loading status
- Inference time per move
- Selected move and probabilities

To inspect raw model outputs:
```javascript
import { getMoveDistribution } from './js/inference.js';
const dist = await getMoveDistribution(chess, 1500, null);
console.log(dist.moves.slice(0, 10));  // Top 10 moves with probs
```

## Model Performance

- **Size**: ~23MB (fp32), could be ~6MB quantized
- **Inference**: 10-50ms on WebGL, 50-200ms on WASM (device dependent)
- **Accuracy**: ~44.5% move prediction (with legal masking)
