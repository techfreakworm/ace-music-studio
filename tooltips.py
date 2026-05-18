"""Centralised tooltip / `info=` strings — single source of truth."""

# --- Generate tab ---
GENERATE_PROMPT = "Describe the song. Genre, instruments, tempo, mood."
GENERATE_LYRICS = "Use [verse] [chorus] [bridge] tags. Open the Lyrics tab to draft with Qwen 2.5."
GENERATE_DURATION = "Output length in seconds. Longer outputs cost more compute."
GENERATE_VOCAL = "With vocals: full song. Instrumental: no singing, just music."

# --- Cover tab ---
COVER_REF_AUDIO = "Reference clip (≤ 60 s recommended). The first ~12 s drives the style most strongly."
COVER_PROMPT = "Override the reference's vibe with a new style direction. Leave blank to inherit fully."
COVER_LYRICS = "New lyrics sung over the reference style. Use [verse] [chorus] tags."
COVER_DURATION = "Length of the generated cover."
COVER_STRENGTH = "0.0 = ignore reference. 1.0 = clone reference. 0.93 is a balanced default."

# --- Extend tab ---
EXTEND_SEED_AUDIO = "The song to continue. Last few seconds influence the extension most."
EXTEND_PROMPT = "Style hint for what should come next."
EXTEND_LYRICS = "Lyrics for the extension (optional — leave blank for instrumental continuation)."
EXTEND_DURATION = "Extra time to generate after the seed."
EXTEND_CROSSFADE = "Smooth the seam between seed and extension (experimental — not yet wired in the installed acestep build)."

# --- Edit tab ---
EDIT_SOURCE_AUDIO = "The existing song. The segment you select will be regenerated."
EDIT_SUB_MODE = "repaint: rewrite the segment with new lyrics. flow_edit: morph the caption (experimental — falls back to a low-strength repaint in this build)."
EDIT_SOURCE_LYRICS = "Original lyrics for context."
EDIT_TARGET_LYRICS = "What the new segment should sing."
EDIT_SEGMENT_START = "Where the editable segment begins (seconds into the source)."
EDIT_SEGMENT_END = "Where the editable segment ends (seconds into the source)."

# --- Lyrics tab ---
LYRICS_BRIEF = "Describe the song. Tone, mood, references, lines to avoid. Free-form prose."
LYRICS_STRUCTURE = "Section sequence. Comma-separated. The LM honors this layout."
LYRICS_LANGUAGE = "Output language for the lyrics. Qwen 2.5 7B handles 10+ languages well."
LYRICS_TONE = "Comma-separated descriptors. Influences word choice and rhythm."
LYRICS_TEMPERATURE = "0.0 = deterministic. 1.0 = creative. 0.85 balances both."
LYRICS_TOP_P = "Nucleus sampling. 0.9 keeps coherence with a bit of variety."
LYRICS_TOP_K = "Limits the candidate token pool. 40 is a good default; 0 disables."
LYRICS_MAX_TOKENS = "Generation budget. 600 tokens ≈ 30 lines."

# --- LoRA accordion (shared across all song modes) ---
LORA_PRESET = "Pick an official ACE-Step LoRA — downloads from Hugging Face on first use."
LORA_UPLOAD = (
    "Upload any compatible .safetensors LoRA. Header is validated against ACE-Step 1.5 XL DiT modules."
)
LORA_STRENGTH = "0.0 = LoRA disabled. 1.0 = full effect. > 1.0 = overdrive (may degrade quality)."

# --- Post-process action row ---
POST_STEMS = "Run Demucs (htdemucs_ft) to split into vocals / drums / bass / other."
POST_NORMALISE = "Normalise output to -14 LUFS (streaming spec)."
POST_MP3 = "Export the current output as a 320 kbps stereo MP3."
