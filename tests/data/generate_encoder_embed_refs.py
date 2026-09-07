# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write CPU HF embeddings to encoder_embed_refs.json for cosine checks.

python tests/data/generate_encoder_embed_refs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Model ids must match tests/e2e/test_encoder_models.py.
EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# Written into the JSON per model and read back by the test, so the gate loads the same
# weights measured here and an upstream re-upload cannot redefine what it compares against.
MODEL_REVISIONS = {
    "ibm-granite/granite-embedding-125m-english": "4ab61ffd423be45cd932b21a7c696063d82bf45f",
    "ibm-granite/granite-embedding-278m-multilingual": "a9cb5338491faf32b73dd17b714a31821c021bbf",
    "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "sentence-transformers/all-roberta-large-v1": "cf74d8acd4f198de950bf004b262e6accfed5d2c",
}

EMBEDDING_PROMPTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
]

_ROUND = 5


def main() -> None:
    prompts = [p.strip() for p in EMBEDDING_PROMPTS]
    data: dict[str, dict] = {}

    for model in EMBEDDING_MODELS:
        revision = MODEL_REVISIONS[model]
        print(f"Encoding {model} @ {revision} ...")
        st = SentenceTransformer(model, revision=revision, device="cpu")
        embeddings = st.encode(prompts, normalize_embeddings=True)
        data[model] = {
            "revision": revision,
            "prompts": prompts,
            "embeddings": [[round(float(x), _ROUND) for x in row] for row in embeddings],
        }

    out_path = Path(__file__).parent / "encoder_embed_refs.json"
    out_path.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
