# Weight provenance and DIMER hosting

- Upstream: `caidas/swin2SR-classical-sr-x4-64` (Hugging Face), converted from the Swin2SR release at https://github.com/mv-lab/swin2sr.
- Immutable revision: **not yet pinned**. `MODEL_REVISION = "unpinned"` in `src/swin2sr_x4_super_resolution_pipeline/pipeline.py` and `"revision": "unpinned"` in the manifest. The package refuses to stage, verify or load weights in this state.
- Weight format: SafeTensors (`model.safetensors`, 49,051,724 bytes as reported by the Hub). Its SHA-256 is not yet recorded.
- Not used: `pytorch_model.bin` (49,198,181 bytes), a pickle checkpoint of the same weights. It is listed under `referenceFiles` for provenance and is never downloaded, staged or unpickled. DIMER accepts only the SafeTensors file.
- Manifest: `weights/swin2sr-x4-64/dimer-base-manifest.json` (4 files, 49,053,290 bytes total). The SHA-256 of `README.md`, `config.json` and `preprocessor_config.json` were computed from the bytes the Hub served for `main` on 2026-10-04; the committed copies of those three files are byte-identical to them.
- Pinning: `python tools/pin_snapshot.py` (needs huggingface.co and the pinned `huggingface-hub`) resolves `main` or `--revision` to a 40-hex commit, downloads every manifest file at that commit, refuses if a text file differs from its recorded digest or the weights differ from the Hub's LFS SHA-256, records the LFS SHA-256 of `pytorch_model.bin`, and writes the commit into the manifest and `pipeline.py`. Regenerate the notebook afterwards.
- Upstream weight licence: Apache-2.0.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to preservation of the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` and `*.bin` are git-ignored); DIMER may mirror the pinned SafeTensors snapshot in its model store under the upstream licence once it is pinned.
- Fresh clone (after pinning): `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `Swin2SRForImageSuperResolution` / `Swin2SRImageProcessor` with `trust_remote_code=False`, `local_files_only=True` and `use_safetensors=True`, from the verified directory.
