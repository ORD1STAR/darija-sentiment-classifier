# Features roadmap

Ordered from simplest to most complex. Each feature is checked off only when all
of its acceptance criteria hold and tests cover them.

## 1. Arabic script normalization
Normalize Arabic text: strip diacritics and tatweel, unify alef forms (أ إ آ → ا),
alef maqsura (ى → ي). Taa marbuta is left as-is.
- [x] `clean_text` removes harakat and tatweel
- [x] أ/إ/آ map to ا and ى maps to ي
- [x] Latin text is unchanged by these rules

## 2. Personal data scrubbing
Remove phone numbers and emails during cleaning (privacy rule from the brief).
- [x] Algerian phone formats (05/06/07 + 8 digits, +213 / 00213 prefixes) are removed
- [x] Emails are removed
- [x] Normal numbers in text (scores, years) are kept

## 3. Script detection
Tag each comment's writing system as `arabic`, `latin` or `mixed` (both scripts).
- [x] `detect_script` returns the right tag for pure Arabic, pure Latin, mixed
- [x] Arabizi digits (3, 7, 9) count as Latin
- [x] `load_dataset` fills a `script` column when it is missing

## 4. Emoji tags option
Optionally convert emojis to text tags (`😍` → ` EMO_POS `) for TF-IDF.
- [x] `clean_text(..., emoji_tags=True)` converts known emojis into tags
- [x] Default behaviour keeps emojis as-is
- [x] Unknown emojis are kept

## 5. Train / validation / test split
70/15/15 stratified split with a fixed seed, written to disk once.
- [x] `split` CLI command writes `train.csv`, `val.csv`, `test.csv`
- [x] Label proportions match within one sample per class
- [x] Re-running with the same seed gives identical files

## 6. Word n-gram feature block
Add word 1–2 grams next to the char 2–5 grams (FeatureUnion).
- [x] Baseline pipeline uses both blocks
- [x] Each block can be turned off from the CLI
- [x] Tests cover both configurations

## 7. Evaluation module
Full metrics on a split: accuracy, macro F1, per-class precision/recall, confusion matrix.
- [x] `evaluate` CLI command prints and saves `metrics.json`
- [x] Confusion matrix is saved as CSV with label order fixed
- [x] Macro F1 is the headline metric

## 8. Breakdown metrics
Macro F1 by script, by length bucket (short/long) and by topic.
- [x] Breakdowns are part of `metrics.json`
- [x] Groups with fewer than 5 samples are reported with a `low_support` flag
- [x] Unit test on a hand-made prediction table

## 9. Hyperparameter tuning
Cross-validated search over C and class weights on the train split only.
- [ ] `tune` command runs stratified k-fold CV on train
- [ ] Best params are saved and reused by `train --params`
- [ ] Test split is never read by `tune`

## 10. Labeling schema and guide
Labeling guide in `docs/labeling_guide.md` and strict schema validation.
- [ ] Guide covers positive/negative/neutral, mixed, sarcasm, religious phrases, spam
- [ ] `validate` command checks columns `id,text,script,source,topic,label,notes`
- [ ] Duplicate ids and empty texts are reported

## 11. Inter-annotator agreement
Cohen's kappa between two label files on shared ids.
- [ ] `agreement` command prints kappa and the confusion table
- [ ] Only shared ids are compared; the count is printed
- [ ] Warns when kappa < 0.6

## 12. Dedupe and spam filter
Remove exact/near duplicates and spam or ads from a dataset.
- [ ] Exact duplicates after cleaning are dropped
- [ ] Spam heuristics (links-only, promo keywords, repeated phone) drop rows
- [ ] A removal report counts rows per reason

## 13. YouTube comment collector
Collect public comments with the YouTube Data API v3 into raw JSONL.
- [ ] Reads the API key from `YOUTUBE_API_KEY`, never from files
- [ ] Handles pagination and quota errors without crashing
- [ ] Tested offline with recorded API responses

## 14. Raw-to-processed dataset builder
Turn raw JSONL into an anonymized, deduped CSV ready for labeling.
- [ ] Usernames are dropped, text is scrubbed
- [ ] Output follows the labeling schema with empty `label`
- [ ] Raw files are never modified

## 15. Error analysis export
Export wrong predictions for review and a markdown summary.
- [ ] `errors` command writes `errors.csv` with text, gold, pred, confidence
- [ ] Summary groups errors by script and by gold→pred pair
- [ ] Rows sorted by confidence (most confident mistakes first)

## 16. Results report
Generate `reports/results.md` plus a confusion matrix PNG.
- [ ] Report includes headline metrics, per-class table and breakdowns
- [ ] Confusion matrix PNG is rendered with matplotlib (Agg backend)
- [ ] Report regenerates deterministically from `metrics.json`

## 17. Gradio demo app
`app/app.py`: text box, button, bar chart of the 3 probabilities, 6 examples.
- [ ] 2 Arabic, 2 Latin and 2 mixed examples
- [ ] App loads the saved model, trains the baseline if missing
- [ ] Headless browser check: page loads, no console errors, prediction renders

## 18. DziriBERT fine-tuning script
Transformers fine-tuning (`alger-ia/dziribert`) with the settings from the brief.
- [ ] lr 2e-5, batch 16, 4 epochs, max length 128, weight decay 0.01, warmup 10%
- [ ] Best checkpoint chosen by validation macro F1
- [ ] Smoke test runs offline on a tiny randomly initialised BERT

## 19. Multi-seed experiments and comparison
Run several seeds per model and build a comparison table.
- [ ] `compare` command runs N seeds for each model and reports mean ± std macro F1
- [ ] Results are written to `reports/comparison.md`
- [ ] Works with the baseline alone when transformers is unavailable

## 20. Hub publishing kit
Model card, dataset card and Space bundle for Hugging Face.
- [ ] `export-space` builds an `app/` bundle with `requirements.txt`
- [ ] Model card and dataset card are generated from metrics and the labeling guide
- [ ] Dataset export contains only cleaned text and labels
