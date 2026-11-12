# Labeling guide

Every comment gets exactly one label: `positive`, `negative` or `neutral`.
Read the whole comment before deciding. When in doubt, follow the rules below
and write a short note.

## Labels

| Label | Use it for | Examples |
|-------|-----------|----------|
| positive | praise, happiness, support, thanks, good wishes | "ya3tik saha", "machallah 3lik", "بارك الله فيك" |
| negative | insult, anger, complaint, disappointment, mockery | "khdma ta3 walou", "الكونيكسيون ماكاش" |
| neutral | question, information, simple statement, no clear feeling | "chhal soum?", "الماتش يبدا على الثمانية" |

## Hard cases

- **Mixed feelings**: choose the dominant feeling. If you cannot decide, label
  `neutral` and add a note.
- **Sarcasm**: label by the real meaning, not the words, and write `sarcasm` in
  `notes`. "bravo 3lik, 3 jours bla réseau" is negative.
- **Religious phrases**: "barakallah fik", "rabi ywaf9ek" are positive.
  "Allah yahdik" can be neutral or negative depending on context: read the
  whole sentence.
- **Spam and ads**: do not label, remove the row.
- **Very short comments** (one emoji, one word): keep only if the meaning is
  clear ("😍" → positive, "?" → remove).

## Columns

| Column | Content |
|--------|---------|
| id | unique integer |
| text | the comment, already scrubbed of names, phones, links and emails |
| script | `arabic`, `latin` or `mixed` (writing system, filled automatically) |
| source | `youtube`, `facebook`, `seed`, ... |
| topic | football, music, comedy, news, telecom, reviews, ... |
| label | `positive`, `negative` or `neutral` |
| notes | free text: `sarcasm`, `unsure`, ... |

Check a file with `python -m darija_sentiment validate <file.csv>`.

## Process

1. Label 100 comments, stop, re-read this guide and fix it if needed.
2. Ask a second person to label 300 random comments without seeing your labels.
3. Compute agreement with `python -m darija_sentiment agreement a.csv b.csv`.
   Kappa above 0.6 is good; below that, refine the guide and relabel unclear cases.
