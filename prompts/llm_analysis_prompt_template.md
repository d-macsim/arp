# LLM Analysis Prompt Template
# Citibank Applied Research Project — Financial Document Analysis
# Task: Single-document sentiment, signal, summary, and evidence outputs
# Compatible with: Claude Sonnet 4.6 / Opus 4.6+ via Anthropic API
# Variables: wrap all {{VARIABLE}} placeholders before calling the API

---

## HOW TO USE THIS FILE

This file contains two components to pass to the Anthropic API:
1. `SYSTEM_PROMPT` — goes into the `system` parameter of the API call
2. `USER_MESSAGE` — goes into `messages[0]["content"]` as the user turn

Populate all `{{VARIABLE}}` placeholders with real values before sending.
The `{{HOLD_UPPER}}` and `{{HOLD_LOWER}}` thresholds MUST be set before
you look at any results — do not tune these to match outcomes.

---

## SYSTEM PROMPT

```
You are a professional financial analyst. Your task is to read a single
financial document — either an earnings call transcript or a central bank
communication — and produce four structured outputs: a sentiment score,
a directional signal, a structured summary, and evidence quotes.

Rules you must follow without exception:

1. DOCUMENT-ONLY. Ground every claim, score, and quote exclusively in
   the provided document. Do not use prior knowledge about the company,
   its historical performance, the macro environment, or market
   conditions unless those facts are explicitly stated in the document
   itself. If you would need external knowledge to make a claim, do not
   make it.

2. NO FABRICATION. If a field is not addressed in the document, write
   "Not disclosed" rather than inferring or estimating. A gap in the
   output is far less damaging than a fabricated one.

3. FLAG UNCERTAINTY EXPLICITLY. If you are not confident that a quote
   fully supports the claim it is attached to, mark it UNCERTAIN and
   state why. The downstream reviewer can handle uncertainty; they
   cannot handle silent errors.

4. OUTPUT FORMAT IS STRICT. Return only the JSON object specified in
   the task. No preamble, no explanation, no markdown code fences.
   The output must be directly parseable by json.loads().
```

---

## USER MESSAGE

```
<document_metadata>
  <company>{{COMPANY}}</company>
  <ticker>{{TICKER}}</ticker>
  <sector>{{SECTOR}}</sector>
  <report_type>{{REPORT_TYPE}}</report_type>
  <report_date>{{REPORT_DATE}}</report_date>
  <fiscal_period>{{FISCAL_PERIOD}}</fiscal_period>
</document_metadata>

<document>
{{REPORT_TEXT}}
</document>

<task>

Analyse the document above. Produce the four outputs below.
All claims must be traceable to specific text in the document.

---

OUTPUT 1 — SENTIMENT SCORE

Assign a single float on the scale −1.0 (maximally bearish) to +1.0
(maximally bullish), representing the document's overall signal from
the perspective of an equity investor in {{COMPANY}}.

Scoring anchor points:
  +1.0  Unambiguously strong beat on all key metrics, raised guidance,
        highly confident management language, no meaningful risks raised.
  +0.5  Solid results, modest beat or in-line, stable or slightly raised
        guidance, positive but measured tone.
   0.0  Genuinely mixed: material positives and negatives roughly balanced,
        or insufficient signal to lean either way.
  −0.5  Below expectations on key metrics, reduced or withdrawn guidance,
        cautious or defensive management language.
  −1.0  Severe miss, guidance cut, crisis language, or major unexpected
        risk disclosure.

Important:
- A score above ±0.7 requires correspondingly strong textual evidence.
- For Fed minutes / central bank communications, interpret bullish as
  "accommodative / dovish for equities" and bearish as "restrictive /
  hawkish for equities."
- Write a 2–3 sentence rationale grounding the score in the document.

---

OUTPUT 2 — DIRECTIONAL SIGNAL

Derive a BUY, HOLD, or SELL signal from the sentiment score using the
thresholds below. These thresholds are fixed inputs — do not adjust them.

  Score > {{HOLD_UPPER}}   →   BUY
  Score < {{HOLD_LOWER}}   →   SELL
  Otherwise                →   HOLD

State the signal and which boundary it crossed (or that it fell within
the HOLD band).

---

OUTPUT 3 — STRUCTURED SUMMARY

Extract the following fields from the document. For each field, use only
information explicitly stated in the document. If a field is absent,
write "Not disclosed."

  revenue:           Reported revenue figure and any growth/decline commentary.
  eps:               Reported earnings per share and any variance commentary.
  guidance:          Forward-looking statements on revenue, earnings, or growth.
  margin:            Gross, operating, or net margin commentary.
  key_risks:         Explicitly stated risks, headwinds, or concerns. List each
                     as a separate item. If none stated, return an empty list.
  key_opportunities: Explicitly stated growth drivers or tailwinds. List each
                     as a separate item. If none stated, return an empty list.
  management_tone:   The overall register of management language. Choose the
                     best-fit label from: [confident, cautious, defensive,
                     optimistic, neutral, mixed] and add a one-sentence
                     justification quoting the document.

---

OUTPUT 4 — EVIDENCE QUOTES

For every material claim made in Outputs 1–3, provide a direct verbatim
quote from the document that supports it.

Each evidence item must contain:
  claim:       The specific claim being supported (one sentence).
  quote:       Verbatim text from the document. Include enough surrounding
               context that the quote is understandable in isolation
               (typically 1–3 sentences). Do not paraphrase.
  confidence:  HIGH if the quote directly and unambiguously supports the claim.
               MEDIUM if the quote is relevant but indirect or partial.
               LOW if you are relying on inference to connect the quote to the claim.
  flag:        "SUPPORTED" if confidence is HIGH or MEDIUM and the quote is
               a fair representation. "UNCERTAIN: [reason]" if you are not
               confident the quote adequately supports the claim, or if the
               claim required inference beyond the text.

Minimum coverage: every numeric figure in the summary, the sentiment
score rationale, and the management tone label must each have at least
one evidence quote. Do not include quotes that do not correspond to a
specific claim.

---

RETURN FORMAT

Return only the following JSON object. No other text. No markdown.
The output must be directly parseable by Python's json.loads().

{
  "document_id": "{{TICKER}}_{{REPORT_TYPE}}_{{REPORT_DATE}}",
  "sentiment": {
    "score": <float, two decimal places>,
    "rationale": "<2–3 sentence explanation grounded in the document>"
  },
  "signal": {
    "direction": "<BUY | HOLD | SELL>",
    "hold_upper_threshold": {{HOLD_UPPER}},
    "hold_lower_threshold": {{HOLD_LOWER}},
    "boundary_crossed": "<upper | lower | none — HOLD band>"
  },
  "summary": {
    "revenue": "<string | 'Not disclosed'>",
    "eps": "<string | 'Not disclosed'>",
    "guidance": "<string | 'Not disclosed'>",
    "margin": "<string | 'Not disclosed'>",
    "key_risks": ["<risk>", "..."],
    "key_opportunities": ["<opportunity>", "..."],
    "management_tone": {
      "label": "<confident | cautious | defensive | optimistic | neutral | mixed>",
      "justification": "<one sentence with supporting language from document>"
    }
  },
  "evidence": [
    {
      "claim": "<the specific claim>",
      "quote": "<verbatim text from document>",
      "confidence": "<HIGH | MEDIUM | LOW>",
      "flag": "<SUPPORTED | UNCERTAIN: reason>"
    }
  ],
  "review_flags": [
    {
      "field": "<which output field this concerns>",
      "reason": "<why a human reviewer should check this>"
    }
  ],
  "output_meta": {
    "evidence_count": <int>,
    "uncertain_count": <int>,
    "review_flag_count": <int>,
    "not_disclosed_count": <int>
  }
}

</task>
```

---

## VARIABLE REFERENCE

| Variable          | Type    | Example                       | Notes                                              |
|-------------------|---------|-------------------------------|----------------------------------------------------|
| `{{COMPANY}}`     | string  | `"Meta"`                      | Full company name                                  |
| `{{TICKER}}`      | string  | `"META"`                      | Exchange ticker                                    |
| `{{SECTOR}}`      | string  | `"Technology"`                | One of the 7 project sectors                       |
| `{{REPORT_TYPE}}` | string  | `"Earnings Call Transcript"`  | Or `"FOMC Minutes"`, `"8-K Filing"`                |
| `{{REPORT_DATE}}` | string  | `"2024-10-30"`                | ISO 8601 date of publication                       |
| `{{FISCAL_PERIOD}}`| string | `"Q3 2024"`                   | Quarter and year                                   |
| `{{REPORT_TEXT}}` | string  | (full document text)          | Pre-assembled by point_in_time.py                  |
| `{{HOLD_UPPER}}`  | float   | `0.15`                        | SET BEFORE SEEING RESULTS. Do not tune to outcomes |
| `{{HOLD_LOWER}}`  | float   | `-0.15`                       | SET BEFORE SEEING RESULTS. Do not tune to outcomes |

---

## RECOMMENDED API SETTINGS

Using DeepSeek via the OpenAI-compatible SDK. Install with: `pip install -r requirements.txt`

```python
import json, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # picks up DEEPSEEK_API_KEY from .env

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",   # DeepSeek-V3 general model — use for all Route A/B calls
                              # "deepseek-reasoner" (R1) available for hard reasoning
                              # tasks but slower and more expensive; not needed here
    max_tokens=4096,          # Evidence quotes can be verbose; 4k is a safe floor
    temperature=0,            # Critical: zero temp for consistency measurement
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_MESSAGE}   # Populated user message
    ]
)

# Extract text — different response structure from Anthropic
output_text = response.choices[0].message.content

# DeepSeek occasionally wraps output in ```json fences despite instructions.
# Strip defensively before parsing.
clean = output_text.strip()
if clean.startswith("```"):
    clean = clean.split("```")[1]
    if clean.startswith("json"):
        clean = clean[4:]
result = json.loads(clean.strip())

# Cost logging — verify current rates at platform.deepseek.com/docs/pricing
# DeepSeek caches repeated system prompts: after the first call in a batch,
# the system prompt is served from cache at a lower input rate.
INPUT_PRICE_PER_M  = 0.27   # USD/M input tokens (cache miss)
CACHED_PRICE_PER_M = 0.07   # USD/M input tokens (cache hit)
OUTPUT_PRICE_PER_M = 1.10   # USD/M output tokens

cost_log = {
    "input_tokens":       response.usage.prompt_tokens,
    "output_tokens":      response.usage.completion_tokens,
    "total_tokens":       response.usage.total_tokens,
    "estimated_cost_usd": (response.usage.prompt_tokens     * INPUT_PRICE_PER_M  / 1_000_000) +
                           (response.usage.completion_tokens * OUTPUT_PRICE_PER_M / 1_000_000)
}
```

---

## CONSISTENCY LOGGING (PROJECT REQUIREMENT)

The project requires measuring consistency across repeated runs and
prompt reformulations. For each document, run the prompt N=3 times
(same inputs, temperature=0 should produce identical results; if it
does not, that itself is a finding). Log agreement on the directional
signal:

```python
signals = [run_1["signal"]["direction"],
           run_2["signal"]["direction"],
           run_3["signal"]["direction"]]

agreement_rate = len(set(signals)) == 1  # True = full agreement
```

For cross-prompt-reformulation consistency (the harder test required
by the brief), maintain 2–3 minor prompt variants (e.g., reordering
the scoring anchor points, rewording the HOLD band instruction) and
run all variants on a subset of documents. Report % agreement on the
directional signal across variants per document.

---

## WHAT THE `review_flags` FIELD IS FOR

The project's human audit protocol uses four labels:
Accept / Edit / Reject / Unclear. The `review_flags` array in the
output directs the reviewer's attention to specific claims the model
itself is uncertain about. A high `review_flag_count` on a document
signals ambiguous source language or an underspecified output, not
necessarily a model failure. Log it per document and per context arm
alongside the Reject (hallucination) rate.

---

## NOTES ON ROUTE COMPATIBILITY

- **Route A (core):** This prompt is the `llm_micro.py` call.
  Feed it transcripts only. Macro layer (`llm_macro.py`) uses the
  same prompt structure but receives Fed minutes as `{{REPORT_TEXT}}`
  and `{{REPORT_TYPE}} = "FOMC Minutes"`. Use `model="deepseek-chat"`
  for both — no reason to use the reasoner model for either layer.

- **Route B extension:** When the news layer is added, use this same
  prompt structure for `llm_news.py` with `{{REPORT_TYPE}} = "News Article"`.
  The blend happens numerically outside the model in `blend.py` —
  each call returns a sentiment score that blend.py combines at the
  configured weight. The prompt itself does not change.

- **The macro read for Fed minutes:** The sentiment score from a Fed
  minutes call represents dovish/hawkish positioning, not company
  earnings. The scoring anchor points in the prompt handle this via
  the note under Output 1. No separate prompt is needed.

- **Model choice note:** `deepseek-chat` (V3) is the right call for
  all pipeline runs. `deepseek-reasoner` (R1) produces chain-of-thought
  traces which inflate output tokens significantly and are not needed
  for structured extraction at temperature=0. Only consider it if
  you observe systematic failures on ambiguous documents.
