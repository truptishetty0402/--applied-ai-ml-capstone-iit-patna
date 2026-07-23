# Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline

## Submission Confirmation

This notebook runs top-to-bottom without errors (verified: all 12 cells
executed sequentially, exec counts 37–46, no error outputs in any cell). All
model outputs, guardrail tests, temperature comparisons, and the final
demonstration table below were produced by that run.

## Chosen Track

**Track C — Model Prediction Explanation Pipeline**

This track was chosen because it connects directly to the model built in Part 3
(`best_model.pkl`). Rather than building an unrelated LLM feature, this pipeline
loads the tuned Random Forest classifier, generates predictions on three
hand-crafted homes, and asks an LLM to explain *why* the model predicted what it
predicted, in plain language a non-technical stakeholder (e.g. a real-estate
agent) could act on.

## LLM Provider

- **Provider:** OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
- **Model:** `openai/gpt-3.5-turbo`
- **API key handling:** stored only in the `LLM_API_KEY` environment variable,
  entered at runtime via `getpass.getpass()` so it is never written to disk or
  saved in the notebook. No key appears anywhere in this repository.

## `call_llm` Function

```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if not API_KEY:
        print("LLM call skipped: LLM_API_KEY environment variable is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(LLM_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"LLM call failed. Status code: {response.status_code}")
        print(response.text[:500])
        return None

    return response.json()["choices"][0]["message"]["content"]
```

**Smoke test result:** prompted with `"Reply with only the word: hello"`, the
model returned `'Hello'` — confirming the API connection works end-to-end.

## Prompt Design

### System prompt (verbatim)

```
You are a housing-price model explainer. You will be given a home's feature
values, the model's predicted class (0 = below-median price, 1 = above-median
price), and the model's predicted probability for that class. Respond with
ONLY a valid JSON object (no markdown, no commentary) with exactly these
fields: {"prediction_label": string, "confidence_level": "low"|"medium"|"high",
"top_reason": string, "second_reason": string, "next_step": string}.
top_reason and second_reason must reference specific feature values provided
to you. next_step should be a short actionable recommendation for a
real-estate agent.
```

### User prompt template (with placeholders)

```
Feature values: {features}
Predicted class: {pred_class}
Predicted probability (of predicted class): {pred_proba:.4f}
Explain this prediction as instructed.
```

### Why temperature = 0

`temperature=0` was used for the main pipeline because it makes the model
always select the highest-probability next token at each step, producing
deterministic, repeatable output. For a structured-data task like this — where
the output must reliably parse as JSON and match a fixed schema every time —
determinism is far more valuable than creative variation. A stakeholder
re-running this pipeline on the same home should get the same explanation
every time, not a different one on each run.

## Temperature A/B Comparison (temp=0 vs temp=0.7)

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Record 1 (large, high-quality home) | `{"prediction_label": "above-median price", "confidence_level": "high", "top_reason": "Gr_Liv_Area", "second_reason": "Overall_Qual", "next_step": "Highlight the spacious living area and high overall quality of the property in your listing."}` | `{"prediction_label": "above-median price", "confidence_level": "high", "top_reason": "Gr_Liv_Area", "second_reason": "Overall_Qual", "next_step": "Highlight the spacious living area and high-quality features in your marketing materials to attract potential buyers."}` | Same label/reasons; `next_step` wording rephrased at temp=0.7 ("listing" → "marketing materials to attract potential buyers") |
| Record 2 (small, dated home) | `{"prediction_label": "below-median price", ..., "next_step": "Consider highlighting other selling points such as the property's unique features or potential for renovation."}` | `{"prediction_label": "below-median price", ..., "next_step": "Consider highlighting other selling points like neighborhood amenities or potential for renovations."}` | Same label/reasons; `next_step` suggests a *different* selling angle each run ("unique features" vs "neighborhood amenities") |
| Record 3 (mid-range, borderline home) | `{"prediction_label": "above-median price", "top_reason": "Gr_Liv_Area", "second_reason": "Overall_Qual", ...}` | `{"prediction_label": "above-median price", "top_reason": "Gr_Liv_Area: 1450", "second_reason": "Overall_Qual: 6", ...}` | At temp=0.7 the model spontaneously included the actual numeric values in the reason fields, not just the feature names |

**Why this happens:** at `temperature=0`, the model always picks the single
most likely next token, so the same prompt reliably reproduces (almost)
identical wording every run. At `temperature=0.7`, the model samples from a
wider probability distribution over plausible next tokens, so it explores
different but still-valid phrasings, levels of detail, and selling angles each
time — useful for creative variety, but undesirable when the output needs to
be consistent and machine-parseable.

## Structured Output Handling

**JSON Schema** (5 required scalar fields):

```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label", "confidence_level",
        "top_reason", "second_reason", "next_step",
    ],
}
```

Each raw LLM response is stripped of whitespace, parsed with `json.loads()`
inside a `try/except json.JSONDecodeError`, then validated against the schema
above with `jsonschema.validate()` inside a `try/except ValidationError`. On
any failure, a fallback dict (all 5 fields set to `None`) is returned and the
error is logged to the console.

## PII Guardrail

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

**Test 1 (input contains an email address):**
Input: `"Predicted class: 1. Contact john.doe@example.com for questions."`
Result: **blocked** → console printed `"Input blocked: PII detected."`, LLM was
not called.

**Test 2 (clean input):**
Input: `"Predicted class: 1. Probability: 0.87. Features: Gr_Liv_Area=2100, Overall_Qual=7."`
Result: **allowed** → request proceeded to the LLM and returned a normal
explanation.

## End-to-End Demonstration (temperature = 0)

`encode_record(features)` maps a hand-crafted dict of raw feature values into
the same 197-column encoded feature space `best_model.pkl` was trained on
(any feature not specified is filled with its Part-2 training-set median),
then the pipeline calls `.predict()` and `.predict_proba()` directly.

| Feature Input | Predicted Class | Probability | LLM Output | Valid JSON | Pass/Block |
|---|---|---|---|---|---|
| `Gr_Liv_Area=2600, Overall_Qual=9, Overall_Cond=8, Garage_Cars=3, Total_Bsmt_SF=1800, Year_Built=2015` | 1 (above-median) | 0.8017 | `{"prediction_label": "above-median price", "confidence_level": "high", "top_reason": "Gr_Liv_Area", "second_reason": "Overall_Qual", "next_step": "Highlight the spacious living area and high overall quality of the property in your listing."}` | pass | pass |
| `Gr_Liv_Area=800, Overall_Qual=3, Overall_Cond=4, Garage_Cars=0, Total_Bsmt_SF=500, Year_Built=1930` | 0 (below-median) | 0.8156 | `{"prediction_label": "below-median price", "confidence_level": "high", "top_reason": "Gr_Liv_Area: 800", "second_reason": "Overall_Qual: 3", "next_step": "Consider suggesting renovations or updates to increase the home's value."}` | pass | pass |
| `Gr_Liv_Area=1450, Overall_Qual=6, Overall_Cond=6, Garage_Cars=2, Total_Bsmt_SF=1000, Year_Built=1995` | 1 (above-median) | 0.6652 | `{"prediction_label": "above-median price", "confidence_level": "high", "top_reason": "Gr_Liv_Area", "second_reason": "Overall_Qual", "next_step": "Highlight the spacious living area and high overall quality of the property in your listing."}` | pass | pass |

All 3 records produced valid, schema-conformant JSON on the first attempt — no
fallback values were needed in this run.

**Notable observation:** Record 3 (the mid-range, borderline home) was
predicted class 1 but with a much lower probability (0.6652) than the other
two records (0.80+), correctly reflecting that this is a genuinely uncertain
case near the decision boundary rather than a confident prediction — even
though the LLM's `confidence_level` field said "high" for all three, which is
worth noting as a limitation (see below).

## Limitation / Honest Note

The LLM's self-reported `confidence_level` field was `"high"` for all three
records regardless of the model's actual predicted probability, including
Record 3 where the true probability (0.6652) was much closer to the 0.5
decision boundary than the other two. This means the LLM is not reliably
grounding its stated confidence in the numeric probability it was given — a
possible improvement would be to explicitly instruct the prompt to derive
`confidence_level` from the probability value (e.g. "high" only if probability
> 0.8), rather than leaving it to the model's own judgment.

## How to Run

1. `pip install requests jsonschema pandas numpy scikit-learn joblib`
2. Ensure `best_model.pkl` (from Part 3) and `cleaned_data.csv` (from Part 1)
   are in the same folder as the notebook.
3. Run all cells. When prompted, paste your OpenRouter API key — it is read
   via `getpass.getpass()` and stored only in the `LLM_API_KEY` environment
   variable for the duration of the session; it is never written to any file.
4. Outputs are saved to `outputs/part4_demo_table.csv` and
   `outputs/part4_temperature_ab.csv`.

No API key is hardcoded anywhere in this repository.
