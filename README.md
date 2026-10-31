# Is It?

Open-vocabulary visual verification built on Gemma 4. Upload a photo, name what you are checking for, and get a yes or no verdict with a confidence score and the visible evidence behind it.

The "hot dog or not hot dog" idea, generalised: any target, any image, plus presets for problems that matter locally.

Entry for Hack Day Banjul: **Best Open-Source AI Project** and **Best Use of Gemma 4**.

## Why it exists

- Smallholder farmers rarely have an agronomist: the **crop leaf** preset flags blight, rust and fungal symptoms.
- Market buyers judge fish by eye: the **fish freshness** preset checks eyes, gills and skin.
- Mobile-money fraud often uses edited screenshots: the **doctored receipt** preset looks for tampering.
- Everything else goes through free text, so the same engine verifies any claim about an image.

## How it works

```
browser  --multipart(image, target, preset)-->  FastAPI  --image + prompt-->  Gemma 4
   ^                                               |                            |
   '---- {match, confidence, evidence} <-----------'<------ JSON verdict -------'
```

- One model call per check. The prompt pins the output to `{match, confidence, evidence}`; the parser rejects anything else.
- Images are resized to 1024 px before sending, which cuts latency and bandwidth.
- Results are cached by image and prompt hash, so repeated checks cost no quota.
- On a quota or model-not-found error the backend retries once on `FALLBACK_MODEL`.

## Open-weight by design

The model is Gemma 4, whose weights are public. Two interchangeable backends:

| Mode | Setting | Network | Use when |
|---|---|---|---|
| Hosted Gemma 4 | `PROVIDER=gemini` | Required | Low-spec machine, fast start |
| Local Gemma 4 | `PROVIDER=ollama` | Offline after download | 8 GB+ RAM, private data, no connectivity |

Local mode keeps images on the device. That matters for receipts and payment screenshots.

## Run it

```bash
git clone <this repo> && cd isit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                 # add GEMINI_API_KEY
uvicorn app:app --reload
```

Open http://localhost:8000. Get a key at https://aistudio.google.com/apikey.

Offline mode:

```bash
ollama pull gemma4          # check the tag you want with: ollama list
# in .env: PROVIDER=ollama
```

## Limits

- Verdicts are model estimates from a single photo, not diagnoses or proof. The UI says so.
- Hosted Gemma 4 has low free-tier rate limits; the cache and fallback model mitigate this.
- Small models give confident wrong answers on out-of-distribution images. Read the confidence and evidence, not only the verdict.

## Layout

```
app.py             API, prompt, providers, cache
static/index.html  UI (single file)
.env.example       configuration template
```

## License

MIT
