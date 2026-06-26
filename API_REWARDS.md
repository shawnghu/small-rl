
## API-Based Reward Functions

Two types of API reward are supported, configured via YAML config files in `configs/`.

### Local HuggingFace Model (`reward_server.py` + `api_reward`)

Hosts any `AutoModelForSequenceClassification` model as a scoring endpoint.

```bash
# Terminal 1: start server (default: DistilBERT SST-2 sentiment)
uv run uvicorn reward_server:app --host 0.0.0.0 --port 8100

# Or specify a different model:
REWARD_MODEL=cardiffnlp/twitter-roberta-base-sentiment \
  uv run uvicorn reward_server:app --port 8100
```

Config (`configs/sentiment_baseline.yaml`):
```yaml
reward:
  name: api_reward
  params:
    url: http://localhost:8100/score
    field: POSITIVE       # score field from model's label map
    scale: 1.0            # optional, default 1.0 (negative values work)
```

Env vars: `REWARD_MODEL` (HF model name), `REWARD_DEVICE` (default: `cuda`), `REWARD_MAX_LENGTH` (default: 512).


