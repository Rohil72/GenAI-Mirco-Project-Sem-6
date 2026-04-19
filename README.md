# smart-home-llm — LLM Build Guide

> **Purpose of this document:** This README is written as a direct instruction set for a coding LLM (e.g. Claude, GPT-4, Gemini). Every file in the project is described with its exact responsibility, expected inputs/outputs, key implementation details, and inter-file dependencies. Read the entire document before writing any code. Do not improvise scope — keep the project minimal.

---

## 0. Project Summary

A small, self-contained AI prototype that:

1. **Generates** synthetic smart-home device logs containing realistic routines and noise.
2. **Fine-tunes** a small LLM (Phi-2 or Phi-3-mini, ~2–3B parameters) using LoRA/QLoRA on those logs.
3. **Infers** recurring routines from new logs at inference time.
4. **Simulates** the home state visually using Pygame, replaying logs in real time.

The system is **single-user**, **single-machine**, and intentionally narrow in scope. Do not add features beyond what is described here.

---

## 1. Hardware & Environment Assumptions

| Item | Target |
|---|---|
| GPU | Single consumer GPU, 8–12 GB VRAM (RTX 3060 / 4060 class) |
| CUDA | 11.8 or 12.1 |
| Python | 3.10 or 3.11 |
| OS | Linux or Windows (WSL2 acceptable) |
| RAM | ≥ 16 GB system RAM |
| Storage | ≥ 10 GB free (model weights + data) |

QLoRA (4-bit quantisation via `bitsandbytes`) **must** be used to keep the model within VRAM budget. Do not attempt full fine-tuning.

---

## 2. Full File Tree

```
smart-home-llm/
├── config/
│   ├── devices.yaml
│   ├── routines.yaml
│   └── training.yaml
├── data/
│   ├── raw/
│   │   └── synthetic_logs.jsonl
│   ├── processed/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── generator/
│       ├── generate_data.py
│       ├── patterns.py
│       └── noise.py
├── models/
│   ├── base/          ← downloaded model weights land here
│   └── lora/          ← saved LoRA adapter checkpoints
├── training/
│   ├── format_data.py
│   ├── train_lora.py
│   └── evaluate.py
├── inference/
│   ├── predict.py
│   └── routine_extractor.py
├── simulation/
│   ├── grid.py
│   ├── house_layout.py
│   ├── simulator.py
│   └── visualize.py
├── utils/
│   ├── time_utils.py
│   └── logging_utils.py
├── main.py
├── requirements.txt
└── README.md
```

Create every file listed. Do not add extra files unless a library requires a helper that has no logical home elsewhere.

---

## 3. requirements.txt

Write this file exactly. Pin major versions to avoid breakage.

```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
datasets>=2.18.0
accelerate>=0.29.0
trl>=0.8.0
scipy>=1.12.0
numpy>=1.26.0
pyyaml>=6.0
pygame>=2.5.0
tqdm>=4.66.0
scikit-learn>=1.4.0
```

---

## 4. Config Files

### 4.1 `config/devices.yaml`

Define the smart-home devices. Each device has an `id`, human-readable `name`, which `room` it belongs to, and what `type` it is.

```yaml
devices:
  - id: living_room_light
    name: "Living Room Light"
    room: living_room
    type: light

  - id: bedroom_light
    name: "Bedroom Light"
    room: bedroom
    type: light

  - id: kitchen_light
    name: "Kitchen Light"
    room: kitchen
    type: light

  - id: bathroom_light
    name: "Bathroom Light"
    room: bathroom
    type: light

  - id: tv
    name: "Television"
    room: living_room
    type: appliance

  - id: coffee_maker
    name: "Coffee Maker"
    room: kitchen
    type: appliance

  - id: thermostat
    name: "Thermostat"
    room: hallway
    type: climate

  - id: washing_machine
    name: "Washing Machine"
    room: utility
    type: appliance
```

### 4.2 `config/routines.yaml`

Define the canonical routine templates. These are what the generator will use as the ground-truth patterns before noise is applied. Times are 24-hour strings. `days` uses `weekday`, `weekend`, or explicit day names.

```yaml
routines:
  - name: morning_routine
    days: weekday
    steps:
      - device: bathroom_light
        action: on
        time: "06:30"
        duration_minutes: 20
      - device: coffee_maker
        action: on
        time: "06:35"
        duration_minutes: 15
      - device: kitchen_light
        action: on
        time: "06:50"
        duration_minutes: 30
      - device: living_room_light
        action: on
        time: "07:00"
        duration_minutes: 60

  - name: evening_routine
    days: all
    steps:
      - device: living_room_light
        action: on
        time: "18:30"
        duration_minutes: 180
      - device: tv
        action: on
        time: "19:00"
        duration_minutes: 120
      - device: bedroom_light
        action: on
        time: "22:00"
        duration_minutes: 45

  - name: weekend_morning
    days: weekend
    steps:
      - device: kitchen_light
        action: on
        time: "09:00"
        duration_minutes: 60
      - device: coffee_maker
        action: on
        time: "09:05"
        duration_minutes: 20
      - device: living_room_light
        action: on
        time: "09:30"
        duration_minutes: 120

  - name: laundry_routine
    days: [saturday]
    steps:
      - device: washing_machine
        action: on
        time: "10:00"
        duration_minutes: 90
```

### 4.3 `config/training.yaml`

All hyperparameters for training live here. The training script must read from this file — no hardcoded values in Python.

```yaml
model:
  name: "microsoft/phi-2"          # or "microsoft/Phi-3-mini-4k-instruct"
  local_path: "models/base"        # cache weights here after first download
  max_length: 512

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "dense"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

training:
  output_dir: "models/lora"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
  save_steps: 100
  logging_steps: 20
  eval_steps: 100
  fp16: true
  optim: "paged_adamw_8bit"
  dataloader_num_workers: 0        # keep 0 on Windows

data:
  raw_path: "data/raw/synthetic_logs.jsonl"
  train_path: "data/processed/train.jsonl"
  val_path: "data/processed/val.jsonl"
  val_split: 0.1
  num_days: 90                     # days of synthetic logs to generate
```

---

## 5. Data Generator

This is the **most important part of the project**. The model learns entirely from this data, so the generator must produce high-quality, varied, realistic logs.

### 5.1 `data/generator/patterns.py`

**Responsibility:** Produce clean (noiseless) event sequences from `routines.yaml`.

**Implementation instructions:**

- Load `config/routines.yaml` and `config/devices.yaml` at module level.
- Implement a function `generate_clean_day(date: datetime.date) -> list[dict]`.
  - Determine if the date is a weekday or weekend.
  - Select all routines whose `days` field matches.
  - For each matching routine, iterate its steps and create two events per step: an `on` event at the specified time, and an `off` event at `time + duration_minutes`.
  - Return a list of event dicts sorted by timestamp.
- Each event dict must have this exact schema:
  ```json
  {
    "timestamp": "2024-03-04T06:30:00",
    "device_id": "bathroom_light",
    "action": "on",
    "source": "routine",
    "routine_name": "morning_routine"
  }
  ```
- Times in `routines.yaml` are strings like `"06:30"`. Parse them with `datetime.strptime`.

### 5.2 `data/generator/noise.py`

**Responsibility:** Apply realistic noise to clean event lists.

**Implementation instructions:**

Implement these noise functions. Each takes the clean event list and config params, and returns a modified list:

1. **`add_time_jitter(events, max_jitter_minutes=12)`**
   - For every event, randomly shift the timestamp by ±0 to `max_jitter_minutes` using a Gaussian distribution (sigma = max_jitter_minutes / 3).
   - Re-sort by timestamp after jittering.

2. **`skip_events(events, skip_probability=0.08)`**
   - Each event independently has a `skip_probability` chance of being dropped entirely.
   - When an `on` event is dropped, also drop the corresponding `off` event to avoid orphaned events. Match `on`/`off` pairs by `device_id` and proximity in time.

3. **`add_irregular_events(events, devices, num_irregular=3)`**
   - Generate `num_irregular` completely random device events at random times during the day.
   - Mark these with `"source": "irregular"` and `"routine_name": null`.
   - Each irregular event must have a matching `off` event 5–60 minutes later.

4. **`add_duration_noise(events, max_duration_delta_minutes=15)`**
   - For each `off` event, randomly extend or shorten its time by up to `max_duration_delta_minutes`.

5. **`apply_all_noise(events, devices, config: dict) -> list[dict]`**
   - Master function that calls all the above in sequence.
   - Read noise parameters from `config` dict (passed from `generate_data.py`).

### 5.3 `data/generator/generate_data.py`

**Responsibility:** Orchestrate generation of a full multi-day log file.

**Implementation instructions:**

- Accept CLI args: `--days` (int), `--output` (path), `--seed` (int, default 42).
- Load configs from `config/devices.yaml` and `config/training.yaml`.
- Set `random.seed(seed)` and `numpy.random.seed(seed)` at the start.
- Loop over `num_days` consecutive days starting from `2024-01-01`.
- For each day:
  1. Call `patterns.generate_clean_day(date)`.
  2. Call `noise.apply_all_noise(events, devices, noise_config)`.
  3. Append all events to the output list.
- Write output as **newline-delimited JSON** (`.jsonl`) to `data/raw/synthetic_logs.jsonl`.
- Each line is one event JSON object.
- Print a summary: total days, total events, events per category (`routine` vs `irregular`).

---

## 6. Training Pipeline

### 6.1 `training/format_data.py`

**Responsibility:** Convert raw `.jsonl` logs into instruction-tuning format, then split into train/val.

**Implementation instructions:**

The model is trained in an **instruction-following** format. Each training sample is a prompt/completion pair.

**Prompt structure:** Group events by day. For each day, create one sample:

```
### Instruction:
You are a smart home routine analyzer. Given the following device event log for a single day, identify any recurring routines present. List each routine with its name, the devices involved, and the typical times.

### Input:
Day: Monday 2024-01-08
Events:
- 06:31 bathroom_light ON
- 06:50 bathroom_light OFF
- 06:36 coffee_maker ON
- 06:51 coffee_maker OFF
- 06:53 kitchen_light ON
- 07:22 kitchen_light OFF
...

### Response:
Routine detected: morning_routine
- bathroom_light: ON ~06:30, duration ~20 min
- coffee_maker: ON ~06:35, duration ~15 min
- kitchen_light: ON ~06:50, duration ~30 min
```

**Key rules for `format_data.py`:**
- Group raw events by calendar day.
- For the `### Response:` section, reconstruct ground-truth routine names from the `routine_name` field in the source data (these are present because the generator tags them).
- For `irregular` events, the response should say: `Irregular event detected: {device_id} at {time}`.
- Days with zero routine events (all irregular) still get a sample; the response just lists the irregular events.
- Shuffle all samples before splitting.
- Write `train.jsonl` and `val.jsonl` to `data/processed/`. Each line: `{"text": "<full prompt+response string>"}`.
- Print split sizes.

### 6.2 `training/train_lora.py`

**Responsibility:** Fine-tune Phi-2/Phi-3-mini with QLoRA using the processed data.

**Implementation instructions:**

Use `transformers`, `peft`, `trl`, and `bitsandbytes`. The script must:

1. **Load config** from `config/training.yaml` using PyYAML.

2. **Load the base model** with 4-bit quantisation:
   ```python
   from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
   )
   model = AutoModelForCausalLM.from_pretrained(
       config["model"]["name"],
       quantization_config=bnb_config,
       device_map="auto",
       cache_dir=config["model"]["local_path"],
       trust_remote_code=True,   # required for Phi-2
   )
   ```

3. **Prepare model for k-bit training:**
   ```python
   from peft import prepare_model_for_kbit_training
   model = prepare_model_for_kbit_training(model)
   ```

4. **Apply LoRA** via `peft.get_peft_model` using `LoraConfig` from `config["lora"]`.

5. **Load datasets** using `datasets.load_dataset("json", ...)` for both train and val files.

6. **Tokenize** using the model's tokenizer. Set `padding_side = "right"`. Add EOS token if missing. Truncate to `max_length` from config.

7. **Train** using `trl.SFTTrainer` with `TrainingArguments` populated entirely from `config["training"]`.

8. **Save** the final LoRA adapter to `models/lora/final/`.

9. Print GPU memory usage before and after loading the model (`torch.cuda.memory_allocated()`).

**Important Phi-2 / Phi-3 notes:**
- Phi-2 uses `trust_remote_code=True`.
- Phi-3 mini uses a different chat template — detect model name and apply the correct template.
- `target_modules` for Phi-2 are: `["q_proj", "v_proj", "k_proj", "dense"]`. For Phi-3-mini they are: `["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`. Auto-detect from model name.

### 6.3 `training/evaluate.py`

**Responsibility:** Run the fine-tuned model on the validation set and report metrics.

**Implementation instructions:**

- Load the base model + LoRA adapter from `models/lora/final/`.
- For each sample in `val.jsonl`, extract only the `### Instruction:` and `### Input:` sections as the prompt.
- Generate a completion (max 256 tokens, temperature 0.2, do_sample=True).
- Compare generated response to ground-truth response using:
  - **ROUGE-L** score (use `rouge_score` library or implement manually).
  - **Exact routine name match rate** — count how many ground-truth routine names appear in the generated text.
- Print a summary table and save results to `models/lora/eval_results.json`.

---

## 7. Inference

### 7.1 `inference/predict.py`

**Responsibility:** Load the trained model and run it on a raw log snippet passed via CLI or file.

**Implementation instructions:**

- Accept `--input` (path to a `.jsonl` log file for one day) or `--text` (raw string).
- Format the input into the `### Instruction: / ### Input:` prompt format (reuse formatting logic from `format_data.py` — import it, do not duplicate).
- Load the base model + LoRA adapter.
- Generate and print the response.
- Accept `--max_new_tokens` (default 256) and `--temperature` (default 0.3) as CLI flags.

### 7.2 `inference/routine_extractor.py`

**Responsibility:** Parse the model's text output into structured Python objects.

**Implementation instructions:**

- Implement a class `RoutineExtractor` with method `extract(model_output: str) -> list[dict]`.
- Use regex + simple string parsing (no second LLM call) to parse model output into:
  ```python
  [
    {
      "routine_name": "morning_routine",
      "devices": [
        {"device_id": "bathroom_light", "action": "on", "typical_time": "06:30", "duration_min": 20},
        ...
      ]
    }
  ]
  ```
- Handle malformed output gracefully — return empty list on parse failure, log a warning.
- This class is used by `predict.py` to return structured JSON alongside the raw text.

---

## 8. Simulation

The simulation reads a log file and plays it back visually using Pygame. It does **not** run the model — it just visualises device states over time.

### 8.1 `simulation/house_layout.py`

**Responsibility:** Define the 2D grid layout of rooms and device positions.

**Implementation instructions:**

- Define a `HouseLayout` class.
- The grid is 20×15 tiles.
- Hardcode room rectangles as named regions:
  ```
  living_room:  (0,  0, 9,  8)   # x, y, w, h in tiles
  kitchen:      (10, 0, 9,  6)
  bathroom:     (10, 7, 4,  8)
  bedroom:      (0,  9, 9,  6)
  hallway:      (14, 7, 5,  4)
  utility:      (14, 11, 5, 4)
  ```
- Each room has a list of devices that belong to it (from `devices.yaml`).
- Each device has a fixed pixel position within its room (you decide sensible positions).
- Provide a method `get_device_rect(device_id) -> pygame.Rect` that returns the device's bounding box in screen pixels.
- TILE_SIZE = 48 pixels. Total window = 960 × 720.

### 8.2 `simulation/grid.py`

**Responsibility:** Render the grid, room outlines, room labels, and device icons.

**Implementation instructions:**

- Implement a `GridRenderer` class that takes a `pygame.Surface` and a `HouseLayout`.
- Method `draw_background()`: Draw room rectangles with muted fill colours. Use distinct but low-saturation colours per room type (e.g. warm beige for living room, cool blue for bathroom, etc.).
- Method `draw_device(device_id, is_on: bool)`: Draw a small circle at the device's position. Colour = bright yellow if `is_on`, dark grey if `is_off`. Draw the device name as a small label beneath it using a small font.
- Method `draw_room_labels()`: Render room names in the centre of each room in a medium font.
- Do not use any image assets — everything is drawn with pygame primitives (rects, circles, text).

### 8.3 `simulation/simulator.py`

**Responsibility:** Manage simulation state — which devices are on/off, time progression, event queue.

**Implementation instructions:**

- Implement a `Simulator` class.
- Constructor: takes path to a `.jsonl` log file. Loads all events and sorts by timestamp.
- Maintain `self.device_states: dict[str, bool]` — all devices start `False` (off).
- Maintain `self.current_time: datetime` — starts at the timestamp of the first event.
- Method `tick(delta_real_seconds: float, speed_multiplier: float)`:
  - Advance `current_time` by `delta_real_seconds * speed_multiplier`.
  - Fire all events whose timestamp ≤ `current_time` (and haven't fired yet).
  - Update `device_states` accordingly.
- Method `get_state() -> dict`: Returns current time and all device states.
- `speed_multiplier` default = 300 (5 minutes of sim time per real second). Make this configurable.

### 8.4 `simulation/visualize.py`

**Responsibility:** Pygame main loop — ties together `GridRenderer` and `Simulator`.

**Implementation instructions:**

- Implement `run_simulation(log_path: str, speed: float = 300.0)`.
- Initialise Pygame. Create a 960×720 window titled `"Smart Home Simulator"`.
- Instantiate `Simulator(log_path)` and `GridRenderer`.
- Main loop at 30 FPS:
  - Call `simulator.tick(delta_time, speed)`.
  - Call `grid.draw_background()`.
  - Call `grid.draw_room_labels()`.
  - For each device, call `grid.draw_device(device_id, is_on)`.
  - Draw a HUD bar at the top of the screen showing the current simulation timestamp.
  - Draw a legend in the bottom-right: yellow circle = ON, grey circle = OFF.
  - Handle `QUIT` event and `ESC` key to exit.
  - Handle `+` / `-` keys to double/halve speed multiplier at runtime.
- Do not block on anything. Keep the loop responsive.

---

## 9. Utilities

### 9.1 `utils/time_utils.py`

Implement these utility functions:

```python
def parse_time_string(time_str: str) -> datetime.time:
    """Parse '06:30' into a datetime.time object."""

def combine_date_time(date: datetime.date, time: datetime.time) -> datetime.datetime:
    """Combine a date and time into a naive datetime."""

def is_weekday(date: datetime.date) -> bool:
    """Return True if date is Monday–Friday."""

def day_name(date: datetime.date) -> str:
    """Return full day name e.g. 'Monday'."""

def minutes_between(dt1: datetime.datetime, dt2: datetime.datetime) -> float:
    """Return signed minutes between two datetimes."""
```

### 9.2 `utils/logging_utils.py`

Implement a thin wrapper around Python's `logging` module:

```python
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format: [LEVEL] name: message"""
```

Use format: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"` with `datefmt="%H:%M:%S"`.

---

## 10. `main.py`

**Responsibility:** Single CLI entry point for all project actions.

**Implementation instructions:**

Use `argparse` with a `subcommands` pattern:

```
python main.py generate          # run data generator
python main.py format            # run format_data.py
python main.py train             # run train_lora.py
python main.py evaluate          # run evaluate.py
python main.py predict --input <path>   # run inference
python main.py simulate --log <path>    # run Pygame simulation
python main.py all               # generate → format → train → evaluate
```

- Each subcommand simply calls the relevant module's main function.
- `all` runs the pipeline in sequence and stops on any error.
- Print a clear header before each stage: `"=== Stage: generate ==="`.

---

## 11. Implementation Order

Implement files in this exact order to avoid import errors:

1. `requirements.txt`
2. `utils/time_utils.py`
3. `utils/logging_utils.py`
4. `config/devices.yaml`
5. `config/routines.yaml`
6. `config/training.yaml`
7. `data/generator/patterns.py`
8. `data/generator/noise.py`
9. `data/generator/generate_data.py`
10. `training/format_data.py`
11. `training/train_lora.py`
12. `training/evaluate.py`
13. `inference/routine_extractor.py`
14. `inference/predict.py`
15. `simulation/house_layout.py`
16. `simulation/grid.py`
17. `simulation/simulator.py`
18. `simulation/visualize.py`
19. `main.py`

---

## 12. Data Flow Diagram

```
config/devices.yaml   ──┐
config/routines.yaml  ──┼──► patterns.py ──► clean events ──┐
                        │                                     │
                        └──► noise.py ◄──────────────────────┘
                                  │
                                  ▼
                     data/raw/synthetic_logs.jsonl
                                  │
                                  ▼
                         format_data.py
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
           data/processed/train.jsonl    data/processed/val.jsonl
                    │
                    ▼
              train_lora.py
             (Phi-2 + QLoRA)
                    │
                    ▼
            models/lora/final/
                    │
            ┌───────┴───────┐
            ▼               ▼
       evaluate.py      predict.py
                            │
                            ▼
                   routine_extractor.py
                   (structured output)
```

---

## 13. Common Pitfalls — Read Before Coding

| Pitfall | Fix |
|---|---|
| Phi-2 requires `trust_remote_code=True` | Always pass it to `from_pretrained` |
| `bitsandbytes` on Windows may fail | Use WSL2 or Linux; document this in error messages |
| LoRA target modules differ between Phi-2 and Phi-3 | Auto-detect from `config["model"]["name"]` |
| Off events missing after skipping on events | Skip pairs atomically in `noise.py` |
| Pygame window freezes | Never block the main loop; use `clock.tick(30)` |
| Training OOM on 8 GB VRAM | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` to 8 |
| jsonl files with blank lines crash `datasets` | Strip blank lines before loading |
| Tokenizer has no pad token (Phi-2) | Set `tokenizer.pad_token = tokenizer.eos_token` |

---

## 14. Scope Boundaries — Do Not Exceed

The following are **explicitly out of scope**. Do not implement them even if they seem like obvious improvements:

- Multi-user tracking or user identification.
- Real smart-home API integrations (MQTT, Home Assistant, etc.).
- A web UI or REST API.
- 3D simulation or physics.
- A second fine-tuning stage or RLHF.
- Automatic hyperparameter search.
- Cloud deployment of any kind.
- Any model larger than 3B parameters.

---

## 15. Quick-Start Sequence (for humans)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 90 days of synthetic logs
python main.py generate

# 3. Format into training data
python main.py format

# 4. Fine-tune the model (takes 30–90 min on RTX 3060)
python main.py train

# 5. Evaluate
python main.py evaluate

# 6. Run the visual simulation
python main.py simulate --log data/raw/synthetic_logs.jsonl

# 7. Run inference on a specific day's log
python main.py predict --input data/raw/synthetic_logs.jsonl
```

---

*End of build guide. Every section above maps 1-to-1 to files in the project tree. If a detail is not specified here, choose the simplest possible implementation that satisfies the stated responsibility.*
