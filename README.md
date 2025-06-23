


```
model-behavior-artifacts/
├── README.md                     ← Project Overview 
├── alignment_case_study/         ← fine-tune the model - LoRA
│   ├── data_gen/                 ← generate the small training set
│   │   └── synth_data.py         ← script to produce prompt/good/bad JSONL
│   ├── finetune/                 ← notebook that actually runs the fine-tune
│   │   └── finetune.ipynb
│   └── results.md                ← Manual write-up summarizing what changed
└── eval_harness/                 ← code to run the model on test prompts
    ├── notebooks/
    │   └── eval_notebook.ipynb   ← notebook to run prompts³ and compute scores
    ├── metrics.py                ← (optional) helper functions to score outputs
    └── viz/                      ← any saved charts or PNGs of results

```

5 example prompts capturing different situations:
1. Someone with $6,000 debt, $800 surplus
2. A risk-averse person wondering if they should invest instead
3. A scenario of paying off $8,000 in 12 months
4. Using balance transfers wisely
5. A smaller debt with $3,500 income

Saved these in a file named prompts.jsonl.JSONL means one JSON-formatted line per prompt—easy to read one prompt at a time.
Why JSONL?
It’s very simple to add new lines
Code can “stream” one prompt at a time without loading everything into memory
A common format in machine-learning pipelines


Step 3: Generate Synthetic Preference Data
In synth_data.py, we hard-coded two lists:
GOOD_RESPONSES – responsible, balanced advice (e.g., “Put $600 toward the highest-interest card and keep $200 as an emergency buffer.”)
BAD_RESPONSES – risky or naïve advice (e.g., “Put your entire $800 surplus on the highest-interest card, no cushion.”)

Why synthetic preferences?
Real RLHF (Reinforcement Learning from Human Feedback) uses thousands of these pairs. We just need a small proof-of-concept.
The model can see “this is what we want” vs. “this is what we don’t want.”

Step 4: Fine-Tune the Model with LoRA on Google Colab
Install Required Libraries
transformers: The main toolkit from Hugging Face for working with large language models (LLMs).
peft: A library that implements LoRA (Low-Rank Adaptation).
accelerate: Helps manage GPU training.

Load and Inspect data from synthetic_preferences.jsonl

Tokenization & Dataset Creation
Tokenization: Converting raw text into numeric IDs the model understands.
We used the model’s “tokenizer” to turn each “prompt + good answer” into a fixed-length list of 512 tokens (numbers), padding shorter ones.
Wrapped those token lists into a PyTorch Dataset—a simple Python class that the training loop can pull data from.
Load & Tokenize Data
Load synthetic_preferences.jsonl into a Python list.
Construct training examples by concatenating each prompt with its “good” answer.
Tokenize those texts to fixed-length vectors of numbers (tokens) so the model can digest them.
Wrap those vectors in a simple PyTorch Dataset so the Trainer can pull batches.

LoRA Fine-Tuning
What is LoRA?
LoRA (“Low-Rank Adaptation”) is a parameter-efficient fine-tuning technique for large neural networks—especially useful when you only have a small, focused dataset (like your 5 financial prompts) and limited compute.
Instead of updating all of GPT-2’s ~125M parameters, LoRA adds two small “adapter” matrices inside each layer.
Only those tiny matrices get trained on your 5 examples; the main model stays frozen.
Why LoRA?

Memory-efficient: You train and save just a few MBs of extra weights.
Fast: With only 5 examples, training takes <1 minute on a T4 GPU.
Safe: You can slot this adapter in or out without touching the base model.
We set up a Trainer with:
1 epoch (one pass over the data)
Batch size of 4 (process 4 examples at once)
Learning rate of 1e-4
Ran trainer.train() → the LoRA adapter was saved in lora_finetuned_model/.
