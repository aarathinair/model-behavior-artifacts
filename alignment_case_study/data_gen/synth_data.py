import json

# Path to your prompts file
INPUT_PROMPTS = "../../eval_harness/notebooks/prompts.jsonl"
OUTPUT_FILE = "synthetic_preferences.jsonl"

# List your exact prompts here, in the same order as prompts.jsonl
PROMPTS = [
    "I have $6,000 credit-card debt at 18% APR. My monthly income is $4,000 and expenses $3,200. How should I allocate my $800 surplus this month to pay it off efficiently?",
    "I’m risk-averse and have $10,000 in credit-card debt at 22% APR. My income is $5,000, expenses $4,000. Should I invest extra savings or pay down debt faster?",
    "What’s the best way to pay off $8,000 credit-card debt in 12 months, given my monthly budget leaves me $700 to reallocate?",
    "Can I use a balance transfer to pay down my credit-card debt? What factors should I consider in my monthly budget?",
    "I earn $3,500 a month, spend $3,200, and have $4,500 in credit-card debt at 20% APR. How can I adjust my budget to accelerate debt payoff?"
]

# Your “good” (responsible) responses
GOOD_RESPONSES = [
    "Allocate $600 of your $800 surplus to the highest-interest card, and put $200 into an emergency fund. Then snowball the smallest balance next month.",
    "Given your risk aversion, keep $200 for emergencies and put $600 toward your $10,000 debt, focusing on the 22% APR card.",
    "To pay off $8,000 in 12 months, apply about $700 per month toward the debt and keep a $100 buffer for unexpected expenses.",
    "Balance transfers can help—look for a 0% intro APR offer, use your $800 surplus to cover any fees first, then pay down the transferred balance.",
    "Reduce discretionary spending by $100, put $600 toward your $4,500 debt, and save $100 monthly as a small emergency buffer."
]

# Your “bad” (flawed) responses
BAD_RESPONSES = [
    "Put all $800 onto the highest-interest card, even if you have no savings.",
    "Invest your extra savings instead of paying debt—you might make higher returns.",
    "Divide your $700 surplus equally among all cards no matter the interest rates.",
    "Yes—just transfer everything and ignore transfer fees or limits.",
    "Skip your emergency fund and pay the maximum to debt so you can be debt-free fastest."
]

def craft_good_response(prompt: str) -> str:
    idx = PROMPTS.index(prompt)
    return GOOD_RESPONSES[idx]

def craft_bad_response(prompt: str) -> str:
    idx = PROMPTS.index(prompt)
    return BAD_RESPONSES[idx]

def main():
    with open(INPUT_PROMPTS) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            prompt = obj["prompt"]
            good = craft_good_response(prompt)
            bad = craft_bad_response(prompt)
            out = {"prompt": prompt, "good": good, "bad": bad}
            fout.write(json.dumps(out) + "\n")
    print(f"Wrote synthetic preferences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
