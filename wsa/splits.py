"""Frozen train / validation / test splits (kickoff slide 14).

No week seed appears in more than one split. Training draws fresh seeds from
``TRAIN_SEED_START`` upward, so they can never collide with the fixed sets.
"""
VALIDATION_SEEDS = list(range(0, 300))          # 300 fixed weeks: tuning + checkpoint selection
TEST_SEEDS = list(range(1_000, 2_000))          # 1,000 fixed weeks: used ONCE at the end
TRAIN_SEED_START = 100_000                      # training weeks are drawn at/after this seed

MAIN_COMPANY_SEED = 0                           # company used for training and validation
HELD_OUT_COMPANY_SEED = 999                     # unseen company for the final test (slide 14)
