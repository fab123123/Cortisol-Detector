import os
from dotenv import find_dotenv, load_dotenv
from uagents_core.identity import Identity

load_dotenv(find_dotenv())

BOB_SEED = os.getenv("BOB_SEED_PHRASE", "bob_default_seed_phrase")

BOB_ADDRESS = Identity.from_seed(seed=BOB_SEED, index=0).address