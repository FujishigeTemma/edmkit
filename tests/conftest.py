import os

from hypothesis import settings


settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=50)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
