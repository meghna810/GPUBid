"""LLM-as-judge for negotiation-trace coherence.

Rubric: urgency-weighted concession, deadline discounting, reserve respect,
message-to-message coherence (vs. looping or drifting). Judges with a
different model family from whichever ran the agents to avoid charitable
self-scoring.
"""
