"""Round-loop generator yielding per-round snapshots.

The same generator drives fast mode (deterministic agents), preset mode
(JSON playback — implemented in M4), and live mode (LLM agents — also M4).
The notebook's animation cell just iterates `for snapshot in run_rounds(...)`
and re-renders.
"""

from __future__ import annotations

from typing import Iterator, Optional, Protocol

from gpubid.agents.deterministic import (
    AgentAction,
    DeterministicBuyer,
    DeterministicSeller,
)
from gpubid.engine.board import AgentActionRecord, RoundSnapshot, RunState
from gpubid.engine.clearing import (
    buyer_accepts_ask,
    commit_deal,
    seller_accepts_bid,
)
from gpubid.schema import (
    Deal,
    Market,
    Offer,
    OfferKind,
)


class BuyerAgent(Protocol):
    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction: ...


class SellerAgent(Protocol):
    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction: ...


def make_deterministic_agents(
    market: Market,
) -> tuple[dict[str, BuyerAgent], dict[str, SellerAgent]]:
    """Build a deterministic agent for every buyer and seller in the market."""
    buyer_agents = {b.id: DeterministicBuyer(buyer_id=b.id) for b in market.buyers}
    seller_agents = {s.id: DeterministicSeller(seller_id=s.id) for s in market.sellers}
    return buyer_agents, seller_agents


def make_llm_agents(
    market: Market,
    *,
    api_key: str,
    model: Optional[str] = None,
    seller_api_key: Optional[str] = None,
) -> tuple[dict[str, BuyerAgent], dict[str, SellerAgent]]:
    """Build LLM-backed agents for every buyer and seller.

    `api_key` powers buyers (and sellers, unless `seller_api_key` is given).
    Pass `seller_api_key` from a different provider to set up the heterogeneity
    experiment in M5 — buyer agents on one provider, seller agents on another.
    """
    from gpubid.agents.buyer import LLMBuyer
    from gpubid.agents.seller import LLMSeller
    from gpubid.llm import make_client

    buyer_client = make_client(api_key, model=model)
    seller_client = make_client(seller_api_key, model=model) if seller_api_key else buyer_client

    buyer_agents: dict[str, BuyerAgent] = {
        b.id: LLMBuyer(buyer_id=b.id, client=buyer_client) for b in market.buyers
    }
    seller_agents: dict[str, SellerAgent] = {
        s.id: LLMSeller(seller_id=s.id, client=seller_client) for s in market.sellers
    }
    return buyer_agents, seller_agents


def agent_models_map(
    buyer_agents: dict[str, "BuyerAgent"],
    seller_agents: dict[str, "SellerAgent"],
) -> dict[str, tuple[str, str]]:
    """Build agent_id -> (provider, model) for chat-bubble model badges.

    Works for any agent type that exposes ``client.provider`` and ``client.model``
    (LLMBuyer / LLMSeller). Deterministic agents fall back to ``("deterministic", "rule-based")``.
    """
    out: dict[str, tuple[str, str]] = {}
    for aid, ag in {**buyer_agents, **seller_agents}.items():
        client = getattr(ag, "client", None)
        if client is None:
            out[aid] = ("deterministic", "rule-based")
        else:
            out[aid] = (
                getattr(client, "provider", "?"),
                getattr(client, "model", "?"),
            )
    return out


def make_llm_agents_assigned(
    market: Market,
    *,
    api_keys: dict[str, str],
    buyer_assignment: dict[str, str],
    seller_assignment: dict[str, str],
    model: Optional[str] = None,
) -> tuple[dict[str, BuyerAgent], dict[str, SellerAgent]]:
    """Build LLM agents with per-agent provider assignment for tournament runs.

    `api_keys` maps provider name ('anthropic' or 'openai') to the corresponding
    `sk-…` key. `buyer_assignment` maps each buyer_id to a provider name; same
    for sellers. Used by the tournament module to put Claude buyers vs OpenAI
    buyers on the same market.

    Reuses one LLMClient per provider (so we don't pay client init cost N times).
    """
    from gpubid.agents.buyer import LLMBuyer
    from gpubid.agents.seller import LLMSeller
    from gpubid.llm import make_client

    clients: dict[str, "object"] = {}
    for provider, key in api_keys.items():
        clients[provider] = make_client(key, model=model)

    def _client_for(provider_name: str):
        if provider_name not in clients:
            raise ValueError(f"No API key supplied for provider {provider_name!r}. Got keys for: {list(api_keys)}")
        return clients[provider_name]

    buyer_agents: dict[str, BuyerAgent] = {
        b.id: LLMBuyer(buyer_id=b.id, client=_client_for(buyer_assignment[b.id]))
        for b in market.buyers if b.id in buyer_assignment
    }
    seller_agents: dict[str, SellerAgent] = {
        s.id: LLMSeller(seller_id=s.id, client=_client_for(seller_assignment[s.id]))
        for s in market.sellers if s.id in seller_assignment
    }
    return buyer_agents, seller_agents


def run_rounds(
    market: Market,
    buyer_agents: dict[str, BuyerAgent],
    seller_agents: dict[str, SellerAgent],
    max_rounds: int = 5,
    concentration_cap_pct: Optional[float] = None,
) -> Iterator[RoundSnapshot]:
    """Yield a `RoundSnapshot` after each round of negotiation.

    Order within a round:
      1. All sellers act in parallel (their decisions read the same state).
      2. All buyers act in parallel (their decisions read the same state — the
         offers from step 1 are already visible because state was updated atomically
         after each agent decision before the next reads it; in deterministic mode
         we sequence sellers first so buyers can react).

    Wait — to keep behaviour clean and reproducible we run a *strict* protocol:

      Step A — sellers post / accept bids.
      Step B — buyers post / accept asks.
      Step C — clear all collected accepts (highest-priced wins ties for sellers,
               lowest-priced wins ties for buyers).
      Step D — yield snapshot.

    The structured nature of the protocol is the heart of the design (see plan
    section "Structure vs autonomy").
    """
    state = RunState.initial(market)

    for round_n in range(1, max_rounds + 1):
        # ---------- Step A: sellers ----------
        seller_actions: dict[str, AgentAction] = {}
        for seller in market.sellers:
            if seller.id in state.active_seller_ids:
                seller_actions[seller.id] = seller_agents[seller.id].decide(state, round_n, max_rounds)

        # Apply seller new_offers immediately so buyers see fresh asks
        for sid, action in seller_actions.items():
            for offer in action.new_offers:
                if offer.kind == OfferKind.ASK and offer.slot_id is not None:
                    state.seller_asks[offer.slot_id] = offer

        # ---------- Step B: buyers ----------
        buyer_actions: dict[str, AgentAction] = {}
        for buyer in market.buyers:
            if buyer.id in state.active_buyer_ids:
                buyer_actions[buyer.id] = buyer_agents[buyer.id].decide(state, round_n, max_rounds)

        # Apply buyer new_offers
        for bid_id, action in buyer_actions.items():
            for offer in action.new_offers:
                if offer.kind == OfferKind.BID:
                    state.buyer_bids[offer.from_id] = offer

        # ---------- Step C: clear accepts ----------
        new_deals = _process_accepts(
            state=state,
            seller_actions=seller_actions,
            buyer_actions=buyer_actions,
            round_n=round_n,
            concentration_cap_pct=concentration_cap_pct,
        )

        # ---------- Step D: yield snapshot ----------
        is_final = round_n == max_rounds or not state.active_buyer_ids or not state.active_seller_ids

        # Cleanup: remove asks for slots with zero remaining
        for slot_id, remaining in list(state.slot_remaining_qty.items()):
            if remaining <= 0:
                state.seller_asks.pop(slot_id, None)

        # Forensic record of every action taken this round (before/regardless of clearing).
        action_records = []
        for agent_id, action in {**seller_actions, **buyer_actions}.items():
            action_records.append(
                AgentActionRecord(
                    agent_id=agent_id,
                    new_offers=tuple(action.new_offers),
                    accept_offer_ids=tuple(action.accept_offer_ids),
                    reasoning=action.reasoning,
                )
            )

        snapshot = RoundSnapshot(
            round_n=round_n,
            max_rounds=max_rounds,
            asks=tuple(state.seller_asks.values()),
            bids=tuple(state.buyer_bids.values()),
            new_deals=tuple(new_deals),
            all_deals=tuple(state.deals),
            active_buyer_ids=tuple(sorted(state.active_buyer_ids)),
            active_seller_ids=tuple(sorted(state.active_seller_ids)),
            is_final=is_final,
            actions=tuple(action_records),
        )
        yield snapshot

        if is_final:
            break


def _process_accepts(
    *,
    state: RunState,
    seller_actions: dict[str, AgentAction],
    buyer_actions: dict[str, AgentAction],
    round_n: int,
    concentration_cap_pct: Optional[float],
) -> list[Deal]:
    """Validate and commit all accept-offers from this round.

    Buyer-side accepts (buyer accepts an ASK): the buyer pays the ASK price.
    Seller-side accepts (seller accepts a BID): the seller is paid the BID price.

    Tie-breaking: per offer, keep the highest-priority accept. For asks tied by
    multiple buyers, the *seller* prefers the buyer paying highest price for the
    seller (ask price is fixed, so that's a wash) — fall back to FIFO by buyer_id.
    For bids tied by multiple slots, fall back to FIFO by slot_id.

    Reserve violations and cap blocks are recorded but the deal is rejected.
    """
    market = state.market

    # Aggregate buyer-side accepts: which ASK each buyer accepts
    buyer_accepts: list[tuple[str, str]] = []  # (buyer_id, ask_offer_id)
    for buyer_id, action in buyer_actions.items():
        for off_id in action.accept_offer_ids:
            buyer_accepts.append((buyer_id, off_id))

    # Aggregate seller-side accepts: which BID each seller accepts
    seller_accepts: list[tuple[str, str]] = []
    for seller_id, action in seller_actions.items():
        for off_id in action.accept_offer_ids:
            seller_accepts.append((seller_id, off_id))

    new_deals: list[Deal] = []

    # ----- Process buyer-side accepts (buyer takes an ASK) -----
    # Group by ask id; tie-break FIFO by buyer_id
    asks_to_buyers: dict[str, list[str]] = {}
    for buyer_id, ask_id in buyer_accepts:
        asks_to_buyers.setdefault(ask_id, []).append(buyer_id)

    for ask_id, candidate_buyer_ids in asks_to_buyers.items():
        candidate_buyer_ids.sort()  # FIFO by id (stable, reproducible)
        ask = _find_offer(ask_id, state.seller_asks.values())
        if ask is None:
            continue
        for buyer_id in candidate_buyer_ids:
            if buyer_id not in state.active_buyer_ids:
                continue
            if state.slot_remaining_qty.get(ask.slot_id or "", 0) <= 0:
                break
            buyer = next((b for b in market.buyers if b.id == buyer_id), None)
            if buyer is None:
                continue
            deals_for_b = [d for d in state.deals if d.buyer_id == buyer_id]
            deal, _reason = buyer_accepts_ask(
                market=market,
                buyer=buyer,
                ask=ask,
                state=state,
                round_n=round_n,
                deals_for_buyer_so_far=deals_for_b,
                concentration_cap_pct=concentration_cap_pct,
            )
            if deal is not None:
                commit_deal(state, deal)
                new_deals.append(deal)
                break  # only one buyer can take this ASK (if qty exhausted)

    # ----- Process seller-side accepts (seller takes a BID) -----
    bids_to_seller_slots: dict[str, list[tuple[str, str]]] = {}  # bid_id -> [(seller_id, slot_id)]
    for seller_id, bid_id in seller_accepts:
        # Determine which of the seller's slots can serve this bid
        slot_ids = state.slot_ids_for_seller(seller_id)
        for slot_id in slot_ids:
            bids_to_seller_slots.setdefault(bid_id, []).append((seller_id, slot_id))

    for bid_id, candidates in bids_to_seller_slots.items():
        bid = _find_offer(bid_id, state.buyer_bids.values())
        if bid is None:
            continue
        if bid.from_id not in state.active_buyer_ids:
            continue
        candidates.sort(key=lambda t: t[1])  # FIFO by slot_id
        for seller_id, slot_id in candidates:
            slot = state.slot_by_id(slot_id)
            if slot is None or state.slot_remaining_qty.get(slot_id, 0) <= 0:
                continue
            buyer = next((b for b in market.buyers if b.id == bid.from_id), None)
            if buyer is None:
                continue
            deals_for_b = [d for d in state.deals if d.buyer_id == bid.from_id]
            deal, _reason = seller_accepts_bid(
                market=market,
                seller_id=seller_id,
                slot=slot,
                bid=bid,
                state=state,
                round_n=round_n,
                deals_for_buyer_so_far=deals_for_b,
                concentration_cap_pct=concentration_cap_pct,
            )
            if deal is not None:
                commit_deal(state, deal)
                new_deals.append(deal)
                break  # buyer's job is filled; no second deal for this bid

    return new_deals


def _find_offer(offer_id: str, offers) -> Optional[Offer]:
    for o in offers:
        if o.id == offer_id:
            return o
    return None


__all__ = [
    "BuyerAgent",
    "SellerAgent",
    "make_deterministic_agents",
    "make_llm_agents",
    "make_llm_agents_assigned",
    "agent_models_map",
    "run_rounds",
]
