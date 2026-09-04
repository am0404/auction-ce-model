# tests/fixtures

**Everything in this directory is fabricated.**

`real_player_input_v1_example.json` is a hand-written example of
`schemas/real_player_input_v1.schema.json`. Every name, number, hash and label
in it is invented for testing the contract's *shape*. It contains no real
player, no vendor value, no real content hash and no real league configuration.

It is deliberately not a sample of any real dataset. This repository is public,
and the real inputs it describes are subscriber-gated vendor data that is not
redistributable. The fixture exists so the schema can be tested without either.

Two players, chosen to exercise opposite paths:

* **Fabricated Alpha** — a full stat line, an availability profile and both
  expert labels present.
* **Fabricated Beta** — deliberately sparse: null team, null bye, most stat
  fields null, no availability profile and no expert labels. This is the case
  that matters, because a null in this contract means **not projected**, which
  is not the same as zero.

Both carry `central_tendency: "unknown"`, `games_assumed: null` and
`fumbles_are_lost_fumbles: null` — the honest values for the real source today,
and the ones a consumer must refuse to guess past.
