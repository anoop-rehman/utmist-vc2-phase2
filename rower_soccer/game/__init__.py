"""4-human LAN 2v2 creature-soccer game (WS4).

Modules:
  recording  demo file schema v1 (the BC dataset) -- writer + reader
  replay     deterministic replay: state-render + action-resimulate verification
  skills     the (skill, target) -> action layer; WS3's SkillController + a fallback
  match      the authoritative CPU dm_control sim, stepped at control rate
  lobby      4 claimable player slots + spectators, reconnect-safe
  server     Flask LAN server + browser client
  sim_client scripted HTTP clients, so the whole loop is testable without 4 humans
"""
