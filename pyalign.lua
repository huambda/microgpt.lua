-- microgpt_random_aligned.lua
-- Replaces the default RNG with pyrand (MT19937, Python-compatible),
-- then runs the identical algorithm from microgpt.lua.
-- Result: loss and samples match CPython microgpt.py exactly.

local pyrand = require("pyrand")
local rng = pyrand.new()
require("microgpt").run(rng)
