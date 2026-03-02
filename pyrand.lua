-- pyrand.lua for Lua 5.1
-- Python-compatible MT19937 random number generator
-- Matches CPython's random module: seed(), random(), gauss(),
--   getrandbits(), randbelow(), shuffle(), choices()

local M = {}

local N          = 624
local MT_M       = 397
local MATRIX_A   = 0x9908b0df
local UPPER_MASK = 0x80000000
local LOWER_MASK = 0x7fffffff
local TWO32      = 4294967296  -- 2^32

-- 32-bit unsigned integer truncation
local function uint32(n)
    return n % TWO32
end

-- 32-bit unsigned multiplication (low 32 bits)
local function mul32(a, b)
    -- a, b are 32-bit unsigned integers (0 <= a,b < 2^32)
    local a_lo = a % 65536
    local a_hi = math.floor(a / 65536)
    local b_lo = b % 65536
    local b_hi = math.floor(b / 65536)

    local low = a_lo * b_lo
    local mid = a_hi * b_lo + a_lo * b_hi
    local high = a_hi * b_hi

    local mid_low = mid % 65536
    local mid_high = math.floor(mid / 65536)

    local res = low + mid_low * 65536          -- 0 <= res < 2^33
    -- total = res + (mid_high + high) * 2^32
    -- take low 32 bits:
    return res % TWO32
end

-- Bitwise AND for 32-bit unsigned integers
local function band(a, b)
    a = a % TWO32
    b = b % TWO32
    local r = 0
    local p = 1
    while a > 0 and b > 0 do
        if a % 2 == 1 and b % 2 == 1 then
            r = r + p
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        p = p * 2
    end
    return r
end

-- Bitwise OR for 32-bit unsigned integers
local function bor(a, b)
    a = a % TWO32
    b = b % TWO32
    local r = 0
    local p = 1
    while a > 0 or b > 0 do
        if a % 2 == 1 or b % 2 == 1 then
            r = r + p
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        p = p * 2
    end
    return r
end

-- Bitwise XOR for 32-bit unsigned integers
local function bxor(a, b)
    a = a % TWO32
    b = b % TWO32
    local r = 0
    local p = 1
    while a > 0 or b > 0 do
        if (a % 2) ~= (b % 2) then
            r = r + p
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        p = p * 2
    end
    return r
end

-- Left shift for 32-bit unsigned integers (logical shift)
local function lshift(a, n)
    if n >= 32 then return 0 end
    a = a % TWO32
    local r = 0
    for i = 0, 31 - n do
        local bit = math.floor(a / 2^i) % 2
        if bit == 1 then
            r = r + 2^(i + n)
        end
    end
    return r
end

-- Right shift for 32-bit unsigned integers (logical shift)
local function rshift(a, n)
    if n >= 32 then return 0 end
    a = a % TWO32
    return math.floor(a / 2^n)
end

-- Bit length (number of bits needed to represent n)
local function bit_length(n)
    if n == 0 then return 0 end
    local b = 0
    local v = n
    while v > 0 do
        b = b + 1
        v = math.floor(v / 2)
    end
    return b
end

local RNG = {}
RNG.__index = RNG

function RNG:seed(s)
    local mt = {}
    mt[1] = uint32(19650218)
    for i = 2, N do
        local prev = mt[i - 1]
        local val = uint32(bxor(prev, rshift(prev, 30)))
        mt[i] = uint32(mul32(1812433253, val) + (i - 1))
    end
    local key  = { s }
    local klen = #key
    local i    = 2
    local j    = 1
    for _ = 1, math.max(N, klen) do
        local prev = mt[i-1]
        local val = uint32(bxor(prev, rshift(prev, 30)))
        local t = mul32(val, 1664525)
        mt[i] = uint32(bxor(mt[i], t) + key[j] + (j - 1))
        i = i + 1;  j = j + 1
        if i > N then mt[1] = mt[N]; i = 2 end
        if j > klen then j = 1 end
    end
    for _ = 1, N - 1 do
        local prev = mt[i-1]
        local val = uint32(bxor(prev, rshift(prev, 30)))
        local t = mul32(val, 1566083941)
        mt[i] = uint32(bxor(mt[i], t) - (i - 1))
        i = i + 1
        if i > N then mt[1] = mt[N]; i = 2 end
    end
    mt[1] = 0x80000000
    self._mt         = mt
    self._index      = N + 1
    self._gauss_next = nil
end

function RNG:_generate()
    local mt    = self._mt
    local mag01 = { [0] = 0, [1] = MATRIX_A }
    for i = 1, N - MT_M do
        local y = uint32(bor(band(mt[i], UPPER_MASK), band(mt[i + 1], LOWER_MASK)))
        mt[i] = bxor(bxor(mt[i + MT_M], rshift(y, 1)), mag01[band(y, 1)])
    end
    for i = N - MT_M + 1, N - 1 do
        local y = uint32(bor(band(mt[i], UPPER_MASK), band(mt[i + 1], LOWER_MASK)))
        mt[i] = bxor(bxor(mt[i + MT_M - N], rshift(y, 1)), mag01[band(y, 1)])
    end
    local y = uint32(bor(band(mt[N], UPPER_MASK), band(mt[1], LOWER_MASK)))
    mt[N] = bxor(bxor(mt[MT_M], rshift(y, 1)), mag01[band(y, 1)])
    self._index = 1
end

function RNG:_genrand_uint32()
    if self._index > N then
        self:_generate()
    end
    local y = self._mt[self._index]
    self._index = self._index + 1
    y = bxor(y, rshift(y, 11))
    y = uint32(bxor(y, band(lshift(y, 7), 0x9d2c5680)))
    y = uint32(bxor(y, band(lshift(y, 15), 0xefc60000)))
    y = bxor(y, rshift(y, 18))
    return uint32(y)
end

function RNG:random()
    local a = rshift(self:_genrand_uint32(), 5)
    local b = rshift(self:_genrand_uint32(), 6)
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
end

function RNG:gauss(mu, sigma)
    local z = self._gauss_next
    self._gauss_next = nil
    if z == nil then
        local x2pi = self:random() * (2.0 * math.pi)
        local g2rad = math.sqrt(-2.0 * math.log(1.0 - self:random()))
        z = math.cos(x2pi) * g2rad
        self._gauss_next = math.sin(x2pi) * g2rad
    end
    return mu + z * sigma
end

function RNG:getrandbits(k)
    assert(k >= 1 and k <= 32, "getrandbits: k must be 1..32")
    return rshift(self:_genrand_uint32(), 32 - k)
end

function RNG:randbelow(n)
    assert(n > 0, "randbelow: n must be > 0")
    local k = bit_length(n)
    while true do
        local r = self:getrandbits(k)
        if r < n then return r end
    end
end

function RNG:shuffle(lst)
    local n = #lst
    for i = n, 2, -1 do
        local j = self:randbelow(i) + 1
        lst[i], lst[j] = lst[j], lst[i]
    end
end

function RNG:choices(population, weights, k)
    k = k or 1
    local n = #population
    assert(#weights == n, "choices: weights and population must be same length")

    local cum = {}
    local total = 0.0
    for i = 1, n do
        total = total + weights[i]
        cum[i] = total
    end
    assert(total > 0, "choices: total weight must be > 0")

    local function bisect_right(val)
        local lo, hi = 1, n
        while lo <= hi do
            local mid = lo + math.floor((hi - lo) / 2)
            if cum[mid] <= val then lo = mid + 1 else hi = mid - 1 end
        end
        return lo > n and n or lo
    end

    local result = {}
    for i = 1, k do
        result[i] = population[bisect_right(self:random() * total)]
    end
    return result
end

function M.new()
    return setmetatable({
        _mt         = {},
        _index      = N + 1,
        _gauss_next = nil,
    }, RNG)
end

return M