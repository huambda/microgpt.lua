--[[
The most atomic way to train and run inference for a GPT in pure, dependency-free Lua.
This file is the complete algorithm.
Everything else is just efficiency.

Original work (Python): Andrej Karpathy
Lua implementation: huambda
--]]

local math = require("math")

-- 1. Let there be Autograd to recursively apply the chain rule through a computation graph
local V = {}; V.__index = V

local function val(d, ch, lg)
  return setmetatable({
         data=d,         -- scalar value of this node calculated during forward pass
         grad=0,         -- derivative of the loss w.r.t. this node, calculated in backward pass
         _ch=ch or {},   -- children of this node in the computation graph
         _lg=lg or {}    -- local derivative of this node w.r.t. its children
  }, V)
end

function V:__add(o)
  o = type(o)=="number" and val(o) or o
  return val(self.data+o.data, {self,o}, {1,1})
end

function V:__mul(o)
  o = type(o)=="number" and val(o) or o
  return val(self.data*o.data, {self,o}, {o.data,self.data})
end

function V:__pow(e) return val(self.data^e, {self}, {e*self.data^(e-1)}) end
function V:log()    return val(math.log(self.data), {self}, {1/self.data}) end
function V:exp()    return val(math.exp(self.data), {self}, {math.exp(self.data)}) end
function V:relu()   return val(math.max(0,self.data), {self}, {self.data > 0 and 1 or 0}) end
function V:__unm()  return self * -1 end
function V:__sub(o) return self + (-o) end
function V:__div(o) return self * o^-1 end

function V:backward()
  local topo, visited = {}, {}
  local function build(v)
    if not visited[v] then
      visited[v] = true
      for _, c in ipairs(v._ch) do build(c) end
      topo[#topo+1] = v
    end
  end
  build(self); self.grad = 1
  for i = #topo, 1, -1 do
    for j, c in ipairs(topo[i]._ch) do
      c.grad = c.grad + topo[i]._lg[j] * topo[i].grad
    end
  end
end

-- 2. Initialize the parameters, to store the knowledge of the model
local n_layer=1;    -- depth of the transformer neural network (number of layers)
local n_embd=16;    -- width of the network (embedding dimension) 
local block_size=16 -- maximum context length of the attention window (note: the longest name is 15 characters)
local n_head=4;     -- number of attention heads
local head_dim = math.floor(n_embd / n_head) -- derived dimension of each head
local state_dict,params  = {},{}

local function init_parameters(vocab_size, rng)
  local function matrix(nout, nin, std)
    std = std or 0.08; local m = {}
    for i=1,nout do m[i]={}
      for j=1,nin do m[i][j]=val(rng:gauss(0,std)) end
    end
    return m
  end

  state_dict.wte=matrix(vocab_size,n_embd)
  state_dict.wpe=matrix(block_size,n_embd)
  state_dict.lm_head=matrix(vocab_size,n_embd)

  for li = 0, n_layer-1 do
    state_dict["layer"..li..".".."attn_wq"]=matrix(n_embd,n_embd)
    state_dict["layer"..li..".".."attn_wk"]=matrix(n_embd,n_embd)
    state_dict["layer"..li..".".."attn_wv"]=matrix(n_embd,n_embd)
    state_dict["layer"..li..".".."attn_wo"]=matrix(n_embd,n_embd)
    state_dict["layer"..li..".".."mlp_fc1"]=matrix(4*n_embd,n_embd)
    state_dict["layer"..li..".".."mlp_fc2"]=matrix(n_embd,4*n_embd)
  end
  for _, mat in pairs(state_dict) do
    for _, row in ipairs(mat) do
      for _, p in ipairs(row) do params[#params+1]=p end
    end
  end
  print(("num params: %d"):format(#params))
end

-- 3. Define the model architecture: a function mapping tokens and parameters to logits over what comes next
-- Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
local Z = val(0)

local function linear(x, w)
  local o = {}
  for i=1,#w do
    local s=Z; for j=1,#w[i] do s=s+w[i][j]*x[j] end; o[i]=s
  end
  return o
end

local function softmax(logits)
  local mx = -math.huge
  for _, v in ipairs(logits) do if v.data>mx then mx=v.data end end
  local exps, tot = {}, Z
  for i,v in ipairs(logits) do
    exps[i]=(v-mx):exp(); tot=tot+exps[i]
  end
  local o={}; for i=1,#exps do o[i]=exps[i]/tot end; return o
end

local function rmsnorm(x)
  local ms=Z; for _,xi in ipairs(x) do ms=ms+xi*xi end
  local sc=(ms*(1/#x)+1e-5)^(-0.5)
  local o={}; for i,xi in ipairs(x) do o[i]=xi*sc end; return o
end

local function gpt(tok, pos, keys, vals)
  local x = {}
  for i = 1,n_embd do 
    x[i] = state_dict.wte[tok+1][i] + state_dict.wpe[pos+1][i] -- joint token and position embedding
  end
  x = rmsnorm(x) -- note: not redundant due to backward pass via the residual connection

  for li=0, n_layer-1 do
    -- 1) Multi-head Attention block
    local x_residual=x; x=rmsnorm(x)
    local q=linear(x,state_dict["layer"..li..".".."attn_wq"])
    local k=linear(x,state_dict["layer"..li..".".."attn_wk"])
    local v=linear(x,state_dict["layer"..li..".".."attn_wv"])
    keys[li+1][#keys[li+1]+1]=k; vals[li+1][#vals[li+1]+1]=v
    local xa={}
    for h=0, n_head-1 do
      local hs=h*head_dim; local qh,kh,vh={},{},{}
      for j=1,head_dim do qh[j]=q[hs+j] end
      for t=1,#keys[li+1] do
        kh[t]={}; vh[t]={}
        for j=1,head_dim do
          kh[t][j]=keys[li+1][t][hs+j]
          vh[t][j]=vals[li+1][t][hs+j]
        end
      end
      local al={}
      for t=1,#kh do
        local s=Z
        for j=1,head_dim do s=s+qh[j]*kh[t][j] end
        al[t]=s*head_dim^(-0.5)
      end
      local aw=softmax(al)
      for j=1,head_dim do
        local s=Z
        for t=1,#vh do s=s+aw[t]*vh[t][j] end
        xa[hs+j]=s
      end
    end
    x=linear(xa,state_dict["layer"..li..".".."attn_wo"])
    for i=1,n_embd do x[i]=x[i]+x_residual[i] end
    -- 2) MLP block
    x_residual=x; x=rmsnorm(x); x=linear(x,state_dict["layer"..li..".".."mlp_fc1"])
    for i=1,#x do x[i]=x[i]:relu() end
    x=linear(x,state_dict["layer"..li..".".."mlp_fc2"])
    for i=1,n_embd do x[i]=x[i]+x_residual[i] end
  end
  return linear(x, state_dict.lm_head)
end

-- 4. Let there be a default RNG using Lua's math.random
local _gnext = nil
local default_rng = {
  seed    = function(_, s) math.randomseed(s) end,
  gauss   = function(_, mu, sigma)
    if _gnext then
      local z = _gnext; _gnext = nil; return mu + z * sigma
    end
    local a = math.random() * 2 * math.pi
    local r = math.sqrt(-2 * math.log(1 - math.random()))
    _gnext = math.sin(a) * r
    return mu + math.cos(a) * r * sigma
  end,
  shuffle = function(_, t)
    for i = #t, 2, -1 do local j = math.random(i); t[i], t[j] = t[j], t[i] end
  end,
  choices = function(_, pop, w)
    local tot, cum = 0, 0
    for _, x in ipairs(w) do tot = tot + x end
    local r = math.random() * tot
    for i, x in ipairs(w) do cum = cum + x; if cum > r then return {pop[i]} end end
    return {pop[#pop]}
  end,
}

-- 5. RUN
local function run(rng)
  rng:seed(42) -- Let there be order among chaos

  -- Let there be a Dataset `docs`: list[str] of documents
  local docs = {}
  for line in io.lines("input.txt") do
    line = line:match("^%s*(.-)%s*$")
    if #line > 0 then docs[#docs+1] = line end
  end
  rng:shuffle(docs)
  print(("num docs: %d"):format(#docs))

  -- Let there be a Tokenizer
  local uchars, seen = {}, {}
  for _, doc in ipairs(docs) do
    for c in doc:gmatch(".") do
      if not seen[c] then seen[c]=true; uchars[#uchars+1]=c end
    end
  end
  table.sort(uchars)
  local BOS = #uchars
  local vocab_size = #uchars + 1
  print(("vocab size: %d"):format(vocab_size))
  local stoi = {}
  for i, c in ipairs(uchars) do stoi[c] = i - 1 end

  init_parameters(vocab_size, rng)

  -- Let there be Adam, the blessed optimizer and its buffers
  local learning_rate, b1, b2, eps = 0.01,0.85,0.99,1e-8
  local m = {}  -- first moment buffer
  local v = {}  -- second moment buffer
  for i=1,#params do m[i]=0.0; v[i]=0.0 end

  -- Repeat in sequence
  local num_steps = 1000 -- number of training steps
  for step=0, num_steps-1 do
    -- Take single document, tokenize it, surround it with BOS special token on both sides
    local doc = docs[(step % #docs)+1]
    local tokens = {BOS}
    for c in doc:gmatch(".") do tokens[#tokens+1]=stoi[c] end
    tokens[#tokens+1]=BOS
    local n=math.min(block_size, #tokens-1)

    -- Forward the token sequence through the model, building up the computation graph all the way to the loss
    local keys,vals={},{}
    for i=1,n_layer do keys[i]={}; vals[i]={} end
    local losses={}
    for pos=0,n-1 do
      local probs=softmax(gpt(tokens[pos+1],pos,keys,vals))
      losses[#losses+1]=-(probs[tokens[pos+2]+1]:log())
    end
    local loss=Z
    for _,l in ipairs(losses) do loss=loss+l end
    loss = loss * (1/n) -- final average loss over the document sequence. May yours be low.

    -- Backward the loss, calculating the gradients with respect to all model parameters
    loss:backward()

    -- Adam optimizer update: update the model parameters based on the corresponding gradients
    local lrt = learning_rate * (1 - step / num_steps)
    for i,p in ipairs(params) do
      m[i] = b1 * m[i] + (1 - b1) * p.grad
      v[i] = b2 * v[i] + (1 - b2) * p.grad^2
      p.data = p.data - lrt * (m[i] / (1 - b1^(step + 1)))
                           / ((v[i] / (1 - b2^(step + 1)))^0.5 + eps)
      p.grad=0
    end

    local line = ("step %4d / %4d | loss %.4f"):format(
      step+1, num_steps, loss.data)
    if step < 10 then print(line)
    else io.write(line.."\r"); io.flush() end
  end

  -- Inference: may the model babble back to us
  local temperature = 0.5 -- in (0, 1], control the "creativity" of generated text, low to high
  print("\n--- inference (new, hallucinated names) ---")
  local pop={}; for i=0,vocab_size-1 do pop[i+1]=i end
  for si=1,20 do
    local keys,vals={},{}
    for i=1,n_layer do keys[i]={}; vals[i]={} end
    local tok,sample=BOS,{}
    for pos=0,block_size-1 do
      local logits=gpt(tok,pos,keys,vals)
      local scaled={}
      for i,l in ipairs(logits) do scaled[i]=l*(1/temperature) end
      local w={}
      for _,p in ipairs(softmax(scaled)) do w[#w+1]=p.data end
      tok=rng:choices(pop,w)[1]
      if tok==BOS then break end
      sample[#sample+1]=uchars[tok+1]
    end
    print(("sample %2d: %s"):format(si, table.concat(sample)))
  end
end

if arg and arg[0] == "microgpt.lua" then
    run(default_rng)
end

return { run = run }
