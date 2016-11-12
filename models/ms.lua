local nn = require 'nn'

local Convolution = nn.SpatialConvolution
local ReLU        = nn.ReLU
local Max         = nn.SpatialMaxPooling
local View        = nn.View
local Linear      = nn.Linear
local Reshape     = nn.Reshape

local model = nn.Sequential()

-- STAGE I
model:add(Convolution(3, 108, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))

-- STAGE II
branch = nn.Concat(2)

-- add left branch
left = nn.Sequential()
left:add(Convolution(108, 200, 5, 5))
left:add(ReLU())
left:add(Max(2,2,2,2))
left:add(Reshape(200*5*5))
branch:add(left)

-- add right branch
branch:add(Reshape(108*14*14))

-- add branching to model
model:add(branch)

-- CLASSIFIER
model:add(Linear(108*14*14 + 200*5*5, 100))
model:add(ReLU())
model:add(Linear(100, 43))

return model
