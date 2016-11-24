require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require 'jitter' -- small library to add jitter to images

local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 32, 32
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local percentJitter = 0.8
local testMode = false

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(opt.nThreads)
torch.manualSeed(opt.manualSeed)

local x1, y1, x2, y2
function crop(img)
    return image.crop(img, x1, y1, x2, y2)
end

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

function transformInput(inp, addJitter)
    addJitter = addJitter and torch.uniform(0,1) > percentJitter
    f = tnt.transform.compose{
        [1] = crop,
        [2] = resize,
        [3] = addJitter and jitter or function (img) return img end
    }
    return f(inp)
end

function sampleUniform(epoch)
    -- Determine if we should sample uniformly
    -- across class ids for this current sample
    local y = torch.uniform(0,1)
    local e = opt.nEpochs - 5
    return y < (-1.0/e)*epoch + 1
end

function findSampleWithLabel(dataset, idx, label)
    -- Search forward from the given index 'idx' and
    -- return the first sample with the provided label
    local i = idx
    local len = dataset:size(1)
    while true do
        ret = dataset[(i%len)+1]
        if ret[9] == label then
            return ret
        end
        i = i + 1
    end
end

local epoch = 1
local label = 0
function getTrainSample(dataset, idx)
    -- get train sample at index 'idex' from dataset.
    -- NOTE: special sampling occurs when sampleUniform
    -- returns true.
    local r
    if sampleUniform(epoch) then
        r = findSampleWithLabel(dataset, idx, label)
        label = (label + 1) % 43
    else
        r = dataset[idx]
    end
    classId, track, file = r[9], r[1], r[2]
    x1, y1, x2, y2 = r[5], r[6], r[7], r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    seed = idx
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file), true)
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    x1, y1, x2, y2 = r[4], r[5], r[6], r[7]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file), false)
end

function getTestLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][8]+1}
end

function getIterator(dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',

    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        },
        replacement = true
    },
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]},
            target = getTestLabel(testData, idx)
        }
    end
}

local model = require("models/".. opt.model)
-- model:cuda()
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
-- print(model)
engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    elseif testMode then
        mode = 'Test'
    else
        mode = 'Val'
    end
end

--local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
--engine.hooks.onSample = function(state)
--    igpu:resize(state.sample.input:size() ):copy(state.sample.input)
--    tgpu:resize(state.sample.target:size()):copy(state.sample.target)
--    state.sample.input  = igpu
--    state.sample.target = tgpu
-- end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == 'true' then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
              mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
                        mode, meter:value(), clerr:value{k = 1}, timer:value()))
end


while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }

    testMode = true
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(testDataset)
    }
    testMode = false

    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

--  Serialize and write model to file
torch.save('ala458_gtsrb', model:clearState())

print("The End!")
