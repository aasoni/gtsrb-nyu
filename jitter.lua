-- Small set of functions that provide random jittering
-- of different kinds (rotation, shift, brightness) that
-- can be applied to input images used the 'jitter' function

function rotate(img)
    mask = torch.DoubleTensor(img:size()):fill(1.0)
    theta = torch.uniform(-0.2, 0.2)
    img_rot = image.rotate(img, theta, bilinear)
    mask = (image.rotate(mask, theta) - 1) * (-1)
    return torch.cmul(mask,img) + img_rot
end

function shift(img)
    mask = torch.DoubleTensor(img:size()):fill(1.0)
    x_shift = torch.round(torch.uniform(-2.5, 2.5))
    y_shift = torch.round(torch.uniform(-2.5, 2.5))
    img_shift = image.translate(img, x_shift, y_shift)
    mask = (image.translate(mask, x_shift, y_shift) - 1) * (-1)
    return torch.cmul(mask, img) + img_shift
end

function bright(img)
    val = torch.uniform(-0.1, 2.0)
    hsv = image.rgb2hsv(img)
    hsv[3] = hsv[3] + val
    return image.hsv2rgb(hsv)
end

function jitter(img)
    return tnt.transform.compose{
        [1] = bright,
        [2] = shift,
        [3] = rotate
    }(img)
end
