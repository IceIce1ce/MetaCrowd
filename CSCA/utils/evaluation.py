import cv2

def eval_game(output, target, L=0): # [1, 1, 60, 80], [1, 480, 640]
    output = output[0][0].cpu().detach().numpy() # [60, 80]
    target = target[0] # [480, 640]
    H, W = target.shape # 480, 640
    ratio = H / output.shape[0] # 8
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio) # [480, 640]
    assert output.shape == target.shape
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i * H//p: (i + 1) * H//p, j * W//p: (j + 1) * W//p]
            target_block = target[i * H//p: (i + 1) * H//p, j * W//p: (j + 1) * W//p]
            abs_error += abs(output_block.sum() - target_block.sum().float())
            square_error += (output_block.sum() - target_block.sum().float()).pow(2)
    return abs_error, square_error

def eval_relative(output, target): # [1, 1, 60, 80], [1, 480, 640]
    output_num = output.cpu().data.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num - target_num) / target_num
    return relative_error