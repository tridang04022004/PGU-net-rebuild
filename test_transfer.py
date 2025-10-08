import torch
from unet import ProgressiveUNet

def run_transfer_test():
    model = ProgressiveUNet(in_channels=3, num_classes=1)
    # Create prev/cur state dicts (stage1 -> stage2)
    prev = model.stage1.state_dict()
    cur = model.stage2.state_dict()
    new = model.transfer_weights(prev, cur, stage=2)
    # Load with non-strict to allow unmatched keys
    model.stage2.load_state_dict(new, strict=False)
    print('Loaded stage2 weights (partial) from stage1.');

    # Run forward for all stages to validate shapes
    for s in [1,2,3,4]:
        model.set_stage(s)
        res = model.get_current_resolution()
        x = torch.randn(1, 3, res, res)
        y = model(x)
        print(f'stage {s} -> input {res}x{res}, output shape: {tuple(y.shape)}')

if __name__ == '__main__':
    run_transfer_test()
