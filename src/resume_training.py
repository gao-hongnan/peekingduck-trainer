
"""
https://www.kaggle.com/c/seti-breakthrough-listen/discussion/247574
You have to save your optimizer parameters as you save your model weights when you checkpoint.

Then when you resume you load them.

For instance here are functions I use to save and restore checkpoints with pytorch. Of course you need to adjust it to your code.
It is important that you create your model, optimizer, etc in the load checkpoint function exactly as you create them in your training loop.
The scaler is only useful if you use mixed precision (torch.cuda.amp).
"""
import torch


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, fold, seed, fname):
    """Save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'seed': seed,
    }
    torch.save(checkpoint, '../checkpoints/%s/%s_%d_%d.pt' %
               (fname, fname, fold, seed))


def load_checkpoint(fold, seed, fname):
    """The sequence below is important, if you load state dict then initialize you will reset.
    Initilize optimizer
    Initilize scheduler
    Load state_dict of optimizer
    Load state_dict of scheduler
    """
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    pct_start=PCT_START,
                                                    div_factor=DIV_FACTOR
                                                    max_lr=MAX_LR,
                                                    epochs=EPOCHS,
                                                    steps_per_epoch=int(np.ceil(len(train_data_loader)/GRADIENT_ACCUMULATION)))
    scaler = GradScaler()
    checkpoint = torch.load('../checkpoints/%s/%s_%d_%d.pt' %
                            (fname, fname, fold, seed))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scheduler, scaler, epoch
