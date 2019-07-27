 import torch
 import os
 
 def _save_checkpoint(epoch, loss, model, checkpoint_dir):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict()
        }

        filename = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, filename)