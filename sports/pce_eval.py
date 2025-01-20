from model import EventDetector
import torch
from torch.utils.data import DataLoader
from dataloader import GolfDB
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import random

import torch.nn.functional as F
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pce_eval(model, csv_dir, seq_length, n_cpu, disp):
    dataset = GolfDB(csv_dir=csv_dir)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    deltas = []
    preds = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        batch = 0
        probs_list = []
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :]
                # Pad the last batch with last frame
                pad = image_batch[:, -1:, :, :].repeat(1, (batch + 1) * seq_length - images.shape[1], 1, 1)
                image_batch = torch.cat((image_batch, pad), dim=1)
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :]
            logits = model(image_batch.cuda())
            # logits shape: [1, seq_length, num_classes]

            # Apply softmax over num_classes dimension (dim=2)
            probs = F.softmax(logits.data, dim=2).cpu().numpy()  # Shape: [1, seq_length, num_classes]

            # Collect probabilities
            probs_list.append(probs)
            batch += 1

        # Concatenate probabilities over the time dimension
        probs = np.concatenate(probs_list, axis=1)  # Shape: [1, total_length, num_classes]

        # Remove the batch dimension
        probs = probs.squeeze(0)  # Shape: [total_length, num_classes]

        # Convert labels to numpy array and remove batch dimension
        labels = labels.cpu().numpy().squeeze(0)  # Shape: [total_length]

        # Call correct_preds with probs and labels
        _, pred, delta, _, c = correct_preds(probs, labels)
        if disp:
            print(f"Sample {i}: Correct predictions = {c}")
        correct.append(c)
        deltas.append(delta)
        preds.append(pred)

    PCE = np.mean(correct)
    return PCE, deltas, correct, preds


if __name__ == '__main__':
    seed = 42  
    set_seed(seed) 

    split = 1
    seq_length = 30
    n_cpu = 6
    fps = 30


    src_files = f'{fps}fps'
    csv_dir = f'src_data/{src_files}_test'


    # Create the model
    hidden_dim = 256
    num_classes = 9
    model = EventDetector(hidden_dim=hidden_dim, num_classes=num_classes)

    save_dict = torch.load('models/30fps_seq30_over0.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    print('model loaded')
    PCE, deltas, correct, preds = pce_eval(model, csv_dir, seq_length, n_cpu, False)
    print('Average PCE: {}'.format(PCE))
    correct_means = np.round(np.mean(correct, axis=0), 2)
    deltas_means = np.round(np.mean(deltas, axis=0), 2)
    correct_median = np.median(correct, axis=0)
    deltas_median = np.median(deltas, axis=0)
    unsorted_cnt = 0
    sorted_cnt = 0
    for pred in preds:
        eq = (np.sort(pred) == pred).all()
        if eq == False:
            unsorted_cnt += 1
        else:
            sorted_cnt += 1
    print(f'Correct means: {correct_means}')
    print(f'Correct median: {correct_median}')
    print(f'Deltas means: {deltas_means}')
    print(f'Deltas median: {deltas_median}')
    print(f'Unsorted cnt: {unsorted_cnt}')
    print(f'Sorted cnt: {sorted_cnt}')