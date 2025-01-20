import torch
from torch.utils.data import DataLoader
from dataloader import GolfDB
from model import EventDetector
from util import AverageMeter
import numpy as np
import os
from pce_eval import pce_eval
from generate_splits import split
from aim import Run
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(seq_length, overlap_size, fps, iterations, lr, bs):
    seed = 42  
    set_seed(seed) 

    run = Run()
    src_files = f'{fps}fps'
    src_dir = f'src_data/{src_files}_train'
    csv_dir = f'splits/{src_files}_seq{seq_length}_over{overlap_size}'

    # Check if csv_dir exists
    if not os.path.exists(csv_dir):
        print("Splits don't exist. Generating splits!")
        os.makedirs(csv_dir)
        split(src_dir, csv_dir, seq_length, overlap_size)
        print("Splits generated!")
      
    it_save = iterations // 10  # Save model every 1/10 iterations
    n_cpu = 6

    run["hparams"] = {
        "seq_length": seq_length,
        "overlap_size": overlap_size,
        "fps": fps,
        "lr": lr,
        "bs": bs,
        "iterations": iterations,
    }

    # Create the model
    hidden_dim = 256
    num_classes = 9
    model = EventDetector(hidden_dim=hidden_dim, num_classes=num_classes, dropout=True)
    model.train()
    model.cuda()

    # Create the dataset and data loader
    dataset = GolfDB(csv_dir=csv_dir)
    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # Define the loss function and optimizer
    weights = torch.FloatTensor([1/8]*8 + [1/130]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for batch, sample in enumerate(data_loader):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)  # logits shape: [batch_size, seq_length, num_classes]

            # Flatten logits and labels for computing loss
            logits_flat = logits.view(-1, num_classes)  # shape: [batch_size * seq_length, num_classes]
            labels_flat = labels.view(-1)  # shape: [batch_size * seq_length]

            loss = criterion(logits_flat, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            run.track(losses.val, name='loss', step=i, context={"subset": "train"})
            run.track(losses.avg, name='loss_avg', step=i, context={"subset": "train"})

            i += 1
            if i % it_save == 0:
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            if i == iterations:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()},
                           f'models/{src_files}_seq{seq_length}_over{overlap_size}.pth.tar')
                break

    # Evaluation
    test_csv_dir = f'src_data/{src_files}_test'
    model.eval()
    PCE, deltas, correct, preds = pce_eval(model, test_csv_dir, seq_length, n_cpu, False)
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
    for cl, val in enumerate(correct_means):
        run.track(val, name=f'acc_{cl}', step=i, context={'subset': 'eval'})
    run.track(PCE, name='PCE', step=i, context={ "subset":"eval" })
    run.track(unsorted_cnt, name='unsorted', step=i, context={ "subset":"eval" })

    print('Average PCE: {}'.format(PCE))
    return PCE






if __name__ == '__main__':
     

    overlap_size = 0
    iterations = 10000
    lr = 0.001
    bs = 128

    fps = 30
    seq_length = 30

    print(f"Params: fps: {fps}, seq_length: {seq_length}, overlap_size: {overlap_size}")
    PCE = train(seq_length, overlap_size, fps, iterations, lr, bs)