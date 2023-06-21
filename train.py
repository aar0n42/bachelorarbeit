import torch
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Models import ssformer
from Data import datacfg
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torchvision import transforms
from torch import nn
import torch.distributed as dist
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "17356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.barrier()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.loss =  nn.BCEWithLogitsLoss()
        self.loss2 = smp.losses.DiceLoss('binary')
 
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = self.model(source)
            loss = self.loss(output, targets) + self.loss2(output, targets)
        print("loss:  " + str(loss.item()),flush=True)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print('test',flush=True)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}",flush=True)
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "ssformer" + str(epoch) + ".pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}", flush=True)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            print(epoch, flush=True)
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(688)


def load_train_objs():
    transform = transforms.Compose([transforms.ToTensor(),transforms.RandomCrop((480,640)),transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.25, hue=0.01),])
    transform2 = transforms.Compose([transforms.ToTensor(),])

    train_set = datacfg.UlcerData('DFUC2022_/train/images', 'DFUC2022_/train/labels',transform, transform2)
    model = ssformer.mit_PLD_b4() #fcbformer.FCBFormer() hardmseg.HarDMSEG
    
    #model.load_state_dict(msd)
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=0.0001),])
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=4
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    print(rank, flush=True)
    print(world_size, flush=True)
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    start_time = time.time()
    trainer.train(total_epochs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time),flush=True)
    destroy_process_group()


if __name__ == "__main__":
    total_epochs = 300
    save_every = 20
    batch_size = 4
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, total_epochs,batch_size), nprocs=world_size)
