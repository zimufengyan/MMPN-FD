# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      trainer
   Author :         zmfy
   DateTime :       2024/5/18 19:37
   Description :    
-------------------------------------------------
"""
import torchsummary
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataloader.loader import init_dataloader
from networks.mmfd import MultiModalFD, UniModalFD
from utils import *


class TrainerBase:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    @staticmethod
    def evaluate(model, dataloader, n_way, k_shot, n_query, device, **kwargs):
        raise NotImplementedError

    def init_train(self):
        logging.info('starting load train data and data_loader...')
        dataloader = init_dataloader(self.config, mode='train')
        eval_dataloader = init_dataloader(self.config, mode='eval')

        logging.info(f'saving config into {self.config.trained_model_dir}')
        config = vars(self.config)
        with open(os.path.join(self.config.trained_model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, cls=MyJsonEncoder)

        return config, dataloader, eval_dataloader

    def init_optimizer(self, grouped_parameters, lr=None):
        lr = lr if lr is not None else self.config.initial_lr
        optimizer = Adam(grouped_parameters, lr=lr)
        if self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.num_train_epochs)
        elif self.config.scheduler == 'cosine-warm':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=self.config.min_lr)
        else:
            raise RuntimeError

        return optimizer, scheduler


def format_metrics(**kwargs):
    metrics = {k: f'{v:.4f}' for k, v in kwargs.items()}
    return metrics


def cal_accuracy(logits, labels):
    # computing acc
    logit = torch.softmax(logits, dim=-1)
    _, y_hat = logit.max(1)
    acc = y_hat.eq(labels).float().mean()

    return acc


class MMFDTrainer(TrainerBase):
    def __init__(self, config, modal='all', **kwargs):
        super(MMFDTrainer, self).__init__(config)
        self.modal = modal
        input_shape = kwargs['input_shape']
        if modal == 'all':
            self.model = MultiModalFD(config, input_shape, device=self.device)
        else:
            self.model = UniModalFD(config, input_shape, modal=modal)
        logging.info(torchsummary.summary(self.model))

        self.model.init_weights(config.init_type, config.init_gain)

        self.model.to(self.device)

    def load_model(self):
        state = torch.load(self.config.trained_model_dir)
        self.model.load_state_dict(state)

    def train(self):
        config, dataloader, eval_dataloader = self.init_train()

        state = self.model.state_dict()

        grouped_parameters = self.get_grouped_parameters(self.model)
        optimizer, scheduler = self.init_optimizer(grouped_parameters, lr=self.config.initial_lr)

        logging.info("====================== Running training ======================")
        logging.info(f"Num Batch Step: {len(dataloader)}, Num Epochs: {self.config.num_train_epochs}")

        best_accuracy = 0.0

        for epoch in range(int(self.config.num_train_epochs)):
            self.model.train()
            tr_loss, tr_acc = [], []
            with tqdm(total=len(dataloader), desc='Training') as pbar:
                for batch, batch_data in enumerate(dataloader):
                    x, labels = prepare_input(batch_data, self.device, self.config.n_way,
                                              self.config.n_query_tr, self.modal)

                    logit, loss = self.model(x, labels, self.config.k_shot_tr)

                    # computing acc
                    acc = cal_accuracy(logit, labels)

                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
                    optimizer.step()
                    optimizer.zero_grad()

                    tr_loss.append(loss.item())
                    tr_acc.append(acc.item())

                    # pbar.set_description(f'Train')
                    metrics = format_metrics(loss=loss.item(), acc=acc.item(), lr=scheduler.get_last_lr()[0])

                    pbar.set_postfix(metrics)
                    pbar.update(1)
            scheduler.step()

            tr_loss_avg, tr_acc_avg = np.mean(tr_loss), np.mean(tr_acc)

            if self.config.do_eval:

                eval_loss_avg, eval_acc_avg = self.evaluate(
                    self.model, eval_dataloader, self.config.n_way, self.config.k_shot_eval,
                    self.config.n_query_eval, self.device, modal=self.modal, desc="Eval"
                )
                info = (f'Train Epoch [{epoch + 1}/{self.config.num_train_epochs}] : '
                        f'Train Loss: {tr_loss_avg:.4f} Train Acc: {tr_acc_avg:.4f}  '
                        f'Eval Loss: {eval_loss_avg:.4f} Eval Acc: {eval_acc_avg:.4f}').replace('\n', '')
                if eval_acc_avg > best_accuracy:
                    # 保存最佳模型
                    logging.info(
                        f'save best model with accuracy {eval_acc_avg:.4f} into {self.config.trained_model_dir}')
                    best_state = self.model.state_dict()
                    torch.save(best_state, os.path.join(self.config.trained_model_dir, 'model_best.pt'))
                    best_accuracy = eval_acc_avg
            else:
                info = (f'Train Epoch [{epoch + 1}/{self.config.num_train_epochs}] : '
                        f'Train Loss: {tr_loss_avg:.4f} Train Acc: {tr_acc_avg:.4f}'
                        ).replace('\n', '')
            logging.info(info)

        # 保存最终模型和配置文件
        logging.info(f'Training Finished! Save model into {self.config.trained_model_dir}')
        torch.save(state, os.path.join(self.config.trained_model_dir, 'model_final.pt'))
        # wandb.finish()

    def test(self):
        logging.info('starting load test data and data_loader...')
        dataloader = init_dataloader(self.config, mode='test')

        loss, acc = self.evaluate(
            self.model, dataloader, self.config.n_way, self.config.k_shot_eval,
            self.config.n_query_eval, self.device, modal=self.modal, desc="Test"
        )
        logging.info(f'Test Loss: {loss:.4f} Test Acc: {acc:.4f}')

    @staticmethod
    def evaluate(model, dataloader, n_way, k_shot, n_query, device, modal='all', desc='Eval'):
        model.to(device)
        model.eval()
        eval_loss, eval_acc = [], []
        loop = tqdm(dataloader, total=len(dataloader))
        for batch, batch_data in enumerate(loop):
            x, labels = prepare_input(batch_data, device, n_way, n_query, modal)
            with torch.no_grad():
                logits, loss = model(x, labels=labels, k_shot=k_shot)

            # computing acc
            acc = cal_accuracy(logits, labels)

            eval_loss.append(loss.item())
            eval_acc.append(acc.item())
            loop.set_description(desc)
            loop.set_postfix(loss=loss.item(), acc=acc.item())

        return np.mean(eval_loss), np.mean(eval_acc)

    @staticmethod
    def get_grouped_parameters(model):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.00},
        ]
        return optimizer_grouped_parameters


def prepare_input(batch_data, device, n_way, query_shot, modal='all'):
    if modal == 'all':
        ts, freq, img = [data.float().to(device) for data in batch_data[:3]]
        x = (ts, freq, img)
    else:
        x = batch_data[0].float().to(device)

    # labels size: [n_way*n_query, ]  form: abcdabcd, i.e., 01230123
    labels = torch.tensor([i for _ in range(query_shot)
                           for i in range(n_way)]).long().to(device)

    return x, labels
