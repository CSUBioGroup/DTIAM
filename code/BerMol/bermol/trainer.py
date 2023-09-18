import os
import pickle
import dill as pickle
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from bermol.model import BerMol
from bermol.dataloading import mol_dataloader
from bermol.utils import TASK_DICT, get_linear_schedule_with_warmup
from bermol.tokenizer import BerMolTokenizer


class BerMolPreTrainer:
    def __init__(self, task_list, vocab, config):
        
        task_modules = nn.ModuleList([TASK_DICT[name](name, config) for name in task_list])
        if config.from_pretrained:
            self.model = torch.load(config.from_pretrained)
        else:
            self.model = BerMol(task_modules, config)
        self.vocab = vocab
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.set_path(config.bermol_path)
        self.config = config


    def train(self, train_corpus_path, val_corpus_path=None, on_memory=True):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train_iter = mol_dataloader(train_corpus_path, self.vocab, batch_size=self.config.batch_size, shuffle=True, on_memory=on_memory)
        self.scheduler = self._initialise_lr_scheduler(self.optimizer, len(train_iter))
        if val_corpus_path:
            val_iter = mol_dataloader(val_corpus_path, self.vocab, batch_size=self.config.batch_size, shuffle=False, on_memory=on_memory)
        
        min_loss = float("inf")
        for epoch in range(self.config.epochs):
            print("Epoch %s" % (epoch+1))

            train_loss = self.iteration(train_iter, train=True)
            print("Train loss %s" % (train_loss))

            if val_corpus_path:
                val_loss = self.iteration(val_iter, train=False)
                print("Val loss %s" % (val_loss))

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.save_model()
            
            else:
                self.save_model()
            
            self.save(epoch=epoch, silent=True)
        
        self.save()
        return self
    
    def iteration(self, data_iter, train=True):
        total_loss = 0
        data_bar = tqdm(data_iter) if train else data_iter
        for batch in data_bar:
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            all_task_predictions = self.model(token_ids, attention_mask)
            loss, all_loss = None, []
            for task_module in self.model.task_modules:
                task_name = task_module.name
                predictions = all_task_predictions[task_name]
                labels = batch[task_name].to(self.device)
                task_loss = task_module.compute_loss(predictions, labels) * task_module.weight
                loss = task_loss if loss is None else loss+task_loss
                all_loss.append(round(float(task_loss.data), 6))
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                data_bar.set_description("Training")
                data_bar.set_postfix(batch_loss=float(loss.data), task_loss=all_loss)
            
            total_loss += float(loss.data)*len(batch)
        
        return total_loss/len(data_iter)
    
    def _initialise_lr_scheduler(self, optimizer, num_batches):
        num_training_steps = num_batches // self.config.accumulate_grad_batches * self.config.epochs
        warmup_steps = int(num_training_steps * self.config.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        return scheduler
    
    def transform(self, smiles, device='cpu'):
        tokenizer = BerMolTokenizer(self.vocab)
        token_ids = tokenizer.encode(smiles)
        sequence_output, pooled_output = self.model.encoder(token_ids.to(device))
        return sequence_output, pooled_output

    def set_path(self, path):
        if path is None:
            import datetime
            cur_time = datetime.datetime.now()
            timestamp = datetime.datetime.strftime(cur_time,'%Y%m%d_%H%M%S')
            name = 'BerMol-' + timestamp
            self.path = './checkpoints/' + name + '/'
        else:
            self.path = path
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def save_model(self):
        torch.save(self.model, self.path + 'best_model.pth')

    def load_model(self, path):
        return torch.load(path + 'best_model.pth')

    def save(self, epoch=None, silent=False):
        file_name = 'BerMolModel_epoch' + str(epoch) + '.pkl' if epoch is not None else 'BerMolModel.pkl'
        with open(self.path + file_name, 'wb') as f:
            pickle.dump(self, f)
        if not silent:
            print(f'BerMolPreTrainer saved. To load, use: predictor = BerMolPreTrainer.load("{self.path}")')

    @staticmethod
    def load(path, name=None) -> 'BerMolPreTrainer':
        file_name = name if name is not None else 'BerMolModel.pkl'
        with open(path + file_name, 'rb') as f:
            predictor = pickle.load(f)
            predictor.model = predictor.load_model(path)
        return predictor


def train():

    import argparse
    from bermol.vocab import MolVocab

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_corpus_path", type=str, required=True, help="output molecular corpus path of training set")
    parser.add_argument("--val_corpus_path", type=str, default=None, help="output molecular corpus path of validation set")
    parser.add_argument("--vocab_path", type=str, required=True, help="output molecular vocab path")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading the corpus on memory: true or false")
    parser.add_argument("--bermol_path", type=str, default=None, help="path for saving the model")
    parser.add_argument('--from_pretrained', type=str, default=None, help='Directory containing config and pretrained model weights')
    parser.add_argument('--mask_task_weight', type=int, default=1, help='the loss weight of mask task, set it to 0 to not train this task')
    parser.add_argument('--motif_task_weight', type=int, default=100, help='the loss weight of motif task, set it to 0 to not train this task')
    parser.add_argument('--desc_task_weight', type=int, default=300, help='the loss weight of molecular description task, set it to 0 to not train this task')

    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="number of multi-head attention layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="intermediate size")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="dropout probability of attention layer")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="dropout probability of hidden layer")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help=" the eps value in layer normalization components ")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches or as set up in the dict')
    parser.add_argument('--warmup_proportion', type=float, default=0.05, help='Proportion of training to perform linear learning rate warmup')
    parser.add_argument("--device", type=str, default='cuda', help="device for training the model ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc)")

    config = parser.parse_args()

    vocab = MolVocab.load_vocab(config.vocab_path)

    config.pad_token_id = vocab.pad_index
    config.vocab_size = len(vocab)
    config.motif_size = len(vocab.fgtoi)
    config.desc_size = len(vocab.descriptor)

    task_list = []
    if config.mask_task_weight != 0:
        task_list.append('mask_task')

    if config.motif_task_weight != 0:
        task_list.append('motif_task')

    if config.desc_task_weight != 0:
        task_list.append('desc_task')
    
    trainer = BerMolPreTrainer(task_list, vocab, config=config)
    trainer.train(config.train_corpus_path, config.val_corpus_path, on_memory=config.on_memory)


if __name__ == '__main__':
    train()
