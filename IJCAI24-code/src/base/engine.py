import os
import time
import torch
import numpy as np
import networkx as nx
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics



def get_n_hop_subgraph(adj_matrix, nodes, n):
    G = nx.from_numpy_array(adj_matrix)  # 从邻接矩阵创建图

    n_hop_nodes = set(nodes)  # 存储 n-hop 子图的节点集合
    neighbors = set(nodes)  # 存储当前节点的邻居集合

    for _ in range(n):
        new_neighbors = set()
        for neighbor in neighbors:
            new_neighbors.update(G.neighbors(neighbor))
        n_hop_nodes.update(new_neighbors)
        neighbors = new_neighbors
    return np.array(list(n_hop_nodes)) 


class BaseEngine():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed,save_path=None,learning_step=0):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = save_path
        self._logger = logger
        self._seed = seed
        self._learningstep=0
        
        self.all_steps=learning_step
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 

        self.train_time=0

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)
    def fre_parameter(self):
        self.model.fre_parameter()
    def load_best_model(self,model,fine_lune=False):
        if fine_lune==True:
            filename = 'final_model_step{}.pt'.format(self._learningstep)
            all_par=torch.load(os.path.join(self._save_path, filename))
            all_par={name:key for name,key in all_par.items() if "adaver" not in name and 'end' not in name}
            print([name for name,_ in all_par.items()])
            model.load_state_dict(all_par,strict = False)
        self.model=model
        self.model.to(self._device)
        del model
        self.fre_parameter()
    def updata_epoch(self,epoch):
        self._max_epochs=epoch
    def load_best_model_test(self,model):
        #filename = 'final_model_step{}.pt'.format(self._learningstep)
        all_par=torch.load('/home/wbw/Large-graph/LargeST-main/LargeST-main/results/2024-01-13 06:43:42/gwnet/GLA-doubletransition-1/final_model_step0.pt')
        print('load_best_file_name:','/home/wbw/Large-graph/LargeST-main/LargeST-main/results/2024-01-13 06:43:42/gwnet/GLA-doubletransition-1/final_model_step0.pt')
        res_par={name:key for name,key in all_par.items() if "adaver" not in name}
        model.load_state_dict(res_par,strict = False)
        self.model=model
        self.model.to(self._device)
        del model,all_par
        self.fre_parameter()
    def update_step(self):
        self._learningstep=self._learningstep+1
    def updata_adj(self,adj):
        self.model.updata_adj(adj)
    def update_data(self,dataloader): 
        self._dataloader = dataloader
    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_step{}.pt'.format(self._learningstep)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    def update_nor_adj(self,adj):
        self.nor_adj=adj

    def load_model(self, save_path):
        filename = 'final_model_step{}.pt'.format(self._learningstep)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   
    def get_adj(self,adj):
        self._trainadj=adj[0]
        self._valadj=adj[1]
        self._testadj=adj[2]

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, self._trainadj,label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()
            self.train_time=self.train_time+v2-t1
            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
        if self._learningstep==self.all_steps:               
            self.evaluate('test')
        else:
            self.evaluate('replay')
            self.evaluate('test')


    def evaluate(self, mode,test_state=0):
        
        if mode == 'test' or mode=='replay':
            if test_state==0:
                self.load_model(self._save_path)
            if mode=='test':
                adj=self._testadj
                mode1=mode
            else:
                adj=self._valadj
                mode1='val'
        else:
            adj=self._valadj
            mode1='val'
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode1 + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, adj,label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse
        
        if mode == 'replay':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            adj_test_mae=[]
            for i in range(self.node_num):
                res = compute_all_metrics(preds[:,:,i], labels[:,:,i], mask_value)
                adj_test_mae.append(res[0])
            self.replay_sort=sorted(range(len(adj_test_mae)), key=lambda x: adj_test_mae[x])  #包括replay node+current nodes
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f},Time:{:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),self.train_time))

    def set_nodesnum(self,num):
        self.node_num=num

    def update_optimizer(self,op):
        self._optimizer=op

    def get_replay_node(self,replay_num,n_hop):
        replay_nodes=self.replay_sort[:replay_num]
        out=get_n_hop_subgraph(self.nor_adj,replay_nodes,n_hop)
        return out
    
    