import os
import argparse
import numpy as np
import networkx as nx
import metis
import torch_geometric
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import time
import torch
torch.set_num_threads(3)

from src.models.basSTG import basSTG
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def find_indices(lst, clusters):
    indices={}
    for cluster in range(clusters):
        indices_tem = []
        for i in range(len(lst)):
            if lst[i] == cluster:
                indices_tem.append(i)
        indices[cluster]=np.array(indices_tem)     
    return indices

def par_graph(adj,cluster=5):
    adj=adj.astype(int)
    G=nx.from_numpy_array(adj)
    G.graph['edge_weight_attr']='weight'         
    [cost, parts] = metis.part_graph(G, nparts=cluster, recursive=True)
    indices=find_indices(parts,cluster)
    return  indices

def get_config():
    parser = get_public_config()
    parser.add_argument('--adj_type', type=str, default='doubletransition')
    parser.add_argument('--adp_adj', type=int, default=1)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--skip_dim', type=int, default=256)
    parser.add_argument('--end_dim', type=int, default=512)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--cluster', type=int, default=5)#replay_size
    #parser.add_argument('--dataset', type=str, default='SD')
    parser.add_argument('--replay_size', type=float, default=0.1)

    current_timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp))
    dir_path='/home/wbw/Large-graph/LargeST-main/LargeST-main/results/'+current_time_str
    os.makedirs(dir_path, exist_ok=True)

    parser.add_argument('--save_path', type=str, default=dir_path)
    args = parser.parse_args()

    folder_name = '{}-{}-{}'.format(args.dataset, args.adj_type, args.adp_adj)
    log_dir = '{}/{}/{}/'.format(dir_path,args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    vars(args)["save_path"]=log_dir 
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.dataset=='GBA':
        complete_adj = load_adj_from_numpy('./gba/gba_rn_adj.npy')
        print(complete_adj.shape)
    elif args.dataset=='GLA':
        complete_adj = load_adj_from_numpy('./gla/gla_rn_adj.npy')
    else:
        complete_adj = load_adj_from_numpy('./ca/ca_rn_adj.npy')
    compile_adj_mx = normalize_adj_mx(complete_adj, args.adj_type)

    test_supports = [torch.tensor(i).to(device) for i in compile_adj_mx]
    supports_len=len(test_supports)

    graph_parts=par_graph(complete_adj,args.cluster)#{}
    print([len(graph_parts[i]) for i in range(args.cluster)])
    butter_replay=[]
    for learning_step in range(args.cluster):
        node_list=list()
        node_list.extend(graph_parts[learning_step])
        #replay nodes
        if learning_step!=0:
            replay_num=int(len(graph_parts[learning_step-1])*0.001)
            replay_idx=engine.get_replay_node(replay_num,2) #include 2-hop

            old_idx=graph_parts[learning_step-1]
            old_idx_len=len(graph_parts[learning_step-1])
 
            replay_list=old_idx[ [i for i in replay_idx if i < old_idx_len]]
            butter_replay.extend(replay_list)
            node_list.extend(butter_replay)  
        node_all=np.array(list(set(node_list)))
        adj_mx=complete_adj[node_all,:]
        adj_mx=adj_mx[:,node_all]
        nor_adj= normalize_adj_mx(adj_mx, args.adj_type)

        save_path=args.save_path+str(learning_step)+'_adj.npy'
        np.save(save_path,nor_adj)

        train_supports = [torch.tensor(i).to(device) for i in nor_adj]
        dataloader, scaler = load_dataset(args, logger,node_all)

        if learning_step==0:
            model = basSTG(input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        adp_adj=args.adp_adj,
                        dropout=args.dropout,
                        residual_channels=args.init_dim,
                        dilation_channels=args.init_dim,
                        skip_channels=args.skip_dim,
                        end_channels=args.end_dim,
                        supports_len=supports_len,
                        adaver=True)
    
            loss_fn = masked_mae
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
            scheduler = None

            engine = BaseEngine(device=device,
                                model=model,
                                dataloader=dataloader,
                                scaler=scaler,
                                sampler=None,
                                loss_fn=loss_fn,
                                lrate=args.lrate,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                clip_grad_value=args.clip_grad_value,
                                max_epochs=1500,
                                patience=10,
                                log_dir=log_dir,
                                logger=logger,
                                seed=args.seed,
                                save_path=args.save_path,
                                learning_step=args.cluster,
                                )
            engine.set_nodesnum(len(graph_parts[learning_step]))
            engine.get_adj([train_supports,train_supports,train_supports])
            engine.update_nor_adj(adj_mx)
            engine.train()
        else:
             
             model = basSTG(input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        adp_adj=args.adp_adj,
                        dropout=args.dropout,
                        residual_channels=args.init_dim,
                        dilation_channels=args.init_dim,
                        skip_channels=args.skip_dim,
                        end_channels=args.end_dim,
                        supports_len=supports_len,
                        adaver=True,cnn_learning_init=1,gcn_learning_init=0)
             scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 50, 80], gamma=0.5)
             optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
             if learning_step==10000:
                 engine.load_best_model_test(model)
             else:
                engine.load_best_model(model,fine_lune=True)
             
             engine.update_optimizer(optimizer)
             engine.set_nodesnum(len(graph_parts[learning_step]))
             engine.get_adj([train_supports,train_supports,train_supports]) 
             engine.update_nor_adj(adj_mx)
             engine.update_step()
             engine.update_data(dataloader)
             engine.updata_epoch(200)
             #engine.evaluate('test',test_state=1)
             engine.train()


if __name__ == "__main__":
    print('No replay')
    main()
    


    #第一个窗口：完整的，K==3 Log directory: /home/wbw/Large-graph/LargeST-main/LargeST-main/results/2024-01-12 00:32:38/gwnet/GLA-doubletransition-1/
    #第二个窗口，完整的，k==4 Log directory: /home/wbw/Large-graph/LargeST-main/LargeST-main/results/2024-01-12 00:32:38/gwnet/GLA-doubletransition-1/
    #CMD:跳过第一个步骤 Log directory: /home/wbw/Large-graph/LargeST-main/LargeST-main/results/2024-01-12 00:33:50/gwnet/GLA-doubletransition-1/