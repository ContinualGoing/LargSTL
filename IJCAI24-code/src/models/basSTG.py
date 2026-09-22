import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel
from src.utils.conv import Conv2d
from functools import reduce
from torch import autograd
from torch.autograd import Variable
#import utils


class GWNET(BaseModel):
    def __init__(self, adp_adj, dropout, residual_channels, dilation_channels, \
                 skip_channels, end_channels, supports_len,kernel_size=2, blocks=4, layers=2,adaver=False,cnn_learning_init=2,gcn_learning_init=2, **args):
        super(GWNET, self).__init__(**args)
        self.supports_len=supports_len
        self.adp_adj = adp_adj

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.adaver=adaver
        self.cnn_learning_init=cnn_learning_init
        self.gcn_learning_init=gcn_learning_init
        self.start_conv = Conv2d(in_channels=self.input_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1),adaver=False,learning_init=self.cnn_learning_init)

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation,adaver=self.adaver,learning_init=self.cnn_learning_init))

                self.gate_convs.append(Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1,kernel_size), dilation=new_dilation,adaver=self.adaver,learning_init=self.cnn_learning_init))

                self.skip_convs.append(Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1,1),adaver=adaver,learning_init=self.cnn_learning_init))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(dilation_channels, residual_channels, self.dropout, support_len=self.supports_len,adaver=self.adaver,learning_init=self.gcn_learning_init))
        self.receptive_field = receptive_field
        
        self.end_conv_1 = Conv2d(in_channels=skip_channels,out_channels=end_channels,kernel_size=(1,1),bias=True,adaver=False)

        self.end_conv_2 = Conv2d(in_channels=end_channels, out_channels=self.output_dim * self.horizon,kernel_size=(1,1),bias=True,adaver=False)

        #self.end_mlp1=nn.Parameter(torch.randn((skip_channels, end_channels)))
        #self.end_mlp2=nn.Parameter(torch.randn((end_channels, self.output_dim * self.horizon)))
        #nn.init.xavier_normal_(self.end_mlp1) 
        #nn.init.xavier_normal_(self.end_mlp2)
    
    def fre_parameter(self):
        for name, param in self.named_parameters():
            if 'adaver' not in name:
                param.requires_grad=False
    def forward(self, input, adj,label=None): 
        # (b, t, n, f) 
        input = input.transpose(1,3)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input


        new_supports = adj

        x = self.start_conv(x)

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:         
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, new_supports)
            
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        '''
        x = F.relu(skip)
        x=x.transpose(3,1)
        x = F.relu(torch.matmul(x,self.end_mlp1))
        x=torch.matmul(x,self.end_mlp2).transpose(3,1)
        '''
        return x

    def updata_adj(self,adj):
        self.supports=adj


    def estimate_fisher(self, data_loader, sample_size,to_device, to_tensor,trainadj,inverse_transform,batch_size=32):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        self.ewc_para={n:p for n, p in self.named_parameters() if 'adaver' not in n}
        for x, y in data_loader.get_iterator():

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = to_device(to_tensor([X, label]))
            pred = self.model(X, trainadj,label)
            pred, label = inverse_transform([pred, label])

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.ewc_para.items()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.ewc_para.items():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.ewc_para.items():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        #print(x.shape,A.shape)
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)


    def forward(self,x):
        return self.mlp(x)
class embedding_laryer(nn.Module):
    def __init__(self,num_nodes, node_dim,if_spatial,if_time_in_day,if_day_in_week):
                # spatial embeddings
        self.num_nodes=num_nodes
        self.node_dim=node_dim
        self.if_spatial=if_spatial
        self.if_time_in_day=if_time_in_day
        self.if_day_in_week=if_day_in_week
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb  = nn.Parameter(
                torch.empty(7, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb )

        # embedding layer
        self.time_sires_emb_layer = nn.Conv2d(
            self.input_dim * self.input_len, self.embed_dim, kernel_size=(1, 1), bias=True
        )
    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2,adaver=False,learning_init=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        #self.mlp = linear(c_in, c_out)
        self.mlp=nn.Parameter(torch.randn((c_in, c_out)))
        nn.init.xavier_normal_(self.mlp) 
        self.dropout = dropout
        self.order = order
        self.adaver_state=adaver
        self.adaver=nn.Parameter(torch.randn((c_in, c_out)))
        #nn.init.zeros_(self.adaver)
        if learning_init==2:
            nn.init.xavier_normal_(self.adaver)
        elif learning_init==1:
            nn.init.ones_(self.adaver)
        else:
            nn.init.zeros_(self.adaver)
        self.weight=nn.Parameter(torch.mul(self.adaver,self.mlp))
        

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h=h.transpose(3,1)
        if self.adaver_state==True:
            h = torch.matmul(h,self.weight).transpose(3,1)
        else:
            h = torch.matmul(h,self.mlp).transpose(3,1)
        h = F.dropout(h, self.dropout, training=self.training)
        return h