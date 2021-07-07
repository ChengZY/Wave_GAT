import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAT0(nn.Module):
    def __init__(self, data, nhid=8, nhead=8, nhead_out=1, alpha=0.2, dropout=0.6):
        super(GAT0, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.attentions = [GATConv(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nhead)]
        self.out_atts = [GATConv(nhid * nhead, nclass, dropout=dropout, alpha=alpha) for _ in range(nhead_out)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.out_atts):
            self.add_module('out_att{}'.format(i), attention)
        self.reset_parameters()

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self, data):
        x, edge_list = data.features, data.edge_list
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = F.elu(x)
        x = torch.sum(torch.stack([att(x, edge_list) for att in self.out_atts]), dim=0) / len(self.out_atts)
        return F.log_softmax(x, dim=1)


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, support_len, order, bias=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


        self.nconv = nconv()
        c_in = (order * support_len + 1) * in_features
        self.mlp = linear(c_in, out_features)
        self.dropout = dropout
        self.order = order

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_lists):
        print('train')
        print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # h = torch.matmul(x, self.weight[0])

        out = [x]
        h_primes = []
        for edge_list in edge_lists:
            h1 = [x]
            x1 = self.nconv(x, edge_list)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, edge_list)
                out.append(x2)
                x1 = x2

                h = torch.cat(h1, dim=1)

                edge_list_ix = (edge_list > 0).nonzero().t()
                source, target = edge_list_ix

                '''OOM Here'''
                a_input = torch.cat([h[source], h[target]], dim=1)
                print(self.a.shape)
                print(edge_list.shape)
                print(a_input.shape)

                # a_input = torch.transpose(a_input, 1, 3)
                # e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)
                # e = torch.transpose(e, 1, 3)
                # print(e.shape)

                # N = h.size(0)
                # attention = -1e20*torch.ones([N, N], device=device, requires_grad=True)
                # attention = torch.tensor(edge_list, device=device, requires_grad=True)
                attention = edge_list
                # attention[source, target] = e[:, 0]
                attention = F.softmax(attention, dim=1)
                attention = F.dropout(attention, self.dropout, training=self.training)
                h = F.dropout(h, self.dropout, training=self.training)
                h_prime = torch.matmul(attention, h)
                print('here')
                print(h_prime.shape)
                ''''to do, align bias'''
                # if self.bias is not None:
                #     h_prime = h_prime + self.bias
                h_primes.append(h_prime)

        h_primes = torch.cat(h_prime, dim=1)
        return h_primes


        # return h_prime


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GATCon(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GATCon, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

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
        h1 = h
        h = self.mlp(h)
        h2 = h
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GAT(nn.Module):
    def __init__(self,  c_in, c_out, dropout, support_len=3, order=2, nhid=32, nhead=8, nhead_out=1, alpha=0.2):
        super(GAT, self).__init__()
        # nfeat, nclass =  c_in, c_out
        self.attentions = [GATCon(c_in, nhid, dropout, support_len=3, order=2) for _ in range(nhead)]
        self.out_atts = [GATCon(nhid * nhead, c_out, dropout, support_len=3, order=2) for _ in range(nhead_out)]
        # self.attentions = [GATConv(c_in, nhid, dropout, alpha, support_len=3, order=2,bias=None) for _ in range(nhead)]
        # self.out_atts = [GATConv(nhid * nhead, c_out, dropout, alpha, support_len=3, order=2, bias=None) for _ in range(nhead_out)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.out_atts):
            self.add_module('out_att{}'.format(i), attention)
        # self.reset_parameters()

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self, x, support):
        x, edge_list = x, support
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = F.elu(x)
        x = torch.sum(torch.stack([att(x, edge_list) for att in self.out_atts]), dim=0) / len(self.out_atts)
        # xs = []
        # for att in self.out_atts:
        #     a = att(x, edge_list)
        #     xs.append(a)
        y = F.log_softmax(x, dim=1)
        return y
