import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
import torch.nn.functional as F

class Gater(nn.Module):
    def __init__(
            self,
            num_layers,
            num_heuristics,
            input_dim,
            hidden_dim,
            num_models,
            dropout_ratio,
            norm_type="none",
            add_prediction = False,
            disable_heuristics = False,
        ):
        super(Gater, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.feat_mlp = MLP(num_layers,
                            input_dim,
                            hidden_dim,
                            hidden_dim,
                            dropout_ratio,
                            norm_type,
                            )
        self.add_prediction = add_prediction
        if add_prediction:
            num_h = num_heuristics + num_models
        else:
            num_h = num_heuristics
        self.disable_heuristics = disable_heuristics
        if disable_heuristics == False:
            self.heuristic_mlp = MLP(num_layers,
                            num_h,
                            hidden_dim,
                            hidden_dim,
                            dropout_ratio,
                            norm_type,
                            )
            self.gater_mlp = MLP(num_layers,
                            2 * hidden_dim,
                            hidden_dim,
                            num_models,
                            dropout_ratio,
                            norm_type,
                            )
        else:
            self.gater_mlp = MLP(1,
                            hidden_dim,
                            hidden_dim,
                            num_models,
                            dropout_ratio,
                            norm_type,
                            )


        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self):
        self.feat_mlp.reset_parameters()
        if self.disable_heuristics == False:
            self.heuristic_mlp.reset_parameters()
        self.gater_mlp.reset_parameters()

    def forward(self, x_i, x_j, heuristics, model_prediction, return_weight=False):
#        x = torch.cat((x_i,x_j), dim=1)
        x = x_i * x_j
        hidden1 = self.feat_mlp(x)
        if self.add_prediction:
            input_h = torch.cat((heuristics, model_prediction), dim=1)
        else:
            input_h = heuristics
        if self.disable_heuristics == False:
            hidden2 = self.heuristic_mlp(input_h)
            hidden = torch.cat((hidden1, hidden2), dim=1)
        else:
            hidden = hidden1
        logit = self.gater_mlp(hidden)
        logit = 0*model_prediction
        weight = self.softmax(logit)
        out = torch.sum(weight * model_prediction, dim=1)
        if return_weight:
            return (out+1e-6)/(1+1e-3), weight
        else:
            return (out+1e-6)/(1+1e-3)
       

class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
        
class SAGE(torch.nn.Module):
    def __init__(self, data_name, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_layer, norm_type="none"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))            

        self.convs.append(conv_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_channels))
        self.convs.append(conv_layer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        # import ipdb; ipdb.set_trace()
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm_type != "none":
                    x = self.norms[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return torch.sigmoid(x)
