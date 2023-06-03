import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.modeling_bert import BertPooler
from Transformer_encoder import transformer
from GRU import GRUCell
from bert import BertEncoder
from global_configs import *
from module import *

m_hidden_size = VISUAL_DIM


class CAdapter(nn.Module):
    def __init__(self, args):
        super(CAdapter, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args

        self.down_project_text = nn.Linear(
            TEXT_DIM,
            self.hidden_size
        )

        self.up_project = nn.Linear(
            self.hidden_size,
            TEXT_DIM
        )

        self.dec_list = nn.ModuleList(
            [CA(args) for _ in range(args.ca_layer)]
        )

        self.init_weights()

    def forward(self, text_hidden, other_hidden):
        x = self.down_project_text(other_hidden)
        y = self.down_project_text(text_hidden)

        x_mask = self.make_mask(x)
        y_mask = self.make_mask(y)

        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)
        x = self.up_project(x)
        return x

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def init_weights(self):
        self.down_project_text.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.down_project_text.bias.data.zero_()

        self.up_project.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class SAdapter(nn.Module):
    def __init__(self, args):
        super(SAdapter, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self.down_project_text = nn.Linear(
            TEXT_DIM,
            self.hidden_size
        )
        self.down_project_acoustic = nn.Linear(
            ACOUSTIC_DIM,
            self.hidden_size
        )
        self.down_project_visual = nn.Linear(
            VISUAL_DIM,
            self.hidden_size
        )

        self.up_project = nn.Linear(
            self.hidden_size,
            TEXT_DIM
        )

        self.dec_list = nn.ModuleList(
            [SA(args) for _ in range(args.sa_layer)]
        )

        self.init_weights()

    def forward(self, hidden_state, f):
        # Get hidden vector
        # "v" x: visual    y: text
        # "a" x: acoustic  y: text
        if f == "a":
            x = self.down_project_acoustic(hidden_state)
        else:
            x = self.down_project_visual(hidden_state)

        x_mask = self.make_mask(x)

        for dec in self.dec_list:
            x = dec(x, x_mask)
        x = self.up_project(x)
        return x

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def init_weights(self):
        self.down_project_text.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.down_project_text.bias.data.zero_()
        self.down_project_acoustic.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.down_project_acoustic.bias.data.zero_()
        self.down_project_visual.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.down_project_visual.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.args.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class BertPooler(nn.Module):
    def __init__(self, config, args):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(args.d_l, args.d_l)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, num_labels):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.pooler = BertPooler(self.config, args)
        self.MulGate = MulGate(args.dropout_prob)
        self.num_labels = num_labels

        self.adapter_list1 = args.adapter_list1
        self.adapter_list2 = args.adapter_list2
        self.adapter_num1 = len(self.adapter_list1)
        self.adapter_num2 = len(self.adapter_list2)

        self.sadapter1 = SAdapter(args)
        self.sadapter2 = SAdapter(args)
        self.cadapter1 = nn.ModuleList(
            [CAdapter(args) for _ in range(self.adapter_num1)])
        self.cadapter2 = nn.ModuleList(
            [CAdapter(args) for _ in range(self.adapter_num2)])

        self.style_reid_layer = ChannelGate_sub(60, num_gates=60, return_gates=False,
                                                gate_activation='sigmoid', reduction=15,
                                                layer_norm=False)

        self.com_dense = nn.Linear(args.d_l * 2, args.d_l)
        self.dense = nn.Linear(args.d_l * 2, args.d_l)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.d_l, self.num_labels)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1, self.num_labels)

        self.gru_cell_at = GRUCell(args)
        self.gru_cell_vt = GRUCell(args)

    def forward(self, pretrained_model_outputs, visual, acoustic, labels=None, mode='train'):
        n_batch = visual.size(0)
        outputs = pretrained_model_outputs
        sequence_output = outputs[1]
        hidden_states = outputs[3]  # hidden_states: bert各层的输出
        output_l = outputs[4]

        acoustic_state = self.sadapter1(acoustic, "a")
        adapter_states_1 = acoustic_state
        # for i, cadapter_module in enumerate(self.cadapter1):
        #     bert_state_1 = hidden_states[self.adapter_list1[i]]
        #     adapter_states_at = cadapter_module(bert_state_1, adapter_states_1)
        #     adapter_states_1 = adapter_states_at

        visual_state = self.sadapter2(visual, "v")
        adapter_states_2 = visual_state
        # for i, cadapter_module in enumerate(self.cadapter2):
        #     bert_state_2 = hidden_states[self.adapter_list2[i]]
        #     adapter_states_vt = cadapter_module(bert_state_2, adapter_states_2)
        #     # adapter_states_2 = self.gru_cell_vt(adapter_states_vt, adapter_states_2)
        #     adapter_states_2 = adapter_states_vt

        hidden_states_last = self.MulGate(adapter_states_1, adapter_states_2,
                                          acoustic_state, visual_state)
        # outputs_l = self.pooler(sequence_output)
        com_features = self.com_dense(torch.cat([output_l, hidden_states_last], dim=2))
        output_l = self.pooler(output_l)
        pooled_output = self.pooler(com_features)
        logits = self.out_proj(self.dropout(pooled_output))

        outputs = (logits,) + (sequence_output,)

        output_a = self.pooler(adapter_states_1)
        # adapter_states_1_useful = self.out_proj(self.pooler(adapter_states_1_useful).view(n_batch, -1)).view(-1)
        # adapter_states_1_useless = self.out_proj(self.pooler(adapter_states_1_useless).view(n_batch, -1)).view(-1)

        output_v = self.pooler(adapter_states_2)
        # adapter_states_2_useful = self.out_proj(self.pooler(adapter_states_2_useful).view(n_batch, -1)).view(-1)
        # adapter_states_2_useless = self.out_proj(self.pooler(adapter_states_2_useless).view(n_batch, -1)).view(-1)

        # return outputs, adapter_states_1f, adapter_states_1_useful, adapter_states_1_useless, \
        #        adapter_states_2f, adapter_states_2_useful, adapter_states_2_useless
        return outputs, output_a, output_v, pooled_output, output_l


class MulGate(nn.Module):
    def __init__(self, dropout_prob):
        super(MulGate, self).__init__()
        self.W1 = nn.Linear(TEXT_DIM * 2, TEXT_DIM)
        self.W2 = nn.Linear(TEXT_DIM * 2, TEXT_DIM)

        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, adapter1, adapter2, acoustic_state, visual_state):
        state1 = torch.cat((adapter1, acoustic_state), dim=-1)
        state2 = torch.cat((adapter2, visual_state), dim=-1)
        weight1 = F.relu(self.W1(state1))
        weight2 = F.relu(self.W2(state2))

        gate_fusion = weight1 * adapter1 + weight2 * adapter2

        fusion_state = self.dropout(
            self.LayerNorm(gate_fusion)
        )

        return fusion_state


class ChannelGate_sub(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x
