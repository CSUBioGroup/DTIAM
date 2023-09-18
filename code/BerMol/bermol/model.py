import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss, MSELoss


class MolEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, token_ids):
        embeddings = self.token_embeddings(token_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + torch.unsqueeze(attention_mask, 1)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.view(new_context_layer_shape)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.selfatt = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.selfatt(hidden_states, attention_mask)
        outputs = self.output(self_outputs, hidden_states)
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, sequence_output):
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BerMolEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embeddings = MolEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
    
    def forward(self, token_ids, attention_mask=None):
        embeddings = self.token_embeddings(token_ids)
        sequence_output = self.encoder(embeddings, attention_mask)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class MaskPredictor(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.weight = config.mask_task_weight
        self.loss = CrossEntropyLoss(ignore_index=0)
        self.vocab_size = config.vocab_size

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.predictor = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.predictor.bias = self.bias
    
    def forward(self, sequence_output, pooled_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_scores = self.predictor(hidden_states)
        return prediction_scores
    
    def compute_loss(self, predictions, labels):
        return self.loss(predictions.view(-1, self.vocab_size), labels.view(-1))


class MotifPredictor(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.weight = config.motif_task_weight
        self.loss = MultiLabelSoftMarginLoss()
        self.motif_size = config.motif_size
        self.predictor = nn.Linear(config.hidden_size, config.motif_size)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictor(pooled_output)
        return prediction_scores
    
    def compute_loss(self, predictions, labels):
        return self.loss(predictions, labels)


class DescPredictor(nn.Module):
    def __init__(self, name, config):
        super().__init__()

        self.name = name
        self.weight = config.desc_task_weight
        self.loss = MSELoss()
        self.desc_size = config.desc_size
        self.predictor = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.desc_size),
        )

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictor(pooled_output)
        return prediction_scores
    
    def compute_loss(self, predictions, labels):
        return self.loss(predictions, labels)


class BerMol(nn.Module):

    def __init__(self, task_modules: nn.ModuleList, config):
        super().__init__()

        self.encoder = BerMolEncoder(config)
        self.task_modules = task_modules
    
    def forward(self, token_ids, attention_mask=None):
        sequence_output, pooled_output = self.encoder(token_ids, attention_mask)
        return {task_layer.name: task_layer(sequence_output, pooled_output) for task_layer in self.task_modules}
