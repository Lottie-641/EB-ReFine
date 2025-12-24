import torch.nn as nn

class BertMultiOutputClassificationHeads(nn.Module):
    def __init__(self, gpt_model, output_sizes):
        super(BertMultiOutputClassificationHeads, self).__init__()
        self.gpt_model = gpt_model
        self.output_layers = nn.ModuleList([nn.Linear(gpt_model.config.hidden_size, output_sizes[i]) for i in range(len(output_sizes))])

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        out = []
        for output_layer in self.output_layers:
            out.append(output_layer(pooled_output))

        return out