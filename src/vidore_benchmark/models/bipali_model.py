import torch
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel


class BiPaliLast(PaliGemmaPreTrainedModel):
    """
    TODO: Update w/ Manu's latest code.
    """

    def __init__(self, config):
        super(BiPaliLast, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        self.pooling_strategy = "last"
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        # pooling - last token
        proj = last_hidden_states[:, -1, :]
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class BiPaliMean(PaliGemmaPreTrainedModel):
    """
    TODO: Update w/ Manu's latest code.
    """

    def __init__(self, config):
        super(BiPaliMean, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        self.pooling_strategy = "mean"
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj
