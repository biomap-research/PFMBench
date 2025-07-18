from model_zoom.esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor

# protein = ESMProtein(sequence="AAAAA")
client = ESMC.from_pretrained("esmc_600m").to("cuda") # or "cpu"


sequence_tokens = client._tokenize(["AAAAA", "AAAAAMK"])
protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens).to(
            next(client.parameters()).device
        )
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits, logits_output.embeddings)