import torch
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import numpy as np

class BottleneckT5Autoencoder:
    def __init__(self, model_path="t5-small", device='cpu'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, abstract):
        """
        @params:
            abstract: abstract to be embedded
        @return:
            embedding: BaseModelOutputWithPastAndCrossAttention hidden states of encoder model
        """
        inputs = self.tokenizer(abstract, return_tensors="pt", padding=True, truncation=True)
        embedding = self.model.encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        return_dict=True
                        )
        return embedding

    @torch.no_grad()
    def generate_Abstract_and_Embedding(self, list_embeddings):
        """
        @params:
          list_embedding: list of BaseModelOutputWithPastAndCrossAttention hidden states of encoder model
        @return:
            abstract: generated abstract from the list of embeddings
            aggregated_embedding: aggregated embedding of the list of embeddings
        """
        
        # Initialize with the first embedding
        aggregated_embedding = list_embeddings[0]

        # Create accumulator variables for each attribute
        last_hidden_state_sum = aggregated_embedding.last_hidden_state.clone()
        past_key_values_sum = aggregated_embedding.past_key_values.clone() if aggregated_embedding.past_key_values is not None else None
        hidden_states_sum = aggregated_embedding.hidden_states.clone() if aggregated_embedding.hidden_states is not None else None
        attentions_sum = aggregated_embedding.attentions.clone() if aggregated_embedding.attentions is not None else None
        cross_attentions_sum = aggregated_embedding.cross_attentions.clone() if aggregated_embedding.cross_attentions is not None else None
        
        # Add remaining embeddings to the accumulators
        for l in list_embeddings[1:]:
            last_hidden_state_sum = last_hidden_state_sum + l.last_hidden_state
            
            if past_key_values_sum is not None and l.past_key_values is not None:
                past_key_values_sum = past_key_values_sum + l.past_key_values
                
            if hidden_states_sum is not None and l.hidden_states is not None:
                hidden_states_sum = hidden_states_sum + l.hidden_states
                
            if attentions_sum is not None and l.attentions is not None:
                attentions_sum = attentions_sum + l.attentions
                
            if cross_attentions_sum is not None and l.cross_attentions is not None:
                cross_attentions_sum = cross_attentions_sum + l.cross_attentions

        # Calculate scaling factor
        scaling_factor = 1.0 / len(list_embeddings)

        # Create new BaseModelOutputWithPastAndCrossAttentions with scaled values
        aggregated_embedding = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state_sum * scaling_factor,
            past_key_values=past_key_values_sum * scaling_factor if past_key_values_sum is not None else None,
            hidden_states=hidden_states_sum * scaling_factor if hidden_states_sum is not None else None,
            attentions=attentions_sum * scaling_factor if attentions_sum is not None else None,
            cross_attentions=cross_attentions_sum * scaling_factor if cross_attentions_sum is not None else None
        )
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])

        encoder_outputs = aggregated_embedding
        outputs = self.model.generate(  
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=len(decoder_input_ids[0]) + 250,  # Allow for a reasonably sized abstract /!\ replace with average abstract length in given graph
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,

        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the abstract part
        abstract = generated_text
        return abstract, aggregated_embedding