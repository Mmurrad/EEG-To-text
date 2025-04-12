import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np

""" main architecture for open vocabulary EEG-To-Text decoding"""
class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            labels = target_ids_batch_converted,
            return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        # print(f'forward:{input_embeddings_batch.shape,input_masks_batch.shape,input_masks_invert.shape,target_ids_batch_converted.shape,encoded_embedding.shape}')
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                              return_dict = True, labels = target_ids_batch_converted)
        
        return out


from transformers import T5Tokenizer
""" main architecture for open vocabulary EEG-To-Text decoding"""

class T5Translator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(T5Translator, self).__init__()
        
        self.pretrained = pretrained_layers
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        
        # Additional transformer encoder
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=additional_encoder_nhead, 
            dim_feedforward=additional_encoder_dim_feedforward, 
            batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """Process EEG features through additional encoder"""
        encoded_embedding = self.additional_encoder(
            input_embeddings_batch, 
            src_key_padding_mask=input_masks_invert
        )
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            **kwargs  # Accept variable arguments for generation
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)

        # Add task-specific prefix
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # Generate text
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            **kwargs  # Pass all generation parameters
        )
        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        
        # Add task-specific prefix
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # Forward pass
        out = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch,
            labels=target_ids_batch_converted,
            return_dict=True
        )
        return out


""" crippled open vocabulary EEG-To-Text decoding model w/o additional MTE encoder"""
class BrainTranslatorNaive(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslatorNaive, self).__init__()
        '''no additional transformer encoder version'''
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        return out


""" helper modules """
# modified from BertPooler
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


""" Miscellaneous (not working well) """
class BrainTranslatorBert(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 768):
        super(BrainTranslatorBert, self).__init__()

        self.pretrained_Bert = pretrained_layers
        self.fc1 = nn.Linear(in_feature, hidden_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = target_ids_batch, return_dict = True)
        return out

class EEG2BertMapping(nn.Module):
    def __init__(self, in_feature = 840, hidden_size = 512, out_feature = 768):
        super(EEG2BertMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class ContrastiveBrainTextEncoder(nn.Module):
    def __init__(self, pretrained_text_encoder, in_feature = 840, eeg_encoder_nhead=8, eeg_encoder_dim_feedforward = 2048, embed_dim = 768):
        super(ContrastiveBrainTextEncoder, self).__init__()
        # EEG Encoder
        self.positional_embedding = PositionalEncoding(in_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=eeg_encoder_nhead,  dim_feedforward = eeg_encoder_dim_feedforward, batch_first=True)
        self.EEG_Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.EEG_pooler = Pooler(in_feature)
        self.ln_final = nn.LayerNorm(in_feature) # to be considered
        
        # project to text embedding
        self.EEG_projection = nn.Parameter(torch.empty(in_feature, embed_dim))
        
        # Text Encoder
        self.TextEncoder = pretrained_text_encoder
        
        # learned temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_EEG_features, input_EEG_attn_mask, input_ids, input_text_attention_masks):
        # add positional embedding
        input_EEG_features = self.positional_embedding(input_EEG_features)
        # get EEG feature embedding
        EEG_hiddenstates = self.EEG_Encoder(input_EEG_features,  src_key_padding_mask = input_EEG_attn_mask)
        EEG_hiddenstates = self.ln_final(EEG_hiddenstates)
        EEG_features = self.EEG_pooler(EEG_hiddenstates) # [N, 840]

        # project to text embed size
        EEG_features = EEG_features @ self.EEG_projection # [N, 768]

        # get text feature embedding
        Text_features = self.TextEncoder(input_ids = input_ids, attention_mask = input_text_attention_masks, return_dict = True).pooler_output # [N, 768]
        
        # normalized features
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True) # [N, 768]
        Text_features = Text_features / Text_features.norm(dim=-1, keepdim=True) # [N, 768]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() 
        logits_per_EEG = logit_scale * EEG_features @ Text_features.t() # [N, N]
        logits_per_text = logit_scale * Text_features @ EEG_features.t() # [N, N]

        return logits_per_EEG, logits_per_text
 

class R1Translator(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_feature=840,
        decoder_embedding_size=1024,
        rnn_hidden_size=256,
        bidirectional=True,
        num_rnn_layers=2
    ):
        super(R1Translator, self).__init__()

        self.pretrained = pretrained_layers

        # Additional RNN encoder (LSTM)
        self.rnn_layer = nn.LSTM(
            input_size=in_feature,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Calculate RNN output size for bidirectional processing
        self.rnn_output_size = rnn_hidden_size * (2 if bidirectional else 1)

        # Projection to decoder embedding size
        self.fc1 = nn.Linear(self.rnn_output_size, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """Process EEG features through LSTM encoder"""
        rnn_out, _ = self.rnn_layer(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(rnn_out))
        return encoded_embedding

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        out = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch,
            labels=target_ids_batch_converted,
            return_dict=True
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_masks_invert,
        target_ids_batch_converted,
        **kwargs
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            **kwargs
        )
        return output


class BrainBERT(nn.Module):
    def __init__(self, bert_encoder, bart_decoder, in_feature=840):
        super(BrainBERT, self).__init__()
        self.bert_encoder = bert_encoder
        self.bart_decoder = bart_decoder
        
        # Additional EEG encoder
        self.eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=8, 
            dim_feedforward=2048, 
            batch_first=True
        )
        self.eeg_encoder = nn.TransformerEncoder(self.eeg_encoder_layer, num_layers=6)
        
        # Projection layer to match BERT's hidden size
        self.fc1 = nn.Linear(in_feature, bert_encoder.config.hidden_size)
    
    def forward(self, eeg_input_batch, text_input_ids, text_attention_mask, labels=None):
        # Process EEG data through additional encoder
        eeg_embeddings = self.eeg_encoder(eeg_input_batch)
        eeg_embeddings = F.relu(self.fc1(eeg_embeddings))
        
        # Process text through BERT encoder
        text_embeddings = self.bert_encoder.embeddings(input_ids=text_input_ids)
        
        # Combine EEG and text embeddings
        combined_embeddings = torch.cat([eeg_embeddings, text_embeddings], dim=1)
        combined_attention_mask = torch.cat([
            torch.ones(eeg_embeddings.size(0), eeg_embeddings.size(1), device=eeg_embeddings.device),
            text_attention_mask
        ], dim=1)
        
        # Pass through BERT encoder
        encoder_outputs = self.bert_encoder.encoder(combined_embeddings, attention_mask=combined_attention_mask)
        
        # Use encoder outputs as input to BART decoder
        decoder_outputs = self.bart_decoder(
            inputs_embeds=encoder_outputs.last_hidden_state,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        return decoder_outputs