import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Encoder(nn.Module):

    def __init__(self, embedding_size):

        super(Encoder, self).__init__()

        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Remove the fully connected layers, since we don't need the original resnet classes anymore
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Create a new fc layer based on the embedding size
        self.linear = nn.Linear(
            in_features=resnet.fc.in_features, out_features=embedding_size)
        self.BatchNorm = nn.BatchNorm1d(
            num_features=embedding_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.BatchNorm(self.linear(features))
        return features


class Decoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=vocab_size)

    def forward(self, features, captions, length):
        embeddings = self.embed(captions)

        features = features.unsqueeze(1)

        embeddings = torch.cat((features, embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            input=embeddings, lengths=length, batch_first=True)
        hidden, _ = self.lstm(packed)
        output = self.linear(hidden[0])

        return output

    def sample(self, features, states=None, longest_sentence_length=100):

        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(longest_sentence_length):

            hidden, states = self.lstm(inputs, states)

            output = self.linear(hidden.squeeze(1))
            predicted = output.max(dim=1, keepdim=True)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.view(-1, 1, self.embed_size)

        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids.squeeze()

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, nhead=8, dropout=0.1):
        super(DecoderTransformer, self).__init__()

        self.embed_size = embed_size
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        transformer_decoder_layer = TransformerDecoderLayer(embed_size, nhead, hidden_size, dropout)
        self.transformer_decoder = TransformerDecoder(transformer_decoder_layer, num_layers)

        self.linear = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, features, tgt, tgt_mask=None):
        target_seq_len = tgt.size(1)
        features = self.preprocess_encoder_features(features, target_seq_len)
        tgt = self.embed(tgt)
        tgt = self.pos_enc(tgt)
        output = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask)
        output = self.linear(output)
        return output

    def sample(self, features, sos_token, eos_token, max_len=100):
        features = self.preprocess_encoder_features(features, max_len)
        batch_size = features.size(0)
        sampled_ids = [torch.ones(batch_size, 1).fill_(sos_token).long().to(features.device)]
        inputs = sampled_ids[-1]

        for i in range(max_len):
            tgt = self.embed(inputs)
            tgt = self.pos_enc(tgt)
            output = self.transformer_decoder(tgt, features)
            output = self.linear(output)
            predicted = output.max(dim=2, keepdim=True)[1]
            sampled_ids.append(predicted)
            inputs = predicted

            if (predicted.squeeze() == eos_token).all():
                break

        sampled_ids = torch.cat(sampled_ids, dim=1)
        return sampled_ids.squeeze()

    def preprocess_encoder_features(self, features, target_seq_len):
        features = features.unsqueeze(1)
        features = features.repeat(1, target_seq_len, 1)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

