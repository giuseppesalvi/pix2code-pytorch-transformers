import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerDecoder, TransformerDecoderLayer


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

# Decoder implementation with Transformer 
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
        
    def preprocess_encoder_features(self, features, target_seq_len):
        features = features.unsqueeze(1)
        features = features.repeat(1, target_seq_len, 1)
        return features

    def greedy_search(self, features, start_token_id, end_token_id, max_len=100):
        batch_size = features.size(0)
        generated_sequences = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=features.device)

        for _ in range(max_len):
            output = self.forward(features, generated_sequences)
            _, next_words = torch.max(output[:, -1, :], dim=-1)
            next_words = next_words.unsqueeze(1)
            generated_sequences = torch.cat((generated_sequences, next_words), dim=1)

            # Check if all sequences have reached the end_token_id
            if torch.all((next_words == end_token_id).view(-1)):
                break

        # Remove the initial start_token_id from the generated sequences
        return generated_sequences[:, 1:]


# Original Pix2code models
class P2cVisionModel(nn.Module):
    def __init__(self):
        super(P2cVisionModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(128*32*32, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, images):
        return self.cnn(images)


class P2cLanguageModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(P2cLanguageModel, self).__init__()

        # TODO: check this better
        self.embed_size = embed_size
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm1 = nn.LSTM(input_size=embed_size,
                             hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)

    def forward(self, partial_captions):
        # TODO: check this better
        embeddings = self.embed(partial_captions)
        encoded_texts, _ = self.lstm1(embeddings)
        encoded_texts, _ = self.lstm2(encoded_texts)
        return encoded_texts


class P2cDecoder(nn.Module):
    def __init__(self, output_size):
        super(P2cDecoder, self).__init__()

        self.lstm1 = nn.LSTM(input_size=1024+128,
                             hidden_size=512, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, num_layers=1, batch_first=True)
        self.linear = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoded_images, encoded_texts):
        repeated_images = encoded_images.unsqueeze(
            1).repeat(1, encoded_texts.size(1), 1)
        decoder_input = torch.cat((repeated_images, encoded_texts), dim=2)
        output, _ = self.lstm1(decoder_input)
        output, _ = self.lstm2(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output
