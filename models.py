import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils import save_model, NoamOpt

def configure_model(model_type, vocab, device, config):
    if model_type == "lstm":
        lr= config["lr"] if config["lr"] else 1e-5 
        embed_size = config["embed_size"] if config["embed_size"] else 256
        hidden_size = config["hidden_size"] if config["hidden_size"] else 512
        num_layers = config["num_layers"] if config["num_layers"] else 1
        return LSTMModel(embed_size, hidden_size, vocab, num_layers, device, lr)

    elif model_type == "pix2code":
        lr= config["lr"] if config["lr"] else 1e-5 
        embed_size = config["embed_size"] if config["embed_size"] else 256
        return Pix2codeModel(embed_size, vocab, device, lr)

    elif model_type == "transformer":
        #lr= config["lr"] if config["lr"] else 1e-5 
        lr = 1e-5
        embed_size = config["embed_size"] if config["embed_size"] else 256
        hidden_size = config["hidden_size"] if config["hidden_size"] else 512
        num_layers = config["num_layers"] if config["num_layers"] else 3
        #num_heads = config["num_heads"] if config["num_heads"] else 8
        num_heads = 8
        num_warmups = config["num_warmups"] if config["num_warmups"] else 1000
        optim_params_separated = config["optim_params_separated"] if config["optim_params_separated"] else True
        return TransformerModel(embed_size, hidden_size, vocab, num_layers, device, lr, num_heads, num_warmups=num_warmups, optim_params_separated=optim_params_separated)

class Model:
    def __init__(self, embed_size, device, lr, vocab, label_smoothing=0):
        self.embed_size = embed_size
        self.device = device
        self.lr = lr
        self.vocab = vocab
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing)


class LSTMModel(Model):
    def __init__(self, embed_size, hidden_size, vocab, num_layers, device, lr):
        super().__init__(embed_size, device, lr, vocab)
        self.model_type = "lstm"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = Encoder(embed_size).to(device)
        self.decoder = Decoder(embed_size, hidden_size,
                               len(vocab), num_layers).to(device)
        self.params = list(self.decoder.parameters(
        )) + list(self.encoder.linear.parameters()) + list(self.encoder.BatchNorm.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def forward_prop_calc_loss(self, images, captions, lengths):
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            input=captions, lengths=lengths, batch_first=True)[0]
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = self.criterion(outputs, targets)
        return loss

    def generate_captions(self, images):
        features = self.encoder(images)
        predicted_ids = self.decoder.sample(features)
        return predicted_ids.cpu().data.numpy()

    def save(self, dir, epoch, loss, bleu, batch_size):
        save_model(dir, [self.encoder, self.decoder], self.optimizer,
                   epoch, loss.item(), bleu, batch_size, self.vocab, self.model_type, self.lr)
    
    def step_optimizer_or_scheduler(self):
        self.optimizer.step()

    def load_weights(self, loaded_model):
        self.encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
        self.decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

class Pix2codeModel(Model):
    def __init__(self, embed_size, vocab, device, lr):
        super().__init__(embed_size, device, lr, vocab)
        self.model_type = "pix2code"
        self.vision_model = P2cVisionModel().to(device)
        self.language_model = P2cLanguageModel(
            embed_size, len(vocab)).to(device)
        self.decoder = P2cDecoder(len(vocab)).to(device)
        self.params = list(self.vision_model.parameters(
        )) + list(self.language_model.parameters()) + list(self.decoder.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def eval(self):
        self.vision_model.eval()
        self.language_model.eval()
        self.decoder.eval()

    def train(self):
        self.vision_model.train()
        self.language_model.train()
        self.decoder.train()

    def forward_prop_calc_loss(self, images, captions, lengths):
        encoded_images = self.vision_model(images)
        partial_captions = captions[:, :-1]
        target_captions = captions[:, 1:]
        # Update the lengths list
        new_lengths = [length - 1 for length in lengths]
        encoded_texts = self.language_model(partial_captions)
        outputs = self.decoder(encoded_images, encoded_texts)
        target_captions_packed = torch.nn.utils.rnn.pack_padded_sequence(
            target_captions, new_lengths, batch_first=True, enforce_sorted=False).data
        outputs_packed = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, new_lengths, batch_first=True, enforce_sorted=False).data
        loss = self.criterion(outputs_packed, target_captions_packed)
        return loss

    def generate_captions(self, images):
        # TODO: not implemeted yet!!
        pass

    def save(self, dir, epoch, loss, bleu, batch_size):
        save_model(dir, [self.vision_model, self.language_model, self.decoder],
                   self.optimizer, epoch, loss.item(), bleu, batch_size, self.vocab, self.model_type, self.lr)

    def step_optimizer_or_scheduler(self):
        self.optimizer.step()

    def load_weights(self, loaded_model):
        self.vision_model.load_state_dict(loaded_model["vision_model_state_dict"])
        self.language_model.load_state_dict(loaded_model["language_model_state_dict"])
        self.decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

class TransformerModel(Model):
    def __init__(self, embed_size, hidden_size, vocab, num_layers, device, lr, num_heads, num_warmups, optim_params_separated=False):
        super().__init__(embed_size, device, lr, vocab, label_smoothing=0.1)
        self.model_type = "transformer"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.encoder = Encoder(embed_size).to(device)
        self.decoder = DecoderTransformer(embed_size, hidden_size, len(vocab), num_layers, num_heads).to(device)
        self.num_warmups = num_warmups

        # All parameters together
        self.params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.BatchNorm.parameters())

        # Parameters in two separated groups, one for decoder, one for encoder
        decoder_params = list(self.decoder.parameters())
        encoder_params = list(self.encoder.linear.parameters()) + list(self.encoder.BatchNorm.parameters())

        if optim_params_separated:
            # Transformer decoder and Encoder params separated, warmup applied only on transformer
            self.optimizer = torch.optim.Adam([
                {'params': decoder_params, 'lr': 0, 'betas': (0.9, 0.98), 'eps': 1e-9},
                {'params': encoder_params, 'lr': lr}
            ])
        else:
            # All parameters together 
            self.params = decoder_params + encoder_params 
            self.optimizer = torch.optim.Adam(self.params, lr=0, betas=(0.9, 0.98), eps=1e-9)

        noam_opt_factor = 2
        self.scheduler = NoamOpt(self.embed_size, noam_opt_factor, self.num_warmups, self.optimizer)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward_prop_calc_loss(self, images, captions, lengths):
        features = self.encoder(images)
        captions_input = captions[:, :-1]
        captions_expected = captions[:, 1:]

        # add tgt mask
        tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(
            captions_input.size(1)).to(self.device)
        # add padding mask, real length are in lengths
        tgt_pad_mask = (captions_input == self.vocab.get_id_by_token(
            self.vocab.get_padding_token()))

        outputs = self.decoder(features, captions_input,
                               tgt_mask, tgt_pad_mask)
        loss = self.criterion(
            outputs.view(-1, len(self.vocab)), captions_expected.flatten())
        return loss

    def generate_captions(self, images):
        features = self.encoder(images)

        start_token_id = self.vocab.get_id_by_token(
            self.vocab.get_start_token())
        end_token_id = self.vocab.get_id_by_token(self.vocab.get_end_token())
        padding_token_id = self.vocab.get_id_by_token(
            self.vocab.get_padding_token())

        predicted_ids = self.decoder.greedy_search(
            features, start_token_id, end_token_id, padding_token_id)
        return predicted_ids.cpu().data.numpy()

    def save(self, dir, epoch, loss, bleu, batch_size):
        save_model(dir, [self.encoder, self.decoder], self.optimizer,
                   epoch, loss.item(), bleu, batch_size, self.vocab, self.model_type, self.lr)

    def step_optimizer_or_scheduler(self):
        self.scheduler.step()  

    def load_weights(self, loaded_model):
        self.encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
        self.decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
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

        transformer_decoder_layer = TransformerDecoderLayer(embed_size, nhead, hidden_size, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(transformer_decoder_layer, num_layers)

        self.linear = nn.Linear(in_features=embed_size,out_features=vocab_size)

    def forward(self, features, tgt, tgt_mask=None, tgt_pad_mask=None):
        target_seq_len = tgt.size(1)
        features = self.preprocess_encoder_features(features, target_seq_len)

        tgt = self.embed(tgt)
        tgt = self.pos_enc(tgt)
        output = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.linear(output)
        return output

    def preprocess_encoder_features(self, features, target_seq_len):
        features = features.unsqueeze(1)
        features = features.repeat(1, target_seq_len, 1)
        return features

    def greedy_search(self, features, start_token_id, end_token_id, padding_token_id, max_len=100):
        batch_size = features.size(0)
        generated_sequences = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=features.device)

        # Mask for ended sequences
        end_mask = torch.full((batch_size,), False,
                              dtype=torch.bool, device=features.device)

        for _ in range(max_len):
            output = self.forward(features, generated_sequences)

            # Replace probabilities for ended sequences with 1 for pad token and 0 for other tokens
            output[end_mask] = torch.eye(
                output.size(-1), device=output.device)[padding_token_id]

            _, next_words = torch.max(output[:, -1, :], dim=-1)
            next_words = next_words.unsqueeze(1)
            generated_sequences = torch.cat(
                (generated_sequences, next_words), dim=1)

            # Update end_mask
            new_end_mask = (next_words.squeeze(-1) == end_token_id)
            end_mask = end_mask | new_end_mask

            # Check if all sequences have reached the end_token_id
            if torch.all(end_mask):
                break

        # Remove the initial start_token_id from the generated sequences
        return generated_sequences[:, 1:]

    def beam_search(self, features, start_token_id, end_token_id, padding_token_id, max_len=100, beam_size=3):

        batch_size = features.size(0)

        # Beam options will be processed, as an extension of batch size
        generated_sequences = torch.full(
            (batch_size * beam_size, 1), start_token_id, dtype=torch.long, device=features.device)
        features_extended = features.repeat(beam_size, 1)

        # Create a tensor to store the scores of each sequence
        sequence_scores = torch.zeros(
            batch_size * beam_size, 1).to(features.device)

        # Mask for ended sequences
        end_mask = torch.full((batch_size * beam_size,),
                              False, dtype=torch.bool, device=features.device)

        first_time = True
        for _ in range(max_len):
            # Get the output probabilities for the current step
            output = self.forward(features_extended, generated_sequences)
            output_probs = torch.softmax(output[:, -1, :], dim=-1)

            # Replace probabilities for ended sequences with 1 for end token and 0 for other tokens
            output_probs[end_mask] = torch.eye(
                output_probs.size(-1), device=features.device)[padding_token_id]

            # Multiply the current step's probabilities with the previous steps' accumulated scores (using broadcasting)
            scores = sequence_scores + torch.log(output_probs)

            if first_time:
                scores_together = scores[:batch_size]
                first_time = False
            else:
                # Now put the results for the same beam together to find the top ones
                scores_together = scores.view(batch_size, -1)

            # Reshape the scores to get the top k candidates (beam_size) for each sequence in the batch
            top_k_scores, top_k_indices = torch.topk(
                scores_together, k=beam_size, dim=1)

            top_k_indices_corrected = top_k_indices % output_probs.size(1)
            top_k_indices_starting_sequences = top_k_indices // output_probs.size(
                1)

            sequence_indices = torch.squeeze(
                top_k_indices_starting_sequences, dim=0)
            starting_sequences = generated_sequences.index_select(
                dim=0, index=sequence_indices)

            top_k_indices_corrected = top_k_indices_corrected.view(
                batch_size * beam_size, -1)
            top_k_scores = top_k_scores.view(batch_size * beam_size, -1)

            generated_sequences = torch.concat(
                (starting_sequences, top_k_indices_corrected), dim=1)

            # Update end_mask
            end_mask = (top_k_indices_corrected == end_token_id).squeeze(-1)

            # Update sequence scores only for sequences that did not reach end
            sequence_scores = sequence_scores.masked_scatter(
                ~end_mask.unsqueeze(-1), top_k_scores)
            # Check if all sequences have reached the end_token_id
            if torch.all(end_mask):
                break

        scores_reshaped = sequence_scores.view(batch_size, beam_size, -1)
        sequences_reshaped = generated_sequences.view(
            batch_size, beam_size, -1)

        _, best_indices = scores_reshaped.max(dim=1)
        best_sequences = sequences_reshaped[torch.arange(
            batch_size), best_indices.squeeze(-1)]
        return best_sequences

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
