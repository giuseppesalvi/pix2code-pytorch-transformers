import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import Pix2CodeDataset
from utils import collate_fn, save_model, resnet_img_transformation, original_pix2code_transformation
from models import Encoder, Decoder, DecoderTransformer, P2cVisionModel, P2cLanguageModel, P2cDecoder
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument("--data_path", type=str, default=Path("data", "web", "all_data"), help="Path to the dataset")
    parser.add_argument("--vocab_file_path", type=str, default=None, help="Path to the vocab file")
    parser.add_argument("--cuda", action='store_true', default=False, help="Use cuda or not")
    parser.add_argument("--mps", action='store_true', default=False, help="Use mps or not")
    parser.add_argument("--img_crop_size", type=int, default=224)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_after_epochs", type=int, default=1, help="Save model checkpoint every n epochs")
    parser.add_argument("--models_dir", type=str, default=Path("models"), help="The dir where the trained models are saved")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--print_freq", type=int, default=1, help="Print training stats every n epochs")
    parser.add_argument("--seed", type=int, default=2020, help="The random seed for reproducing")
    parser.add_argument("--model", type=str, default="version2", help="Choose the model to use", choices=["lstm", "transformer", "pix2code"])

    args = parser.parse_args()
    args.vocab_file_path = args.vocab_file_path if args.vocab_file_path else Path(Path(args.data_path).parent, "vocab.txt")

    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.mps:
        torch.mps.manual_seed(args.seed)

    # Load the vocab file
    vocab = Vocab(args.vocab_file_path)
    assert len(vocab) > 0

    # Setup GPU
    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    use_mps = True if args.mps and torch.has_mps else False
    assert use_cuda or use_mps # Trust me, you don't want to train this model on a cpu.
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    if args.model == "pix2code":
        transform_imgs = original_pix2code_transformation()
    else:
        transform_imgs = resnet_img_transformation(args.img_crop_size)

    # Creating the data loader
    train_dataloader = DataLoader(
        Pix2CodeDataset(args.data_path, "train", vocab, transform=transform_imgs),
        batch_size=args.batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=True if use_cuda or use_mps else False,
        num_workers=0,
        drop_last=True)

    # Creating the data loader
    eval_dataloader = DataLoader(
        Pix2CodeDataset(args.data_path, "validation", vocab, transform=transform_imgs),
        batch_size=args.batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=True if use_cuda or use_mps else False,
        num_workers=0,
        drop_last=True)
    print("Created data loaders")
    lr = args.lr
    
    if args.model == "lstm":
        # Creating the models
        embed_size = 256
        hidden_size = 512
        num_layers = 1
        
        encoder = Encoder(embed_size).to(device)
        decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)
     
        # Define optimizer and loss function
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.BatchNorm.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

    elif args.model == "pix2code":
        embed_size = 256

        vision_model =  P2cVisionModel().to(device)
        language_model = P2cLanguageModel(embed_size, len(vocab)).to(device)
        decoder = P2cDecoder(len(vocab)).to(device)

        params = list(vision_model.parameters()) + list(language_model.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

    elif args.model == "transformer":
        embed_size = 256
        hidden_size = 512
        num_layers = 3
        num_heads = 8
        encoder = Encoder(embed_size).to(device)
        decoder = DecoderTransformer(embed_size, hidden_size, len(vocab), num_layers, num_heads).to(device)

        # Define optimizer and loss function
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.BatchNorm.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

    # Log start date and time
    start = datetime.datetime.now()
    print("Start Training date and time: {}".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    # Training the model
    for epoch in range(1, args.epochs):
        # Training Loop
        if args.model == "lstm" or args.model == "transformer":
            encoder.train()
            decoder.train()
        elif args.model == "pix2code":
            vision_model.train()
            language_model.train()
            decoder.train()

        for i, (images, captions, lengths) in enumerate(train_dataloader):
            images = images.to(device)
            captions = captions.to(device)

            if args.model == "lstm":
                targets = torch.nn.utils.rnn.pack_padded_sequence(input=captions, lengths=lengths, batch_first=True)[0]
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
            elif args.model == "pix2code":
                encoded_images = vision_model(images)
                partial_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                new_lengths = [length - 1 for length in lengths]  # Update the lengths list
                encoded_texts = language_model(partial_captions)
                outputs = decoder(encoded_images, encoded_texts)
                target_captions_packed = torch.nn.utils.rnn.pack_padded_sequence(target_captions, new_lengths, batch_first=True, enforce_sorted=False).data
                outputs_packed = torch.nn.utils.rnn.pack_padded_sequence(outputs, new_lengths, batch_first=True, enforce_sorted=False).data
                loss = criterion(outputs_packed, target_captions_packed)

            elif args.model == "transformer":
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].flatten())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        #encoder.eval()
        #decoder.eval()
        #val_losses = []
        #bleu_scores = []

        #with torch.no_grad():
            #for batch_idx, (images, captions, lengths) in enumerate(eval_dataloader):
                #images = images.to(device)
                #captions = captions.to(device)
#
                # Generate captions
                #features = encoder(images)
                #start_token_id = vocab.get_id_by_token(vocab.get_start_token)
                #end_token_id = vocab.get_id_by_token(vocab.get_end_token)
                #padding_token_id = vocab.get_id_by_token(vocab.get_padding_token)
                #generated_captions = decoder.sample_greedy(features, start_token_id, end_token_id)

                # Calculate validation loss
                #val_outputs = decoder(features, captions[:, :-1])
                #val_loss = criterion(val_outputs.view(-1, len(vocab)), captions[:, 1:].flatten())
                #val_losses.append(val_loss.item())

                # Calculate BLEU score
                #smooth_func = SmoothingFunction().method4
                #for gen_caption, gt_caption in zip(generated_captions, captions):
                    #gen_caption_text = [vocab.get_token_by_id[word_id] for word_id in gen_caption.cpu().numpy()]
                    #gt_caption_text = [vocab.get_token_by_id[word_id] for word_id in gt_caption.cpu().numpy()]
#
                    ## Remove start, end and padding tokens from the generated and ground truth captions
                    #list_start_end_padding_tokens = [vocab.get_start_token, vocab.get_end_token, vocab.get_padding_token]
                    #gen_caption_text = [word for word in gen_caption_text if word not in list_start_end_padding_tokens]
                    #gt_caption_text = [word for word in gt_caption_text if word not in list_start_end_padding_tokens]
#
                    #bleu_score = sentence_bleu([gt_caption_text], gen_caption_text, smoothing_function=smooth_func)
                    #bleu_scores.append(bleu_score)
#
        #avg_val_loss = np.mean(val_losses)
        #avg_bleu_score = np.mean(bleu_scores)

        #print(f"Epoch: {epoch}/{args.epochs}, Loss: {loss.item()}, Val Loss: {avg_val_loss}, Val BLEU Score: {avg_bleu_score}")
        print(f"Epoch: {epoch}/{args.epochs}, Loss: {loss.item()}")
        if epoch != 0 and epoch % args.save_after_epochs == 0:
            if args.model == "pix2code":
                save_model(args.models_dir, [vision_model, language_model, decoder], optimizer, epoch, loss, args.batch_size, vocab, args.model)
            else:
                save_model(args.models_dir, [encoder, decoder], optimizer, epoch, loss, args.batch_size, vocab, args.model)

            print("Saved model checkpoint")

    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Training date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time%3600)//60, elapsed_time%60))

    save_model(args.models_dir, encoder, decoder, optimizer, epoch, loss, args.batch_size, vocab, args.model)
    print("Saved final model")


