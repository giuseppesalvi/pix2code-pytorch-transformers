import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import Pix2CodeDataset
from utils import collate_fn, save_model, resnet_img_transformation
from models import Encoder, Decoder, DecoderTransformer
from modelsPix2Code import VisionModel, LanguageModel, Decoder
import math
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
    parser.add_argument("--model", type=str, default="version2", help="Choose the model to use", choices=["lstm", "transformer"])

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

    transform_imgs = resnet_img_transformation(args.img_crop_size)

    # Creating the data loader
    train_loader = DataLoader(
        Pix2CodeDataset(args.data_path, args.split, vocab, transform=transform_imgs),
        batch_size=args.batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=True if use_cuda or use_mps else False,
        num_workers=0,
        drop_last=True)
    print("Created data loader")

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
        encoder.train()
        decoder.train()
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            if args.model == "lstm":
                targets = torch.nn.utils.rnn.pack_padded_sequence(input=captions, lengths=lengths, batch_first=True)[0]
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)

            elif args.model == "transformer":
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].flatten())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if epoch % args.print_freq == 0 and i == 0: print(f'Epoch : {epoch} || Loss : {loss:.4f} || Perplexity : {math.exp(loss):.4f}')

            if epoch != 0 and epoch % args.save_after_epochs == 0 and i % len(train_loader) == 0:
                save_model(args.models_dir, encoder, decoder, optimizer, epoch, loss, args.batch_size, vocab, args.model)
                print("Saved model checkpoint")

    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Training date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time%3600)//60, elapsed_time%60))

    save_model(args.models_dir, encoder, decoder, optimizer, epoch, loss, args.batch_size, vocab, args.model)
    print("Saved final model")
