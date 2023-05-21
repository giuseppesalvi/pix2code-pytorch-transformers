import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import Pix2CodeDataset
from utils import collate_fn, save_model, resnet_img_transformation, original_pix2code_transformation, ids_to_tokens
from models import TransformerModel, LSTMModel, Pix2codeModel
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument("--data_path", type=str, default=Path("data","web", "all_data"), help="Path to the dataset")
    parser.add_argument("--vocab_file_path", type=str,default=None, help="Path to the vocab file")
    parser.add_argument("--cuda", action='store_true',default=False, help="Use cuda or not")
    parser.add_argument("--mps", action='store_true',default=False, help="Use mps or not")
    parser.add_argument("--img_crop_size", type=int, default=224)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_after_epochs", type=int,default=1, help="Save model checkpoint every n epochs")
    parser.add_argument("--models_dir", type=str, default=Path("models"), help="The dir where the trained models are saved")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--print_freq", type=int, default=1, help="Print training stats every n epochs")
    parser.add_argument("--seed", type=int, default=2020, help="The random seed for reproducing")
    parser.add_argument("--model_type", type=str, default="lstm", help="Choose the model to use", choices=["lstm", "transformer", "pix2code"])
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--lr_patience", type=int, default=3)

    args = parser.parse_args()
    print("Training args:", args)
    return args


def configure_transform_imgs(model_type, img_crop_size):
    if model_type == "pix2code":
        return original_pix2code_transformation()
    else:
        return resnet_img_transformation(img_crop_size)


def setup_gpu(cuda, mps):
    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    use_mps = True if args.mps and torch.has_mps else False
    # Trust me, you don't want to train this model on a cpu.
    assert use_cuda or use_mps
    return torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu"), use_cuda, use_mps


def load_vocab(vocab_file_path, data_path):
    vocab_file_path = vocab_file_path if vocab_file_path else Path(
        Path(data_path).parent, "vocab.txt")
    vocab = Vocab(vocab_file_path)
    assert len(vocab) > 0
    return vocab


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.mps:
        torch.mps.manual_seed(seed)
    return


def configure_model(model_type, vocab, device, lr):
    if model_type == "lstm":
        embed_size = 256
        hidden_size = 512
        num_layers = 1
        return LSTMModel(embed_size, hidden_size, vocab, num_layers, device, lr)

    elif model_type == "pix2code":
        embed_size = 256
        return Pix2codeModel(embed_size, vocab, device, lr)

    elif model_type == "transformer":
        embed_size = 256
        hidden_size = 512
        num_layers = 6
        num_heads = 8
        return TransformerModel(embed_size, hidden_size, vocab, num_layers, device, lr, num_heads)


def create_data_loaders(data_path, vocab, transform_imgs, batch_size, pin_memory):
    # Creating the data loader
    train_dataloader = DataLoader(
        Pix2CodeDataset(data_path, "train", vocab, transform=transform_imgs),
        batch_size=batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=True)

    # Creating the data loader
    valid_dataloader = DataLoader(
        Pix2CodeDataset(data_path, "validation", vocab,
                        transform=transform_imgs),
        batch_size=batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=True)

    print("Created data loaders")
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    # Configure training
    args = parse_args()

    set_seed(args.seed)
    device, use_cuda, use_mps = setup_gpu(args.cuda, args.mps)

    vocab = load_vocab(args.vocab_file_path, args.data_path)

    train_dataloader, valid_dataloader = create_data_loaders(args.data_path, vocab, configure_transform_imgs(
        args.model_type, args.img_crop_size), args.batch_size, use_cuda or use_mps)

    lr = args.lr

    model = configure_model(args.model_type, vocab, device, lr)

    # Log start date and time
    start = datetime.datetime.now()
    print("Start Training date and time: {}".format(
        start.strftime("%Y-%m-%d %H:%M:%S")))

    total_steps = len(train_dataloader) * args.epochs

    # Create a scheduler TODO!!!
    #max_lr = lr * 100
    #print("lr=", lr)
    #print("max_lr=", max_lr)
    # TODOscheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, anneal_strategy='linear')
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.lr_patience, verbose=True)

    # Initialize variables for early stopping
    epochs_without_improvement = 0
    best_bleu_score = 0

    # Training the model
    for epoch in range(1, args.epochs + 1):
        # Training Loop
        model.train()

        train_loop = tqdm(enumerate(train_dataloader), total=len(
            train_dataloader), desc=f"Epoch {epoch}/{args.epochs} - train loop")
        for i, (images, captions, lengths) in train_loop:
            images = images.to(device)
            captions = captions.to(device)

            loss = model.forward_prop_calc_loss(images, captions, lengths)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # Update the progress bar
            train_loop.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        bleu_scores = []

        with torch.no_grad():
            eval_loop = tqdm(enumerate(valid_dataloader), total=len(
                valid_dataloader), desc=f"Epoch {epoch}/{args.epochs} - valid loop")
            for i, (images, captions, lengths) in eval_loop:
                images = images.to(device)
                captions = captions.to(device)

                generated_caption_ids = model.generate_captions(images)

                gen_caption_text = [[vocab.get_token_by_id(
                    word_id) for word_id in batch] for batch in generated_caption_ids]
                gt_caption_text = [[vocab.get_token_by_id(
                    word_id) for word_id in batch] for batch in captions.tolist()]

                # Remove start, end and padding tokens from the generated and ground truth captions
                list_start_end_padding_tokens = [vocab.get_start_token(
                ), vocab.get_end_token(), vocab.get_padding_token(), ","]
                gen_caption_text = [
                    [word for word in batch if word not in list_start_end_padding_tokens] for batch in gen_caption_text]
                gt_caption_text = [
                    [word for word in batch if word not in list_start_end_padding_tokens] for batch in gt_caption_text]

                for gt_caption, gen_caption in zip(gt_caption_text, gen_caption_text):
                    #bleu_score = sentence_bleu([gt_caption], gen_caption, smoothing_function=smooth_func)
                    bleu_score = corpus_bleu(
                        [[gt_caption]], [gen_caption], smoothing_function=SmoothingFunction().method4)

                    bleu_scores.append(bleu_score)

                # Update the progress bar
                eval_loop.set_postfix(val_bleu_score=np.mean(bleu_scores))

        avg_bleu_score = np.mean(bleu_scores)

        # Update the learning rate based on validation BLEU score
        # scheduler.step(avg_bleu_score)
        # TODO
        # scheduler.step()

        # Early stopping condition
        if avg_bleu_score > best_bleu_score:
            best_bleu_score = avg_bleu_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= 15 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Stopping training early due to lack of improvement in validation BLEU score.")
            break

        print("Epoch: {}/{}, Loss: {:.3f}, Val BLEU Score: {:03f}".format(epoch,
              args.epochs, loss.item(), avg_bleu_score))
        if epoch != 0 and epoch % args.save_after_epochs == 0:
            model.save(args.models_dir, epoch, loss, args.batch_size)
            print("Saved model checkpoint")

    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Training date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime(
        "%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time % 3600)//60, elapsed_time % 60))

    model.save(args.models_dir, epoch, loss, args.batch_size)
    print("Saved final model")
