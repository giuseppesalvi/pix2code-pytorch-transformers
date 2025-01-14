import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import Pix2CodeDataset
from utils import collate_fn, resnet_img_transformation, original_pix2code_transformation
from models import configure_model 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import datetime
from tqdm import tqdm
import wandb


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
    parser.add_argument("--sweep", action='store_true',default=False, help="sweep to find hyperparameters")

    args = parser.parse_args()
    print("Training args:", args)
    return args


def configure_transform_imgs(model_type, img_crop_size):
    if model_type == "pix2code":
        return original_pix2code_transformation()
    else:
        return resnet_img_transformation(img_crop_size)


def setup_gpu(cuda, mps):
    use_cuda = True if cuda and torch.cuda.is_available() else False
    use_mps = True if mps and torch.has_mps else False
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


def create_data_loaders(data_path, vocab, transform_imgs, batch_size, pin_memory):
    train_dataloader = DataLoader(
        Pix2CodeDataset(data_path, "train", vocab, transform=transform_imgs),
        batch_size=batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=True)

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

def init_model_config(model_type, lr, batch_size):
    config = {}
    if model_type == "transformer":
        config["lr"] = lr 
        config["batch_size"] = batch_size 
        config["embed_size"] = 256
        config["hidden_size"] = 512
        config["num_layers"] = 2#3
        config["num_heads"] = 8
        config["num_warmups"] = 0 
        config["optim_params_separated"] = True
    else:
        # TODO
        pass

    return config

def init_sweep_config():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'bleu',
            'goal': 'maximize'   
        },
        'parameters': {
            'batch_size': {
                'values': [4, 8, 16]
            },
            'embed_size': {
                'values': [256]
            },
            'hidden_size': {
                'values': [256]
            },
            'num_layers': {
                'values': [2, 3, 6]
            },
            'num_warmups': {
                'values': [200, 500, 1000, 2000]
            },
            'optim_params_separated': {
                'values': [True, False]
            },
        }
    }
    return sweep_config


def train(args, device, vocab, model, train_dataloader, valid_dataloader):

    # Log start date and time
    start = datetime.datetime.now()
    print("Start Training date and time: {}".format(
        start.strftime("%Y-%m-%d %H:%M:%S")))

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

            model.step_optimizer_or_scheduler()

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

        # Early stopping condition
        if avg_bleu_score > best_bleu_score:
            best_bleu_score = avg_bleu_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        is_warmup = model.num_warmups and len(train_dataloader) * epoch < model.num_warmups
        if not is_warmup and epochs_without_improvement >= args.early_stopping_patience:
            print("Stopping training early due to lack of improvement in validation BLEU score.")
            break

        if epoch != 0 and epoch % args.save_after_epochs == 0:
            model.save(args.models_dir, epoch, loss, avg_bleu_score, args.batch_size)
            print("Saved model checkpoint")

        print("Epoch: {}/{}, Loss: {:.3f}, Bleu: {:03f}".format(epoch, args.epochs, loss.item(), avg_bleu_score))
        wandb.log({"loss": loss.item(), "bleu": avg_bleu_score, **{f'lr_{i}': param_group['lr'] for i, param_group in enumerate(model.optimizer.param_groups)}})



    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Training date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime(
        "%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time % 3600)//60, elapsed_time % 60))

    model.save(args.models_dir, epoch, loss, avg_bleu_score, args.batch_size)
    print("Saved final model")

def train_with_sweep():
    args = parse_args()

    wandb.init(project="Pix2Code_" + args.model_type)
    config = wandb.config
    set_seed(args.seed)
    device, use_cuda, use_mps = setup_gpu(args.cuda, args.mps)

    vocab = load_vocab(args.vocab_file_path, args.data_path)

    model = configure_model(args.model_type, vocab, device, config)
    args.batch_size = config["batch_size"]
    train_dataloader, valid_dataloader = create_data_loaders(args.data_path, vocab, configure_transform_imgs(args.model_type, args.img_crop_size), args.batch_size, use_cuda or use_mps)

    train(args, device, vocab, model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    args = parse_args()
    
    set_seed(args.seed)
    device, use_cuda, use_mps = setup_gpu(args.cuda, args.mps)

    vocab = load_vocab(args.vocab_file_path, args.data_path)

    if not args.sweep:
        # Single run
        config = init_model_config(args.model_type, args.lr, args.batch_size) 
        model = configure_model(args.model_type, vocab, device, config)
        run = wandb.init(project="Pix2Code_" + args.model_type, config=config)
        train_dataloader, valid_dataloader = create_data_loaders(args.data_path, vocab, configure_transform_imgs(args.model_type, args.img_crop_size), args.batch_size, use_cuda or use_mps)
        train(args, device, vocab, model, train_dataloader, valid_dataloader)
    else:
        # Sweep
        sweep_id = wandb.sweep(init_sweep_config(), project="Pix2Code_" + args.model_type)
        wandb.agent(sweep_id, function=train_with_sweep, count=10)

