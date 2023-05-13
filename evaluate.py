import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Pix2CodeDataset
from utils import collate_fn, save_model, ids_to_tokens, generate_visualization_object, resnet_img_transformation, original_pix2code_transformation
from models import Encoder, Decoder, DecoderTransformer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from tqdm import tqdm
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')

    parser.add_argument("--model_file_path", type=str, help="Path to the trained model file", required=True)
    parser.add_argument("--data_path", type=str, default=Path("data", "web", "all_data"), help="Datapath")
    parser.add_argument("--cuda", action='store_true', default=False, help="Use cuda or not")
    parser.add_argument("--mps", action='store_true', default=False, help="Use mps or not")
    parser.add_argument("--img_crop_size", type=int, default=224)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--viz", action='store_true', default=False,)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2020, help="The random seed for reproducing ")
    parser.add_argument("--model", type=str, default="version2", help="Choose the model to use", choices=["lstm", "transformer", "pix2code"])

    args = parser.parse_args()
    print("Evaluation args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.mps:
        torch.mps.manual_seed(args.seed)

    # Load the vocab file
    assert Path(args.model_file_path).exists()
    loaded_model = torch.load(args.model_file_path)
    vocab = loaded_model["vocab"]

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
    test_dataloader = DataLoader(
        Pix2CodeDataset(args.data_path, args.split, vocab, transform=transform_imgs),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True if use_cuda or use_mps else False,
        num_workers=0,
        drop_last=True)

    # Creating the data loader
    #test_dataloader = DataLoader(
        #Pix2CodeDataset(args.data_path, "test", vocab, transform=transform_imgs),
        #batch_size=args.batch_size,
        #collate_fn=lambda data: collate_fn(data, vocab=vocab),
        #pin_memory=True if use_cuda or use_mps else False,
        #num_workers=0,
        #drop_last=True)


    if args.model == "lstm":
        # Loading the model
        embed_size = 256
        hidden_size = 512
        num_layers = 1

        encoder = Encoder(embed_size)
        decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)

        encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
        decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

        encoder.to(device)
        decoder.to(device)
    elif args.model == "pix2code":
        # TODO
        pass

    elif args.model == "transformer":
        embed_size = 256
        hidden_size = 512
        num_layers = 3
        num_heads = 8

        encoder = Encoder(embed_size).to(device)
        decoder = DecoderTransformer(embed_size, hidden_size, len(vocab), num_layers, num_heads).to(device)

        encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
        decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

        encoder.to(device)
        decoder.to(device)

    # Evaluate the model
    encoder.eval()
    decoder.eval()

    predictions = []
    targets = []

    # Log start date and time
    start = datetime.datetime.now()
    print("Start Evaluation date and time: {}".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    for i, (image, caption) in enumerate(tqdm(test_dataloader.dataset)):
        image = image.to(device)
        caption = caption.to(device)

        features = encoder(image.unsqueeze(0))

        start_token_id = vocab.get_id_by_token(vocab.get_start_token())
        end_token_id = vocab.get_id_by_token(vocab.get_end_token())
        padding_token_id = vocab.get_id_by_token(vocab.get_padding_token())
                
        #sample_ids = decoder.sample(features)
        #sample_ids = sample_ids.cpu().data.numpy()
        
        if args.model == "lstm":
            generated_caption_ids = decoder.sample(features)
            generated_caption_ids= generated_caption_ids.cpu().data.numpy()
        elif args.model == "transformer":
            generated_caption_ids = decoder.greedy_search(features, start_token_id, end_token_id)
            generated_caption_ids= generated_caption_ids[0].cpu().data.numpy()

        predictions.append(generated_caption_ids)
        targets.append(caption.cpu().data.numpy())


    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Evaluation date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time%3600)//60, elapsed_time%60))


    #predictions = [ids_to_tokens(vocab, prediction) for prediction in predictions]
    #targets = [ids_to_tokens(vocab, target) for target in targets]
    predictions_as_text = []
    for prediction in predictions:
        predictions_as_text.append(ids_to_tokens(vocab, prediction))

    targets_as_text = []
    for target in targets:
        targets_as_text.append(ids_to_tokens(vocab, target))


    bleu = corpus_bleu([[target] for target in targets], predictions, smoothing_function=SmoothingFunction().method4)
    print("BLEU score: {}".format(bleu))

    if args.viz:
        generate_visualization_object(test_dataloader.dataset, predictions, targets)
        print("generated visualisation object")
