import argparse
from pathlib import Path
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import Pix2CodeDataset
from utils import collate_fn, generate_visualization_object, resnet_img_transformation, original_pix2code_transformation
from models import configure_model 
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import numpy as np
import datetime


def parse_args():
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
    parser.add_argument("--model_type", type=str, default="transformer", help="Choose the model to use", choices=["lstm", "transformer", "pix2code"])
    parser.add_argument("--beam_search", action='store_true', default=False, help="Use beam search instead of greedy search")
    parser.add_argument("--beam_size", type=int, default=4)

    args = parser.parse_args()
    print("Evaluation args:", args)

def setup_gpu(cuda, mps):
    use_cuda = True if cuda and torch.cuda.is_available() else False
    use_mps = True if mps and torch.has_mps else False
    # Trust me, you don't want to train this model on a cpu.
    assert use_cuda or use_mps
    return torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu"), use_cuda, use_mps

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.mps:
        torch.mps.manual_seed(seed)
    return

def load_vocab(model_file_path, loaded_model):
    assert Path(model_file_path).exists()
    vocab = loaded_model["vocab"]
    return vocab

def configure_transform_imgs(model_type, img_crop_size):
    if model_type == "pix2code":
        return original_pix2code_transformation()
    else:
        return resnet_img_transformation(img_crop_size)


def create_data_loader(data_path, vocab, transform_imgs, batch_size, pin_memory):
#    test_dataloader = DataLoader(
        #Pix2CodeDataset(data_path, "test", vocab, transform=transform_imgs),
        #batch_size=batch_size,
        #collate_fn=collate_fn,
        #pin_memory=pin_memory,
        #num_workers=0,
        #drop_last=True)
    #print("Created data loader")
    #return test_dataloader

    test_dataloader = DataLoader(
        Pix2CodeDataset(data_path, "test", vocab,
                        transform=transform_imgs),
        batch_size=batch_size,
        collate_fn=lambda data: collate_fn(data, vocab=vocab),
        pin_memory=pin_memory,
        num_workers=0,
        drop_last=True)
    return test_dataloader



if __name__ == "__main__":

    # Configure training
    args = parse_args()

    set_seed(args.seed)
    device, use_cuda, use_mps = setup_gpu(args.cuda, args.mps)

    loaded_model = torch.load(args.model_file_path)
    vocab = load_vocab(args.model_file_path)

    test_dataloader = create_data_loader(args.data_path, vocab, configure_transform_imgs(
        args.model_type, args.img_crop_size), args.batch_size, use_cuda or use_mps)

    model = configure_model(args.model_type, vocab, device, 0)
    model.load_weights(loaded_model)

    # Evaluate the model
    model.eval()
    bleu_scores = []

    #predictions = []
    #targets = []

    # Log start date and time
    start = datetime.datetime.now()
    print("Start Evaluation date and time: {}".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    test_loop = tqdm(enumerate(test_dataloader), total=len(
                test_dataloader), desc=f"Testing:")

    #for i, (image, caption) in enumerate(tqdm(test_dataloader.dataset)):
    for i, (images, captions, lengths) in test_loop:
        image = image.to(device)
        caption = caption.to(device)


        generated_captions_ids = model.generate_captions(image)

        gen_caption_text = [[vocab.get_token_by_id(word_id) for word_id in batch] for batch in generated_caption_ids]
        gt_caption_text = [[vocab.get_token_by_id(word_id) for word_id in batch] for batch in captions.tolist()]

        # Remove start, end and padding tokens from the generated and ground truth captions
        list_start_end_padding_tokens = [vocab.get_start_token(), vocab.get_end_token(), vocab.get_padding_token(), ","]
        gen_caption_text = [[word for word in batch if word not in list_start_end_padding_tokens] for batch in gen_caption_text]
        gt_caption_text = [[word for word in batch if word not in list_start_end_padding_tokens] for batch in gt_caption_text]

        for gt_caption, gen_caption in zip(gt_caption_text, gen_caption_text):
            #bleu_score = sentence_bleu([gt_caption], gen_caption, smoothing_function=smooth_func)
            bleu_score = corpus_bleu([[gt_caption]], [gen_caption], smoothing_function=SmoothingFunction().method4)

            bleu_scores.append(bleu_score)

        # Update the progress bar
        test_loop.set_postfix(val_bleu_score=np.mean(bleu_scores))

    avg_bleu_score = np.mean(bleu_scores)

        #features = encoder(image.unsqueeze(0))

        #start_token_id = vocab.get_id_by_token(vocab.get_start_token())
        #end_token_id = vocab.get_id_by_token(vocab.get_end_token())
        #padding_token_id = vocab.get_id_by_token(vocab.get_padding_token())
                #
        #if args.model == "lstm":
            #generated_caption_ids = decoder.sample(features)
        #elif args.model == "transformer":
            #if args.beam_search:
                #generated_caption_ids = decoder.beam_search(features, start_token_id, end_token_id, padding_token_id, beam_size=args.beam_size)
            #else:
                #generated_caption_ids = decoder.greedy_search(features, start_token_id, end_token_id, padding_token_id)
        #
        #generated_caption_ids= generated_caption_ids[0].cpu().data.numpy()
        #predictions.append(generated_caption_ids)
        #targets.append(caption.cpu().data.numpy())


    # Log end date and time elapsed time from start
    end = datetime.datetime.now()
    elapsed_time = (end - start).seconds
    print("End Evaluation date and time: {}, elapsed time  : {:02d}:{:02d}:{:02d}".format(end.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time//3600, (elapsed_time%3600)//60, elapsed_time%60))

    #predictions = [ids_to_tokens(vocab, prediction) for prediction in predictions]
    #targets = [ids_to_tokens(vocab, target) for target in targets]

    # DBG
    #with open("predictions_beam_4.txt", "w") as f:
        #for pred in predictions:
            #print(pred, file=f)
            #print("\n", file=f)
            
    #bleu_scores = []
    #for gt_caption, gen_caption in zip(targets, predictions):
        #bleu_score = sentence_bleu([gt_caption], gen_caption, smoothing_function=SmoothingFunction().method4)
        #bleu_scores.append(bleu_score)
#
    print("Bleu: {}".format(np.mean(bleu_scores)))

    if args.viz:
        generate_visualization_object(test_dataloader.dataset, gen_caption_text, gt_caption_text)
        print("generated visualisation object")
