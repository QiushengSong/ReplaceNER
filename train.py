import torch
import json
import numpy as np
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from dataset.CoNLL import MRCCoNLL
from torch.utils.data import DataLoader
from model import PTMambaNER
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from evaluator import Evaluate
from tqdm import tqdm
from mamba_ssm.models.config_mamba import MambaConfig
from datetime import datetime


def main(args):
    
    logging.basicConfig(filename="training.log", filemode="a",
                        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                        level=logging.DEBUG,
                        datefmt="%d-%M-%Y %H:%M:%S"
                        )
    logger = logging.getLogger(__name__)
    logger.info("Performing train.py!")
    train_data_path = "data/CoNLL2003/train.txt"
    valid_data_path = "data/CoNLL2003/valid.txt"
    loss_file = "loss.json"
    logger.info("Loading Configuration...")
    max_length = args.max_length
    device = 'cuda'
    dtype = torch.float32
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    checkpoint = OrderedDict()
    model_config = MambaConfig()
    logger.info("Completed.")
    logger.info("Loading model...")
    model = PTMambaNER(model_config, device=device, dtype=dtype)

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if not args.resume:
        torch.manual_seed(42)
    else:
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        rng_state = torch.load(checkpoint["rng_state_dict"])
        torch.set_rng_state(rng_state)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Completed")
    train_dataset = MRCCoNLL(path=train_data_path, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_dataset = MRCCoNLL(path=valid_data_path, max_length=max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model.to(device)
    evaluator = Evaluate()
    
    if not os.path.exists(loss_file):
        with open(loss_file, "w") as file:
            json.dump([], file)
    
    metric = -100
    start_epoch = checkpoint.get("epoch", 0)
    for epoch in range(start_epoch, epochs):
        logger.info(f"Start training epoch {epoch}...")
        history = {
            "train_step_loss": list(),
            "train_F1_score": list(),
            "valid_step_loss": list(),
            "valid_F1_score": list(),
            "train_precision_score": list(),
            "train_recall_score": list(),
            "valid_precision_score": list(),
            "valid_recall_score": list(),
        }
        
        model.train()
        with tqdm(total=len(train_loader),
                  desc=f'Epoch: {epoch + 1}/{epochs}',
                  dynamic_ncols=True
                  ) as pbar:
            for data, label in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                label = label.to(device)
                output = model(data).logits
                loss = criterion(output.view(-1, 50280), label.view(-1))
                loss.backward()
                optimizer.step()
                
                output = torch.argmax(torch.softmax(output.detach().cpu(), dim=-1), dim=-1)
                precision_score, recall_score, F1_score = evaluator(output.view(-1), label.detach().cpu().view(-1))
                history["train_step_loss"].append(loss.item())
                history["train_precision_score"].append(precision_score)
                history["train_recall_score"].append(recall_score)
                history["train_F1_score"].append(F1_score)

                pbar.update()
                pbar.set_postfix_str(f'loss: {loss.item()}, F1 score: {F1_score}')

        model.eval()
        with tqdm(total=len(valid_loader),
                  desc=f'Epoch: {epoch + 1}/{epochs}',
                  dynamic_ncols=True
                  ) as pbar:
            with torch.no_grad():
                for data, label in valid_loader:
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data).logits
                    loss = criterion(output.view(-1, 50280), label.view(-1))

                    output2 = torch.argmax(torch.softmax(output.detach().cpu(), dim=-1), dim=-1)

                    precision_score, recall_score, F1_score = evaluator(output2.view(-1), label.detach().cpu().view(-1))
                    history["valid_step_loss"].append(loss.item())
                    history["valid_precision_score"].append(precision_score)
                    history["valid_recall_score"].append(recall_score)
                    history["valid_F1_score"].append(F1_score)

                    pbar.update()
                    pbar.set_postfix_str(f'loss: {loss.item()}, F1 score: {F1_score}')

        snapshot = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state_dict": torch.get_rng_state(),
            "epoch": epoch,
        }
        logger.info("Saving checkpoint...")
        checkpoint.update(snapshot)
        torch.save(checkpoint, checkpoint_path)
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            with open(loss_file, "r") as file:
                loss_history = json.load(file)
            temp_file.write(json.dumps(loss_history))
            temp_file.close()
            
            loss_history.append(history)
            
            with open(loss_file, "w") as file:
                json.dump(loss_history, file, indent=4)
        
        except Exception as e:
            shutil.copy(temp_file.name, loss_file)
            print(f"Error occurred: {e}. Original data has been restored.")
        finally:
            os.unlink(temp_file.name)
        
        train_loss = np.mean(history['train_step_loss']) if history['train_step_loss'] else 0
        train_f1 = np.mean(history['train_F1_score']) if history['train_F1_score'] else 0
        valid_loss = np.mean(history['valid_step_loss']) if history['valid_step_loss'] else 0
        valid_f1 = np.mean(history['valid_F1_score']) if history['valid_F1_score'] else 0
        
        with open("history.txt", "a") as file:
            result = f"{str(datetime.now())}-epoch-{epoch} train_loss: {train_loss} valid loss: {valid_loss} train F1 score: {train_f1} "\
                     f"valid F1 score: {valid_f1}\n"
            file.write(result)
            
        if np.mean(history['valid_F1_score']) >= metric:
            metric = np.mean(history['valid_F1_score'])
            torch.save(model.state_dict(), "best.pt")
            
        logger.info(f"Completed training epoch {epoch}.")
    torch.save(model.state_dict(), 'last_epoch_model.pt')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', default=130, type=int, help='number of train epochs')
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help='initial learning rate')
    parser.add_argument('--max_length', default=256, type=int,
                        help='Controls the maximum length to use by one of the truncation/padding parameters.')
    parser.add_argument('--resume', required=True, type=bool, help='resume from checkpoint')

    args = parser.parse_args()
    checkpoint_path = "checkpoint.pth"

    main(args)