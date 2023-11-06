
import torch
from typing import Callable
from tqdm.auto import tqdm
from timeit import default_timer as timer
from modular import metrics, utils
from modular.metrics import MultiLabelResult
import copy

def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   accuracy_fn: Callable,
                   device: torch.device
                   ):
    test_result = test_step(model=model,
                                        dataloader=dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device
                                       )
    return test_result

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn: Callable,
              device: torch.device,
         ):
    classes = dataloader.dataset.classes
    #Put the model in train mode
    model.train()
    model = model.to(device)
    train_loss = 0.0
    train_acc = 0.0
 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()
     
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        train_acc += accuracy_fn(y_true=y,y_pred=y_pred_logits)
        
        
    
    #Adjust metrics to get avg loss and accuracy per batch
    train_loss /= len(dataloader) 
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn : torch.nn.Module,
             accuracy_fn : Callable,
             device: torch.device):
    classes = dataloader.dataset.classes
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
    
            #Calculate the accuracy
            test_acc += accuracy_fn(y_true=y,y_pred=test_pred_logits)
    #Adjust metrics to get avg loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc
  
#1. Create a train function that takes in various model parameters + optimizer + dataloaders+ loss function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          accuracy_fn : Callable,
          loss_fn: torch.nn.Module,
          device: torch.device,
          epochs: int=5,
          patience: int = 5,
          model_dir = None,
          model_name: str = None
         ):

    min_test_loss = 99
    counter = 0
    model_save_path = ''

    start_time = timer()
    results = {"train_loss":[],"train_acc":[],"test_loss":[], "test_acc":[], }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          accuracy_fn=accuracy_fn,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device
                                       )
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        print(f"Epoch: {epoch}\n--------------------------")
        print(f"Train loss: {train_loss} | Train acc : {train_acc}| Test loss: {test_loss} | Test acc: {test_acc}")

        #If model loss is lowest, copy the model weight into memory
        if test_loss < min_test_loss:
            counter = 0
            min_test_loss = test_loss
            if model_dir:
                utils.save_model(model=model,
                 target_dir=model_dir,
                 model_name=model_name)
            best_model = copy.deepcopy(model.state_dict())
        else:
            counter += 1
        #Load best performing model weight when early stopping
            if counter >= patience:
                print(f'Test loss has not been improving for {patience} epochs. Training is terminated...')
                model.load_state_dict(best_model)
                break
        
        
    end_time = timer()
    print(f'[INFO] Total training time: {end_time-start_time:.3f} seconds')
    return results
