"""
Contains binary training functions for training and testing a PyTorch model.
"""
import torch

# train loop
def train_steps(model:torch.nn.Module,
                dataloader: torch.utils.data,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim,
                device:str):
  train_loss, train_acc = 0, 0
  model.train()
  for X,y in dataloader:
    X, y= X.to(device), y.to(device)
    # datatype of y is int64 so we have to convert it to float32
    y = y.type(torch.float32)
    # forward pass
    train_logit = model(X).squeeze()
    train_pred = torch.round(torch.sigmoid(train_logit))
    
    # calculate the loss
    loss = loss_fn(train_logit,y)    
    train_loss += loss.item()

    # calculate accuracy
    acc = (((train_pred == y).sum().item() / len(train_pred)) * 100 )
    train_acc += acc

    # optimizer zero_grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc


# test loop 
def test_steps(model:torch.nn.Module,
                  dataloader: torch.utils.data,
                  loss_fn: torch.nn.Module,
                  device:str):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      # datatype of y is int64 so we have to convert it to float32
      y = y.type(torch.float32)

      # forward pass
      test_logit = model(X).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logit))

      # calculate the loss
      loss = loss_fn(test_logit, y)
      test_loss += loss.item()

      # calculate the accuracy
      acc = (((test_pred == y).sum().item() / len(test_pred)) * 100)
      test_acc += acc
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def Binary_engine(model:torch.nn.Module,
                  train_dataloader: torch.utils.data,
                  test_dataloader: torch.utils.data,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim,
                  device:str,
                  epochs:int):
  from tqdm.auto import tqdm
  results = {
      "train_loss":[],
      "train_acc":[],
      "test_loss":[],
      "test_acc":[]
  }

  for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch+1}\n----------------")
    train_loss, train_acc = train_steps(model = model,
                                        dataloader = train_dataloader,
                                        loss_fn = loss_fn,
                                        optimizer = optimizer,
                                        device = device)
    test_loss, test_acc = test_steps(model = model,
                                     dataloader = test_dataloader,
                                     loss_fn = loss_fn,
                                     device = device)

    print(f"train loss: {train_loss:.5f}, train acc: {train_acc:.3f}% | test loss: {test_loss:.5f}, test_acc: {test_acc:.3f}%")

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  
  return results
  