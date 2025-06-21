import torch
import torch.nn.functional as F
from math import sqrt

def train(train_loader, model, optimizer, aca_loss, device):
    model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, embed = model(data.x.float(), data.edge_index,
                           data.edge_attr, data.batch)

        loss = aca_loss(data.y,
                        labels = data.y, 
                        predictions = out,
                        embeddings = embed,
                        fps_smiles = data.fp_smiles,
                        fps_scaffold = data.fp_scaffold,                           
                        smiles_list = data.smiles,                           
                        )
        
        #loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)



@torch.no_grad()
def test(test_loader, model, device):
    model.eval()
    mse = []
    for data in test_loader:
        data = data.to(device)
        out, embed = model(data.x.float(), data.edge_index,
                           data.edge_attr, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


@torch.no_grad()
def predict(test_loader, model,  device):
    #embeds = []
    model.eval()
    preds = []
    for data in test_loader:
        data = data.to(device)
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)
        #embeds.append(embeddings)
        preds.append(predictions)
    #embeddings = torch.concat(embeds, axis=0).cpu().numpy()
    predictions = torch.concat(preds, axis=0)
    return predictions