import torch
import torch.nn.functional as F
from math import sqrt

def train(train_loader, model, optimizer, aca_loss, device, dev_mode=False):
    model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, embed = model(data.x.float(), data.edge_index,
                           data.edge_attr, data.batch)
        if dev_mode:
            loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets , n_mined_triplets_origin, n_pos_triplets_origin = aca_loss(data.y, out, embed, fp_values = data.fp, scaffold_values = data.scaffold)
            print(f"[Train] Loss: {loss:.4f}, Reg Loss: {reg_loss:.4f}, TSM Loss: {tsm_loss:.4f} | " \
                    f"Mined Triplets: {n_mined_triplets}, Pos Triplets: {n_pos_triplets} | " \
                    f"Origin Mined: {n_mined_triplets_origin}, Origin Pos: {n_pos_triplets_origin}")
        else:
            loss = aca_loss(data.y, out, embed, fp_values = data.fp, scaffold_values = data.scaffold)
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
