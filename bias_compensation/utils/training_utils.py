import torch
import timm
import time

def eval_one_batch(eval_model,batch_img,targets):
    eval_model.eval()
    with torch.no_grad():
        outputs = eval_model(batch_img)
    # compute the accuracy
    acc1, acc5 = timm.utils.accuracy(outputs, targets, topk=(1, 5))
    print(f'Acc@1: {acc1:>7.3f}   '
            f'Acc@5: {acc5:>7.3f}  ')

def train_one_batch(train_model,optimizer,loss_fn,batch_img,targets):
    optimizer.zero_grad()
    outputs = train_model(batch_img)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_on_datatset(eval_model,loader_eval,device,log_interval=50):
    batch_time_m = timm.utils.AverageMeter()
    top1_m = timm.utils.AverageMeter()
    top5_m = timm.utils.AverageMeter()
    last_idx = len(loader_eval) - 1
    eval_model.to(device)
    eval_model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx,(img,labels) in enumerate(loader_eval):
            # evaluate the model
            last_batch = batch_idx == last_idx
            img,labels  = img.to(device),labels.to(device)
            outputs = eval_model(img)
            # compute the accuracy
            acc1, acc5 = timm.utils.accuracy(outputs, labels, topk=(1, 5))
            if device.type == 'cuda':
                torch.cuda.synchronize()

            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % log_interval == (log_interval-1)):
                log_name = 'Test'
                print('\r'
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})',end=""
                )

