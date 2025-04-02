import logging
import os
import time

from model import get_model
from my_dataset import MyDataset
from tools import load_data
from BpGa import ga_optimization, set_params
import torch
from tqdm import tqdm

def create_logger(logger_file_path):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = create_logger('./log')
    X_train, X_test, Y_train, Y_test = load_data()
    train_loader = torch.utils.data.DataLoader(MyDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MyDataset(X_test, Y_test), batch_size=32, shuffle=False)
    model = get_model()
    if os.path.exists('ga_optimized_weights.pth'):
        model.cuda()
        logger.info("Load model weights from ga_optimized_weights.pth")
        model.load_state_dict(torch.load('ga_optimized_weights.pth'))
        model.double()
    else:
        # 进行遗传算法优化
        logger.info("Start GA optimization")
        best_params = ga_optimization(model, train_loader,val_loader)
        set_params(model, best_params)
        logger.info("GA optimization finished")
        # torch.save(model, 'ga_optimized_model.pth')
        # 仅保存权重
        torch.save(model.state_dict(), 'ga_optimized_weights.pth')

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()
    model.cuda()
    logger.info("====================Start training=====================")
    for epoch in range(1000):
        model.train()
        best_loss = 0
        for data, target in tqdm(train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        model.eval()
        loss = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader):
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                loss += criterion(output, target).item()
            if epoch == 0:
                best_loss = loss
            else:
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), f'loss_{best_loss}_model.pth')
        # print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")
        logger.info(f"Epoch {epoch}, Val_Loss: {loss:.4f}")
    logger.info("====================End training=====================")