import argparse
import copy
import json
import random

import datasets
from client import *
from server import *

if __name__ == '__main__':
    with open('conf.json', 'r') as f:
        conf = json.load(f)
    train_dataset, eval_dataset = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_dataset)
    clients = []
    
    if torch.cuda.is_available():
        server.global_model.cuda()


    # 添加10个客户端到列表
    for client in range(conf["num_models"]):
        clients.append(Client(conf, train_dataset, eval_dataset, client))

    for round in range(conf["global_epochs"]):
        print("Global Epoch %d" % round)
        candidates = random.sample(clients, conf["k"])
        print("\tselect clients is: ")
        for client in candidates:
            print("\t\t", client.client_id, end="\t")
        print()

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 遍历客户端，每个客户端本地训练模型
        for client in candidates:
            update_model = client.local_train(server.global_model)
            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                if params.type() != update_model[name].type():
                    update_model[name].to(torch.int64)
                weight_accumulator[name].add_(update_model[name])

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()
        # acc_list.append(acc)
        # loss_list.append(loss)

        print("Epoch %d, acc: %f, loss: %f\n" % (round, acc, loss))
