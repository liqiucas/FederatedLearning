import torch
import torchvision.models as models
import  torch, copy

class Client(object):
    def __init__(self, conf, train_dataset, eval_dataset, idx=1):
        self.conf = conf
        self.client_id = idx
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # 按ID对训练集合的拆分
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['num_models'])
        indices = all_range[idx * data_len: (idx + 1) * data_len]
        # 生成一个数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
        )
        self.eval_loader = torch.utils.data.DataLoader(
          eval_dataset,
          batch_size=self.conf["batch_size"],
          shuffle=False
        )

    def local_train(self, model):
        """
        # 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到
        for name, param in model.state_dict().items():
            # 客户端首先用服务器端下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        """
        self.local_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])
        acc, loss = self.model_eval()
        print("\t\t\t Client %d local train done acc: %f, loss: %f" % (self.client_id,acc,loss))
        return diff

    def model_eval(self):
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.local_model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))  # 计算准确率
        total_1 = total_loss / dataset_size  # 计算损失值
        return acc, total_1