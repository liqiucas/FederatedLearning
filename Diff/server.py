import torch
import torchvision.models as models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Server(object):
  def __init__(self, conf, eval_dataset):
    self.conf = conf
    self.global_model = models.__dict__[self.conf["model_name"]]()

    self.eval_loader = torch.utils.data.DataLoader(
      eval_dataset,
      batch_size=self.conf["batch_size"],
      shuffle=False
    )
  # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
  def model_aggregate(self, weight_accumulator):
    # 遍历服务器的全局模型
    for name, data in self.global_model.state_dict().items():
      # 更新每一层乘上学习率
      update_per_layer = weight_accumulator[name] * self.conf["lambda"]
      # 累加和
      if data.type() != update_per_layer.type():
          # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
          data.add_(update_per_layer.to(torch.int64))
      else:
          data.add_(update_per_layer)

  # 评估函数
  def model_eval(self):
      self.global_model.eval()
      total_loss = 0.0
      correct = 0
      dataset_size = 0
      for batch_id, batch in enumerate(self.eval_loader):
          data, target = batch
          dataset_size += data.size()[0]
          if torch.cuda.is_available():
              data = data.cuda()
              target = target.cuda()
          output = self.global_model(data)
          total_loss += torch.nn.functional.cross_entropy(output,target,reduction='sum').item()
          # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
          pred = output.data.max(1)[1]
          correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
      acc = 100.0 * (float(correct) / float(dataset_size))    # 计算准确率
      total_1 = total_loss / dataset_size                     # 计算损失值
      return acc, total_1