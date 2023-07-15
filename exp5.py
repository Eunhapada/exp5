#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50


# ## 数据预处理

# In[2]:


#加载预训练模型和分词器，设置固定超参数
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# use_gpu = torch.cuda.is_available()
# if use_gpu:
#     print('GPU')
#     device = torch.device("cuda")
#     torch.cuda.empty_cache() 
# else:
device = torch.device("cpu")
#我的设备GPU内存不足运行代码，因此后续只用cpu运行。

num_classes = 3
num_epoch = 10
max_length=128
criterion = nn.CrossEntropyLoss()


# #### 读取训练数据

# In[ ]:


folder_path = "data/"
train_label_path = "train.txt"
train_label_df = pd.read_csv(train_label_path,sep=",")
column_dict = {"positive": 0, "negative": 1,"neutral":2}
new_df = train_label_df.replace({"tag": column_dict})
labels = list(new_df['tag'])


# #### 处理图片数据

# In[3]:


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 图片转为resnet输入格式
    transforms.ToTensor(),  # 将图片转换为tensor
])

def get_ImagePath(folder_path ,df):
    image_paths = []
    for ind in df['guid']:
        image_path = folder_path+str(ind)+".jpg"
        try:
            image_paths.append(image_path)
            # print(image_path)
        except Exception as e:
            #print(f"file '{file}' not found")
            continue
    
    return image_paths

image_paths = get_ImagePath(folder_path,new_df)


# #### 处理文本数据

# In[ ]:


def get_txt(folder_path,df):
    texts=[]
    for ind in df['guid']:
        file = folder_path+str(ind)+".txt"
        try:
            with open(file, "r",encoding="GB18030") as infile:
                content = infile.read()
                texts.append(content)
        except FileNotFoundError:
            continue
    return texts

texts = get_txt(folder_path,new_df)


# In[4]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels,transform=None):
        self.image_paths = image_paths     
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        
        return image ,input_ids, attention_mask, labels
    def __len__(self):
        return len(self.input_ids)
    
def txt_processing(texts):
    tokenized_texts = [tokenizer(text,padding='max_length',max_length=max_length,truncation=True,return_tensors="pt") for text in texts]
    return tokenized_texts


# In[5]:


# 划分验证集
img_train, img_val, txt_train, txt_val, labels_train, labels_val = train_test_split(image_paths, texts, labels, test_size=0.2, random_state=9)
#文本预处理
tokenized_txt_train = txt_processing(txt_train)
tokenized_txt_val = txt_processing(txt_val)
# 构建Dataset
dataset_train = Dataset(img_train, tokenized_txt_train, labels_train, transform)
dataset_val = Dataset(img_val,tokenized_txt_val, labels_val, transform)


# ## 搭建多模态处理模型

# In[6]:


class ImgModel(nn.Module):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.resnet = resnet50(pretrained=True)  # 使用预训练的ResNet-50作为图片特征提取器
    
    def forward(self, image):
        return self.resnet(image)
    
# 文本特征提取模型定义
class TxtModel(nn.Module):
    def __init__(self):
        super(TxtModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]

# 多模态融合模型定义
class FusionModel(nn.Module):
    def __init__(self, num_classes,option):
        super(FusionModel, self).__init__()
        self.img_model = ImgModel()  
        self.txt_model = TxtModel()
        self.option=option
        #多模态融合
        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )
    def forward(self, image, input_ids,attention_mask):
        if(self.option==0):
            output = self.img_model(image)
            output = self.classifier0(output)
        elif(self.option==1):
            output = self.txt_model(input_ids, attention_mask)
            output = self.classifier0(output)
        else:
            #拼接两类特征
            output = torch.cat((self.txt_model(input_ids,attention_mask),self.img_model(image)), dim=-1)
            output = self.classifier0(output)
        return output
    


# #### 定义训练和预测函数

# In[7]:


# 训练过程
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0
    total_correct = 0 
    for images, input_ids, attention_mask, labels in train_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)     
        labels = labels.to(device)     
        optimizer.zero_grad()     
        outputs = model(images, input_ids,attention_mask)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

# 预测过程
def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask,  _ in test_loader:
        images = images.to(device)
        #texts = texts.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions


# #### 模型训练

# In[8]:


torch.cuda.set_device(0)
batch_size = 64
best_acc = 0
#构建数据加载器DataLoader
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)


# In[9]:


option=2
learning_rate = 1e-7
torch.cuda.empty_cache() 
model = FusionModel(num_classes,option)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epoch):
    train_loss, train_acc = train_model(model, loader_train, criterion, optimizer, device)
    val_predictions = predict_model(model, loader_val, device)
    # 计算验证集准确率    
    val_predictions = np.array(val_predictions)
    val_labels = np.array(labels_val)
    val_acc = (val_predictions == val_labels).sum() / len(val_labels)
    if(val_acc>best_acc):
        best_acc = val_acc
        #保存当前在验证集上表现最好的模型
        torch.save(model, 'multi_model.pt')
    print(f"batch size: {batch_size}, lr: {learning_rate}, Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc:{best_acc:.4f}")


# #### 生成预测文本并写回test_without_label.txt文件

# In[11]:


#读取test文件并生成预测文件
test_path = "test_without_label.txt"
test_df = pd.read_csv(test_path,sep=",")
test_df.iloc[:,-1]=0
test_labels = np.array(test_df['tag'])

#image_paths
image_paths_test = get_ImagePath(folder_path,test_df)
test_texts = get_txt(folder_path,test_df)

tokenized_texts_test = txt_processing(test_texts)
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

best_model = torch.load('multi_model.pt').to(device)
test_predictions = predict_model(best_model, loader_test, device)  
test_predictions = np.array(test_predictions)

column_dict_ = {0:"positive", 1:"negative",2:"neutral"}
test_df['tag'] = test_predictions
pre_df = test_df.replace({"tag": column_dict_})
pre_df.to_csv('test_without_label.txt',sep=',',index=False)


# In[ ]:




