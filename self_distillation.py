from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
# import gs.gumbel_softmax
import torch




class LightningDistillation(ABC):

    @abstractmethod
    def init(self, args, model):
        pass

    @abstractmethod
    def on_train_epoch_start(self, args):
        pass

    @abstractmethod
    def step(self, args):
        pass

    @abstractmethod
    def on_train_epoch_end(self, model):
        pass

    @abstractmethod
    def forward(self, output):
        pass


class ModelDistillation(ABC):

    @abstractmethod
    def model_init(self, args):
        pass

    @abstractmethod
    def get_self_distillation_loss(self, args):
        pass

    @abstractmethod
    def model_distillation_forward(self, last_output):
        pass


class FirstLightningDistillation(LightningDistillation):
    def __init__(self):
        self.criterion = nn.BCELoss()
        self.model = FirstModelDistillation()
        self.alpha = 0.1
        self.beta = 1e-6

    def init(self, args, model):
        self.model.model_init(args)

    def on_train_epoch_start(self, args):
        ...

    # def compute_adaptive_weights(self,args):
    #     # 方法1：线性衰减软损失权重 (beta)
    #     total_epochs = args['total_epoch']
    #     progress = args['current_epoch']  / total_epochs
    #     beta = max(0.0, 1.0 - progress)  # 从1线性降到0
    #     alpha = 1.0 - beta
    #     return alpha, beta
    
        # 方法2：余弦退火调整（平滑过渡）
    def compute_adaptive_weights(self,args):
        progress = args['current_epoch'] / args['total_epoch']
        beta = 0.5 * (1 + math.cos(math.pi * progress))  # 从1降到0，余弦曲线
        alpha = 1 - beta
        return alpha, beta

    def step(self, args):
        """

                :param args: loss, main_pre_output_layer_features, post_main_output_layer, y
                :return:
        """
        
        loss = args['loss']
        y = args['y']
        # current_epoch = args['current_epoch']
        # pdb.set_trace()
        # if (args.get('current_epoch', None) is not None) and (args.get('total_epoch', None) is not None):
        #     # pdb.set_trace()
        #     self.alpha,self.beta= self.compute_adaptive_weights(args)
        feature_loss, entropy_feature_loss = self.model.get_self_distillation_loss(args)
        post_output_layer1, post_output_layer2, \
        post_output_layer3, post_output_layer4 = self.model.get_sub_classifiers()
        sub_loss_1 = self.criterion(post_output_layer1, y)
        sub_loss_2 = self.criterion(post_output_layer2, y)
        sub_loss_3 = self.criterion(post_output_layer3, y)
        sub_loss_4 = self.criterion(post_output_layer4, y)

        entropy_loss = (1 - self.alpha) * (loss + sub_loss_1 + sub_loss_2 + sub_loss_3 + sub_loss_4)
        entropy_feature_loss = self.alpha * entropy_feature_loss
        feature_loss = self.beta * feature_loss

        loss = entropy_loss + entropy_feature_loss + feature_loss
        return loss

    def on_train_epoch_end(self, model):
        ...

    def forward(self, output):
        self.model.model_distillation_forward(output)


class FirstModelDistillation(ModelDistillation):
    def __init__(self):
        self.feature_loss = nn.L1Loss()
        self.entropy_model_loss = nn.BCELoss()

    def get_sub_classifiers(self):
        return self.post_output_layer1, self.post_output_layer2, self.post_output_layer3, self.post_output_layer4

    def model_init(self, args):
        # pdb.set_trace()
        if args.get('hidden_size', None) is not None:
            hidden_size = args['hidden_size']
        if args.get('output_size', None) is not None:
            output_size = args['output_size']
        if args.get('attention', None) is not None:
            self.attention = args['attention']
        if args.get('create_layer_block', None) is not None:
            create_layer_block = args['create_layer_block']
            self.pre_output_layer1, self.output_layer1 = create_layer_block(hidden_size, output_size, self.attention)
            self.pre_output_layer2, self.output_layer2 = create_layer_block(hidden_size, output_size, self.attention)
            self.pre_output_layer3, self.output_layer3 = create_layer_block(hidden_size, output_size, self.attention)
            self.pre_output_layer4, self.output_layer4 = create_layer_block(hidden_size, output_size, self.attention)

    def get_self_distillation_loss(self, args):
        main_pre_output_layer_features = args['main_pre_output_layer_features']
        post_main_output_layer = args['post_main_output_layer']
        # 1
        feature_loss_1 = self.feature_loss(self.pre_output_layer1_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_1 = self.entropy_model_loss(self.post_output_layer1, post_main_output_layer.detach())
        # 2
        feature_loss_2 = self.feature_loss(self.pre_output_layer2_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_2 = self.entropy_model_loss(self.post_output_layer2, post_main_output_layer.detach())
        # 3
        feature_loss_3 = self.feature_loss(self.pre_output_layer3_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_3 = self.entropy_model_loss(self.post_output_layer3, post_main_output_layer.detach())
        # 4
        feature_loss_4 = self.feature_loss(self.pre_output_layer4_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_4 = self.entropy_model_loss(self.post_output_layer4, post_main_output_layer.detach())

        feature_losses = feature_loss_1 + feature_loss_2 + feature_loss_3 + feature_loss_4
        entropy_feature_losses = entropy_feature_loss_1 + entropy_feature_loss_2 + \
                                 entropy_feature_loss_3 + entropy_feature_loss_4

        return feature_losses, entropy_feature_losses

    def model_distillation_forward(self, last_output):
        # output 1
        self.pre_output_layer1_features = self.pre_output_layer1(last_output)
        output_layer1 = self.output_layer1(self.pre_output_layer1_features)
        self.post_output_layer1 = F.sigmoid(output_layer1)

        # output 2
        if self.attention:
            pre_output_layer1_features = last_output
        else:
            pre_output_layer1_features = self.pre_output_layer1_features
        self.pre_output_layer2_features = self.pre_output_layer2(pre_output_layer1_features)
        output_layer2 = self.output_layer2(self.pre_output_layer2_features)
        self.post_output_layer2 = F.sigmoid(output_layer2)

        # output 3
        if self.attention:
            pre_output_layer2_features = last_output
        else:
            pre_output_layer2_features = self.pre_output_layer2_features
        self.pre_output_layer3_features = self.pre_output_layer3(pre_output_layer2_features)
        output_layer3 = self.output_layer3(self.pre_output_layer3_features)
        self.post_output_layer3 = F.sigmoid(output_layer3)

        # output 4
        if self.attention:
            pre_output_layer3_features = last_output
        else:
            pre_output_layer3_features = self.pre_output_layer3_features
        self.pre_output_layer4_features = self.pre_output_layer4(pre_output_layer3_features)
        self.post_output_layer4 = F.sigmoid(self.output_layer4(self.pre_output_layer4_features))


class SecondLightningDistillation(LightningDistillation):
    def __init__(self):
        self.distil_loss = nn.BCELoss()
        self.prev_model = None
        self.current_epoch = 0
        self.total_epoch = 0
        self.alpha = 0

    def init(self, args, model):
        if args.get('total_epoch', None):
            self.total_epoch = args['total_epoch']

        if model is not None:
            self.prev_model = model

    def on_train_epoch_start(self, args):
        model = args['model']
        self.current_epoch = args['current_epoch']
        self.prev_model.load_state_dict(model.state_dict())
        self.alpha = self.current_epoch / self.total_epoch

    def step(self, args):
        # pdb.set_trace()
        x = args['x']
        x_length = args['x_length']
        loss = args['loss']
        post_main_output_layer = args['post_main_output_layer']

        out = self.prev_model(x, x_length)

        distil_loss = self.distil_loss(post_main_output_layer, out.detach())  # self distillation regularizer
        total_loss = (1 - self.alpha) * loss + self.alpha * distil_loss
        return total_loss

    def on_train_epoch_end(self, model):
        ...#self.prev_model.load_state_dict(model.state_dict())

    def forward(self, output):
        ...

# from model import PolicyLSTM
class HierarchicalDistillation(LightningDistillation):
    def __init__(self):
        self.distil_loss = nn.BCELoss()
        self.feature_loss = nn.L1Loss()
        self.criterion = nn.BCELoss()
        self.prev_cnn = None
        self.prev_lstm = None
        self.prev_transformer = None
        self.prev_output_layers = None
        self.current_epoch = 0
        self.total_epoch = 0
        self.alpha = 0
        self.beta = 0.01  # 控制对比学习权重
        self.gamma = 0.01  # 控制混淆的指标
        self.temperature = 3# 温度参数  CNNLSTM 2   //1 0.860 1.8 0.861 3 0.861//  1 0.799 1.15 0.787  1.1 0.786 1.2 0.803  1.25 0.813 1.3 0.792 1.26 0.795 1.24 0.789  1.8 0.794 1.9 0.786 2.5 0.792 2.6 0.795 2.7 0.802 2.8 0.797 3 0.791 3.2 0.785 3.3 0.801 3.5 0.792 4 0.797
        self.pre_output_temperature = 3  # 最终输出层的温度参数 LSTM// 1.1 0.780 1.2 0.791 1.25 0.788 1.5 0.765 1.8 0.776 2 0.791 2.1 0.799 2.15 0.7915 2.2 0.792 2.25 0.786 2.3 0.795 2.4 0.784 2.5 0.779 3 0.773 4 0.770 1 0.765
        self.contrastive_mode = 'prev'  # 'prev' 或 'current' 用于控制对比学习模式

    def set_args_time(self,temperature):
        self.temperature = temperature
    def set_policy_network(self, policy_network):
        self.policy_network = policy_network
    def init(self, args, model):
        if args.get('total_epoch', None):
            self.total_epoch = args['total_epoch']
        if model is not None:
            # pdb.set_trace()
            if model.__class__.__name__ ==  'PolicyLSTM':
                self.prev_lstm = model.lstm
                self.prev_output_layers = nn.Sequential(
                    model.main_pre_output_layer,
                    model.main_output_layer
                )
            else:
            # 保存模型的各个组件
                self.prev_cnn = nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.conv2
                )
                self.prev_lstm = model.lstm
                self.prev_output_layers = nn.Sequential(
                    model.main_pre_output_layer,
                    model.main_output_layer
                )

    def on_train_epoch_start(self, args):
        model = args['model']
        self.current_epoch = args['current_epoch']
        if model.__class__.__name__ ==  'PolicyLSTM':
            self.prev_lstm.load_state_dict(model.lstm.state_dict())
        # 更新各个组件
        else:
            self.prev_cnn[0].load_state_dict(model.conv1.state_dict())
            self.prev_cnn[1].load_state_dict(model.bn1.state_dict())
            self.prev_cnn[2].load_state_dict(model.conv2.state_dict())
            self.prev_lstm.load_state_dict(model.lstm.state_dict())
        
        # 第一个epoch不使用策略网络
        self.use_policy = self.current_epoch > 0
        self.alpha =self.current_epoch / self.total_epoch


    def on_train_epoch_end(self, model):
        pass

    def feature_fusion_step(self, args):
        """
        按照算法伪代码实现:
        1. 获取teacher和student特征
        2. 计算路由权重
        3. 计算损失
        4. 从输入x开始，使用for循环交叉计算out
        """
        x = args['x']
        y = args['y']
        model = args['model']  # student model
        # pdb.set_trace()
        # Step 1: 获取teacher和student特征 (算法第17-19行)
        with torch.no_grad():
            # Teacher features
            t_conv1 = F.relu(self.prev_cnn[1](self.prev_cnn[0](x.transpose(1, 2))))
            t_conv2 = F.relu(self.prev_cnn[2](t_conv1))
            t_lstm, _ = self.prev_lstm(t_conv2.transpose(1, 2))
            t_pre = self.prev_output_layers[0](t_lstm[:, -1])
            logit_t = F.sigmoid(self.prev_output_layers[1](t_pre))
            feat_t = [t_conv2, t_lstm[:, -1], t_pre]  # CNN, LSTM, pre-output features

        # Student features
        s_conv1 = F.relu(model.bn1(model.conv1(x.transpose(1, 2))))
        s_conv2 = F.relu(model.conv2(s_conv1))
        s_lstm, _ = model.lstm(s_conv2.transpose(1, 2))
        s_pre = model.main_pre_output_layer(s_lstm[:, -1])
        logit_s = F.sigmoid(model.main_output_layer(s_pre))
        feat_s = [s_conv2, s_lstm[:, -1], s_pre]  # CNN, LSTM, pre-output features
        # pdb.set_trace()
        # Step 2: 计算路由权重 (算法第20-23行)
        ft = torch.cat((feat_t[-1], feat_s[-1]),1)  # 拼接最后一层特征
        # pdb.set_trace()
        # w = F.gumbel_softmax(self.policy_network(ft.flatten().reshape(1, -1)), tau=1.0).squeeze()

        # w = torch.mean(gs.gumbel_softmax(self.policy_network(ft), 5), dim=0)# 使用Gumbel-Softmax
        w = torch.mean(F.gumbel_softmax(self.policy_network(ft), tau=1.0), dim=0)
        # part1, part2 = torch.split(self.policy_network(ft), 4, dim=1)
        # t_w = torch.mean(F.gumbel_softmax(part1, tau=1.0), dim=0)
        # s_w = torch.mean(F.gumbel_softmax(part2, tau=1.0), dim=0)
        # s_d = s_w.detach()
        # t_d = t_w.detach()

        d = w.detach()  # 停止梯度传播
        # pdb.set_trace()
        # Step 3: 计算损失
        # Cross-entropy loss
        L_s = self.criterion(logit_s, y)
        # KL divergence loss
        L_s += self.beta * F.kl_div(F.log_softmax(logit_s  / self.temperature, dim=1),F.softmax(logit_t.detach() / self.temperature, dim=1),reduction='batchmean') *  d[-1]
        for i in range(len(feat_s)):
            L_s += self.gamma * self.feature_loss(feat_s[i], feat_t[i].detach()) *  d[i]

        # Step 4: 从输入x开始，使用for循环交叉计算out
        # 模型组件列表
        student_modules = [
            lambda x: F.relu(model.conv2(F.relu(model.bn1(model.conv1(x))))),  # CNN
            lambda x: model.lstm(x.transpose(1, 2))[0][:, -1],  # LSTM
            lambda x: model.main_pre_output_layer(x),  # pre-output
            lambda x: model.main_output_layer(x)  # final output
        ]
        
        teacher_modules = [
            lambda x: F.relu(self.prev_cnn[2](F.relu(self.prev_cnn[1](self.prev_cnn[0](x))))),  # CNN
            lambda x: self.prev_lstm(x.transpose(1, 2))[0][:, -1],  # LSTM
            lambda x: self.prev_output_layers[0](x),  # pre-output
            lambda x: self.prev_output_layers[1](x)  # final output
        ]

        # 初始化输入
        out_s = x.transpose(1, 2)
        out_t = x.transpose(1, 2)
        # pdb.set_trace()
        # 逐层计算
        for i in range(len(student_modules)):
            # 通过当前层的模型
            s_out = student_modules[i](out_s)
            t_out = teacher_modules[i](out_t)
            
            # if i == 3:
            #     continue
            # out_s = s_out
            # 交叉计算
            out_s = s_out * (1 -  d[i]) + t_out *  d[i]
            out_t = s_out * (1 -  d[i]) + t_out *  d[i]
            # out_t = t_out
        # 最终输出
        # out = s_out * (1 - w[-1]) + t_out * w[-1]
        out = out_t
        # pdb.set_trace()
        # 训练策略网络
        # print( self.criterion(F.sigmoid(out), y))
        L_routing = self.alpha * self.criterion(F.sigmoid(out), y)
        
        # L_s += self.alpha * self.criterion(F.sigmoid(out), y)

        L_routing.backward(retain_graph=True)
        
        if hasattr(self.policy_network, 'optimizer'):
            self.policy_network.optimizer.step()
            self.policy_network.optimizer.zero_grad()
        return L_s  # 返回student的损失用于student的反向传播

    def policy_lstm_feature_fusion_step(self, args):
        """
        PolicyLSTM的策略蒸馏实现:
        1. 获取teacher和student的LSTM特征
        2. 计算路由权重
        3. 计算损失
        4. 交叉计算特征融合
        """
        # pdb.set_trace()
        x = args['x']
        x_length = args['x_length']
        y = args['y']
        model = args['model']  # student model

        # Step 1: 获取teacher和student特征
        with torch.no_grad():
            # Teacher features
            t_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_length.cpu(), batch_first=True, enforce_sorted=False)
            t_lstm_out, _ = self.prev_lstm(t_packed)
            t_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(t_lstm_out, batch_first=True)
            t_last_idx = torch.LongTensor([len_idx - 1 for len_idx in x_length])
            t_lstm = t_lstm_out[range(t_lstm_out.shape[0]), t_last_idx, :]
            t_pre = self.prev_output_layers[0](t_lstm)
            logit_t = F.sigmoid(self.prev_output_layers[1](t_pre))
            feat_t = [t_lstm, t_pre]  # LSTM, pre-output features
        
        # Student features
        s_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_length.cpu(), batch_first=True, enforce_sorted=False)
        s_lstm_out, _ = model.lstm(s_packed)
        s_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(s_lstm_out, batch_first=True)
        s_last_idx = torch.LongTensor([len_idx - 1 for len_idx in x_length])
        s_lstm =s_lstm_out[range(s_lstm_out.shape[0]), s_last_idx, :]


        

        s_pre = model.main_pre_output_layer(s_lstm)
        logit_s = F.sigmoid(model.main_output_layer(s_pre))
        feat_s = [s_lstm, s_pre]  # LSTM, pre-output features

        # Step 2: 计算路由权重
        ft = torch.cat((feat_t[-1], feat_s[-1]), 1)  # 拼接最后一层特征
        w = torch.mean(F.gumbel_softmax(self.policy_network(ft), tau=1.0), dim=0)  # 使用Gumbel-Softmax
        d = w.detach()  # 停止梯度传播

        # Step 3: 计算损失
        # Cross-entropy loss
        L_s = self.criterion(logit_s, y)
        
        # KL divergence loss with temperature scaling
        L_s += self.beta * F.kl_div(
            F.log_softmax(logit_s / self.temperature, dim=1),
            F.softmax(logit_t.detach() / self.temperature, dim=1),
            reduction='batchmean'
        ) * d[-1]
        
        # Feature distillation loss
        for i in range(len(feat_s)):
            L_s += self.gamma * self.feature_loss(feat_s[i], feat_t[i].detach()) * d[i]

        # Step 4: 特征融合
        student_modules = [
            lambda x: model.lstm(x),  # LSTM
            lambda x: model.main_pre_output_layer(x),  # pre-output
            lambda x: model.main_output_layer(x)  # final output
        ]
        
        teacher_modules = [
            lambda x: self.prev_lstm(x),  # LSTM
            lambda x: self.prev_output_layers[0](x),  # pre-output
            lambda x: self.prev_output_layers[1](x)  # final output
        ]

        # 初始化输入 (packed sequence)
        out_s = s_packed
        out_t = t_packed
        # pdb.set_trace()
        # 逐层计算特征融合
        for i in range(len(student_modules)):
            # 通过当前层的模型
            
            if i == 0:
                s_out,_ = student_modules[i](out_s)
                t_out,_ = teacher_modules[i](out_t)
                s_out, _ = torch.nn.utils.rnn.pad_packed_sequence(s_out, batch_first=True)
                s_last_idx = torch.LongTensor([len_idx - 1 for len_idx in x_length])
                s_out = s_out[range(s_out.shape[0]), s_last_idx, :]
                t_out, _ = torch.nn.utils.rnn.pad_packed_sequence(t_out, batch_first=True)
                t_last_idx = torch.LongTensor([len_idx - 1 for len_idx in x_length])
                t_out = t_out[range(t_out.shape[0]), t_last_idx, :]
            else:
                s_out = student_modules[i](out_s)
                t_out = teacher_modules[i](out_t)
            # 特征融合
            out_s = s_out * (1 - w[i]) + t_out * w[i]
            out_t = s_out * (1 - w[i]) + t_out * w[i]

        # 训练策略网络
        L_routing = self.alpha * self.criterion(F.sigmoid(out_t), y)
        L_routing.backward(retain_graph=True)
        
        if hasattr(self.policy_network, 'optimizer'):
            self.policy_network.optimizer.step()
            self.policy_network.optimizer.zero_grad()

        return L_s


    def mean_step(self, args):
        x = args['x']
        y = args['y']
        model = args['model'] 
        w = torch.tensor([0.25, 0.25, 0.25, 0.25], device=x.device)

        with torch.no_grad():
            # Teacher features
            t_conv1 = F.relu(self.prev_cnn[1](self.prev_cnn[0](x.transpose(1, 2))))
            t_conv2 = F.relu(self.prev_cnn[2](t_conv1))
            t_lstm, _ = self.prev_lstm(t_conv2.transpose(1, 2))
            t_pre = self.prev_output_layers[0](t_lstm[:, -1])
            logit_t = F.sigmoid(self.prev_output_layers[1](t_pre))
            feat_t = [t_conv2, t_lstm[:, -1], t_pre]  # CNN, LSTM, pre-output features

        s_conv1 = F.relu(model.bn1(model.conv1(x.transpose(1, 2))))
        s_conv2 = F.relu(model.conv2(s_conv1))
        s_lstm, _ = model.lstm(s_conv2.transpose(1, 2))
        s_pre = model.main_pre_output_layer(s_lstm[:, -1])
        logit_s = F.sigmoid(model.main_output_layer(s_pre))
        feat_s = [s_conv2, s_lstm[:, -1], s_pre]

        d = w.detach()  # 停止梯度传播
        # Step 2: 计算路由权重
        L_s = self.criterion(logit_s, y)   
        # KL divergence loss
        L_s += self.beta * F.kl_div(F.log_softmax(logit_s  / self.temperature, dim=1),F.softmax(logit_t.detach() / self.temperature, dim=1),reduction='batchmean') *  d[-1]
        for i in range(len(feat_s)):
            L_s += self.gamma * self.feature_loss(feat_s[i], feat_t[i].detach()) * d[i]

        return L_s
    
    def random_step(self, args):
        x = args['x']
        y = args['y']
        model = args['model'] 
        w =  torch.softmax(torch.rand(4, device=x.device), dim=0)

        with torch.no_grad():
            # Teacher features
            t_conv1 = F.relu(self.prev_cnn[1](self.prev_cnn[0](x.transpose(1, 2))))
            t_conv2 = F.relu(self.prev_cnn[2](t_conv1))
            t_lstm, _ = self.prev_lstm(t_conv2.transpose(1, 2))
            t_pre = self.prev_output_layers[0](t_lstm[:, -1])
            logit_t = F.sigmoid(self.prev_output_layers[1](t_pre))
            feat_t = [t_conv2, t_lstm[:, -1], t_pre]  # CNN, LSTM, pre-output features

        s_conv1 = F.relu(model.bn1(model.conv1(x.transpose(1, 2))))
        s_conv2 = F.relu(model.conv2(s_conv1))
        s_lstm, _ = model.lstm(s_conv2.transpose(1, 2))
        s_pre = model.main_pre_output_layer(s_lstm[:, -1])
        logit_s = F.sigmoid(model.main_output_layer(s_pre))
        feat_s = [s_conv2, s_lstm[:, -1], s_pre]

        d = w.detach()  # 停止梯度传播
        # Step 2: 计算路由权重
        L_s = self.criterion(logit_s, y)   
        # KL divergence loss
        L_s += self.beta * F.kl_div(F.log_softmax(logit_s  / self.temperature, dim=1),F.softmax(logit_t.detach() / self.temperature, dim=1),reduction='batchmean') *  d[-1]
        for i in range(len(feat_s)):
            L_s += self.gamma * self.feature_loss(feat_s[i], feat_t[i].detach()) * d[i]

        return L_s


    def step(self, args):
        
        x = args['x']
        x_length = args['x_length']
        loss = args['loss']
        # post_main_output_layer = args['post_main_output_layer']

        # out = self.prev_model(x, x_length)

        # distil_loss = self.distil_loss(post_main_output_layer, out.detach())  # self distillation regularizer
        # total_loss = (1 - self.alpha) * loss + self.alpha * distil_loss
        return loss
    
    
    def forward(self, output):
        pass


