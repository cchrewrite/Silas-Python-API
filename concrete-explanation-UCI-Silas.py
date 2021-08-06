import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import time
import pickle
import json
from sklearn.metrics import r2_score
import pywt
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Variable

data_file = "UCI_data_all.txt"

device = "cuda:0"
#device = "cpu"
batch_size = 64

class TransferNet(nn.Module):
    def __init__(self, core_type, idim, odim):
        super(TransferNet, self).__init__()
        self.pretrained = True
        self.tmat = nn.Linear(idim, 3 * 224 * 224)
        #self.core = models.vgg16(pretrained = True)
        if core_type == "ResNet18":
            self.core = models.resnet18(pretrained = self.pretrained)
        elif core_type == "AlexNet":
            self.core = models.alexnet(pretrained = self.pretrained)
        elif core_type == "VGG16":
            self.core = models.vgg16(pretrained = self.pretrained)
        elif core_type == "SqueezeNet":
            self.core = models.squeezenet1_0(pretrained = self.pretrained)
        elif core_type == "DenseNet":
            self.core = models.densenet161(pretrained = self.pretrained)
        elif core_type == "InceptionV3":
            self.core = models.inception_v3(pretrained = self.pretrained)
        elif core_type == "GoogLeNet":
            self.core = models.googlenet(pretrained = self.pretrained)
        elif core_type == "ShuffleNet":
            self.core = models.shufflenet_v2_x1_0(pretrained = self.pretrained)
        elif core_type == "MobileNet":
            self.core = models.mobilenet_v2(pretrained = self.pretrained)
        elif core_type == "MNASNet":
            self.core = models.mnasnet1_0(pretrained = self.pretrained)
        elif core_type == "ResNeXt":
            self.core = models.resnext50_32x4d(pretrained = self.pretrained)
        elif core_type == "Wide ResNet":
            self.core = models.wide_resnet50_2(pretrained = self.pretrained)
        self.omat = nn.Linear(1000, 2)
        #self.oact = nn.LogSoftmax()

    def forward(self, x):
        y = self.tmat(x)
        y = y.view(y.shape[0], 3, 224, 224)
        y = self.core.forward(y)
        y = self.omat(y)
        #y = self.oact(y)
        y = F.log_softmax(y, dim=1)
        #y = F.softmax(self.omat(y))
        #print(y.shape)
        #print(y)
        #input("Ad")
        return y

    def predict(self, feat_np):
        #device = "cpu"

        feat = []
        r = 0
        for i in range(int(feat_np.shape[0] / batch_size)):
            l = i * batch_size
            r = i * batch_size + batch_size
            x = torch.Tensor(feat_np[l:r])
            feat.append(x)

        if r < feat_np.shape[0]:
            l = r
            r = feat_np.shape[0]
            x = torch.Tensor(feat_np[l:r])
            feat.append(x)
        
        res = []
        with torch.no_grad():
            for i in range(len(feat)):
                inputs = Variable(feat[i])
                if device != "cpu":
                    inputs = inputs.to(device)
                outputs = self(inputs)
                probs = torch.exp(outputs)
                top_prob, top_class = probs.topk(1, dim = 1)
                if device != "cpu":
                    pred = np.array(top_class.cpu())
                else:
                    pred = np.array(top_class)
                res = res + pred.tolist()
        res = np.array(res)
        return res

    def score(self, feat, tgt):
        pred = self.predict(feat)
        res = 0.0
        for i in range(pred.shape[0]):
            if pred[i] == tgt[i]:
                res = res + 1
        res = res / pred.shape[0]
        return res

def train_transfer_net(core_type, feat_np, tgt_np):

    #device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #classes = ("0", "1")

    #feat = torch.Tensor([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[0.10,0.11,0.12],[0.13,0.14,0.15]])
    #tgt = torch.Tensor([0,0,0,1,1]).type(torch.LongTensor)

    feat = []
    tgt = []

    r = 0
    for i in range(int(feat_np.shape[0] / batch_size)):
        l = i * batch_size
        r = i * batch_size + batch_size
        x = torch.Tensor(feat_np[l:r])
        t = torch.Tensor(tgt_np[l:r]).type(torch.LongTensor)
        feat.append(x)
        tgt.append(t)
    if r < feat_np.shape[0]:
        l = r
        r = feat_np.shape[0]
        x = torch.Tensor(feat_np[l:r])
        t = torch.Tensor(tgt_np[l:r]).type(torch.LongTensor)
        feat.append(x)
        tgt.append(t)
    
    idim = feat_np.shape[1]
    odim = 2
    
    net = TransferNet(core_type, idim, odim)
    if device != "cpu":
        net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.NLLLoss(reduction='sum') #CrossEntropyLoss(reduction='sum')
    for epoch in range(100):
        loss_sum = 0
        for i in range(len(feat)):
            inputs = Variable(feat[i])
            targets = Variable(tgt[i])
            if device != "cpu":
                inputs = inputs.to(device)
                targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #probs = torch.exp(outputs)
            #top_prob, top_class = probs.topk(1, dim = 1)
            #print(top_class)
            #print(top_prob)
            loss = F.nll_loss(outputs, targets)
            #loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            #training_loss = loss.data[0]
            loss_sum = loss_sum + loss.item()
            #print(loss.item())
        loss_ave = loss_sum / feat_np.shape[0]
        #input(loss_ave)
    return net



def split_data_and_add_noise(wdir, data_file, prop, tr_data_file, ev_data_file, noise_scale, apply_wavelet):
    f = open(data_file, "r")
    i = 0
    D = []
    for x in f.readlines():
        if i == 0:
            H = x
        else:
            D.append(x)
        i = i + 1

    if apply_wavelet == True:
        for i in range(len(D)):
            DX = eval(D[i])
            DX[-2] = DX[-2] / 365.0
            cA, cB = pywt.dwt(DX[1:len(DX)-1], 'db2')
            D[i] = str(DX[0:len(DX)-1] + cA.tolist() + cB.tolist() + [DX[-1]]) + "\n"
    
    random.shuffle(D)
    l = int(len(D) * prop)
    D_ev = D[0:l]
    D_tr = D[l:len(D)]

    # Add Gaussian noise to training data.
    for i in range(len(D_tr)):
        #print(D_tr[i])
        if noise_scale > 0:
            DX = eval(D_tr[i])
            dt = np.random.normal(loc = 1.0, scale = noise_scale)
            DX[-1] = DX[-1] * dt
            D_tr[i] = str(DX) + "\n"
            #input(D_tr[i])

    for i in range(len(D_ev)):
        if noise_scale > 0:
            DX = eval(D_ev[i])
            dt = np.random.normal(loc = 1.0, scale = noise_scale)
            DX[-1] = DX[-1] * dt
            D_ev[i] = str(DX) + "\n"


    """
    # Add Gaussian noise to test data.
    for i in range(len(D_ev)):
        if noise_scale > 0:
            DX = eval(D_ev[i])
            for j in range(1, len(DX)):
                if j <= 8:
                    continue
                dt = np.random.normal(loc = 1.0, scale = noise_scale)
                DX[j] = DX[j] * dt
            D_ev[i] = str(DX) + "\n"
    """

    tr_file = open(tr_data_file, "w")
    tr_file.write(H)
    for x in D_tr:
        tr_file.write(x)
    tr_file.close()

    ev_file = open(ev_data_file, "w")
    ev_file.write(H)
    for x in D_ev:
        ev_file.write(x)
    ev_file.close()

    return 0




def feat_z_score_normalisation(feat, u, s):
    f = np.asarray(feat)
    for i in range(f.shape[0]):
        f[i] = (f[i] - u) / s
    return f
    
def symbolic_feat_expansion(feat_header, feat):
    f = np.asarray(feat.T)
    h = feat_header + []
    g = []
    for i in range(f.shape[0]):
        g.append("f[%d]"%i)

    # Change feat_order to a higher value if you want to expand features.
    feat_order = 1
    ub = 10000
    lb = 0.0001
    for p in range(feat_order):
        len_f = f.shape[0]
        for i in range(len_f):
            v = f[i]
            for j in range(len_f):
                w = f[j]
                for sym in ["+", "-", "/"]: #["+", "-", "*", "/"]:
                    hexp = "(%s %s %s)"%(h[i],sym,h[j])
                    gexp = "(f[%d] %s f[%d])"%(i,sym,j)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        x = eval(gexp)
                    #print(gexp)
                    #print(hexp)
                    #print(x)
                    #input("sadfsa")
                    x[np.isnan(x)] = 0
                    x[np.isinf(x)] = 0
                    x[np.isneginf(x)] = 0
                    """
                    x_is_nan = np.isnan(x)
                    x_is_inf = np.isinf(x)
                    x_is_neginf = np.isneginf(x)
                    if True in x_is_nan or True in x_is_inf or True in x_is_neginf:
                        #print(x)
                        #input("PPPP")
                        continue
                    """
                    x_is_redundant = False
                    if np.max(np.abs(x)) < lb:
                        x_is_redundant = True
                        continue
                    if np.min(np.abs(x)) > ub:
                        x_is_redundant = True
                        continue

                    for k in range(f.shape[0]):
                        xmk = np.abs(x - f[k])
                        dmax = np.max(xmk)
                        if dmax < lb:
                            x_is_redundant = True
                            #input("QQQQ")
                            break
                        xmk = np.abs(x + f[k])
                        dmax = np.max(xmk)
                        if dmax < lb:
                            x_is_redundant = True
                            #input("QQQQ")
                            break

                    if x_is_redundant == True:
                        continue

                    f = np.vstack((f,x.reshape(1,x.shape[0])))
                    h.append(hexp)
                    g.append(gexp)
    res = np.array(f.T)
    return [res, h, g]


    
def symbolic_feat_expansion_with_gexp(gexp_list, feat):
    f = np.asarray(feat.T)
    res = []
    for i in range(feat.shape[1], len(gexp_list)):
        gexp = gexp_list[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            x = eval(gexp)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        x[np.isneginf(x)] = 0
        f = np.vstack((f,x.reshape(1,x.shape[0])))
    res = np.array(f.T)
    return res

def np_data_to_csv_file(in_data, out_file):
    f = open(out_file, "w")
    y = ""
    for i in range(in_data.shape[1]-1):
        y = y + "feat%d,"%i
    y = y + "target"
    f.write(y + "\n")
    for x in in_data:
        y = ""
        for i in range(in_data.shape[1]-1):
            y = y + str(x[i]) + ","
            #y = y + "%.6lf"%x[i] + ","
        y = y + str(int(x[-1]))
        f.write(y + "\n")
    f.close()
    return 0
       
def Silas_train_multi(tr_feat, tr_tgt, wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)
    cmd = "cp ../di-projects/cmake-build-debug/silasCLI silasProExp"
    #cmd = "cp ../silas-edu/silas silasProExp"
    os.system(cmd)
    classes = []
    for i in tr_tgt:
        if not(tr_tgt[i] in classes):
            classes.append(tr_tgt[i])
    classes.sort()
    f = open("%s/classes.txt"%wdir, "w")
    f.write(str(classes) + "\n")
    f.close()

    for cls in classes:
        tr_tgt_bi = []
        for i in tr_tgt:
            if tr_tgt[i] == cls:
                tr_tgt_bi.append(0)
            else:
                tr_tgt_bi.append(1)

        tr_tgt_bi = np.asarray(tr_tgt_bi)
        tr_data = np.hstack((tr_feat,tr_tgt_bi.reshape(tr_tgt_bi.shape[0], 1)))

        wdir_bi = wdir + "/class_%s"%str(cls)
        cmd = "mkdir %s"%wdir_bi
        os.system(cmd)
        tr_csv_file = wdir_bi + "/train.csv"
        cv_csv_file = tr_csv_file

        np_data_to_csv_file(tr_data, tr_csv_file)
        Silas_train_binary(tr_csv_file, cv_csv_file, wdir_bi)
        #time.sleep(1)
        input(cls)

        # Validate.



def Silas_train_binary(train_csv_file, test_csv_file, wdir):

    cmd = "mkdir %s"%wdir
    os.system(cmd)

    #cmd = "cp ../silas-edu/silas silasProExp"
    cmd = "cp ../di-projects/cmake-build-debug/silasCLI silasProExp"
    os.system(cmd)

    cmd = "cp %s %s/train.csv"%(train_csv_file, wdir)
    os.system(cmd)

    cmd = "cp %s %s/test.csv"%(test_csv_file, wdir)
    os.system(cmd)

    cmd = "./silasProExp gen-all -o %s %s/train.csv %s/test.csv"%(wdir, wdir, wdir)
    os.system(cmd)

    sf = wdir + "/settings.json"
    with open(sf, "r") as load_f:
        hp_json = json.load(load_f)

    
    # We use settings similar to sklearn random forests.
    # "AdaBoostForest" = "PrototypeSampleForest" > "SimpleForest" = SKRF 
    #hp_json["learner-settings"]["grower-settings"]["forest-settings"]["type"] = "PrototypeSampleForest" # Very good accuracy
    hp_json["learner-settings"]["grower-settings"]["forest-settings"]["type"] = "SimpleForest"
    hp_json["learner-settings"]["grower-settings"]["forest-settings"]["number-of-trees"] = 1000
    hp_json["learner-settings"]["grower-settings"]["forest-settings"]["sampling-proportion"] = 1.0
    hp_json["learner-settings"]["grower-settings"]["tree-settings"]["max-depth"] = 32
    hp_json["learner-settings"]["grower-settings"]["tree-settings"]["desired-leaf-size"] = 2
    hp_json["learner-settings"]["grower-settings"]["tree-settings"]["type"] = "SimpleTreeGrower" #"RdGreedy1D" #"SimpleTreeGrower"
    

    # Regression
    #hp_json["learner-settings"]["grower-settings"]["tree-settings"]["type"] = "RdGreedyReg1DP" #"RdGreedy1D" #"SimpleTreeGrower"
    #hp_json["learner-settings"]["mode"] = "regression"
    #hp_json["learner-settings"]["grower-settings"]["forest-settings"]["type"] = "SimpleOOBRegForest"


    with open(sf, "w") as dump_f:
        json.dump(hp_json, dump_f, indent=4, separators=(',', ': '))
    #input("sadfa")
    """
    # set hyper-parameters
    new-settings = [["sampling-proportion", 1.0], [], []]
    sf = wdir + "/settings.json"
    hp = open(sf, "r")
    hp-json = hp.readlines()
    for i in range(len(hp-json)):
        for p in new-settings:
            if p[0] in hp-json[i]:
                t = "\"%s\": %s"%(p[0], p[1])

    """



    cmd = "./silasProExp learn -o %s/model %s/settings.json > %s/log.tmp"%(wdir, wdir, wdir)
    os.system(cmd)
    #input("sadfa")

    """
    hp_json["learner-settings"]["grower-settings"]["tree-settings"]["type"] = "RdGreedyReg1D" #"RdGreedy1D" #"SimpleTreeGrower"
    #hp_json["learner-settings"]["mode"] = "regression"
    #hp_json["learner-settings"]["grower-settings"]["forest-settings"]["type"] = "SimpleOOBRegForest"


    with open(sf, "w") as dump_f:
        json.dump(hp_json, dump_f, indent=4, separators=(',', ': '))
    input("sadfa")
    """

    return 0


def Silas_predict_multi(feat, wdir):
    f = open("%s/classes.txt"%wdir, "r")
    classes = f.readlines()[0]
    classes = eval(classes)
    
    feat_csv_file = wdir + "/feat.csv"
    feat_tgt = np.asarray([0] * feat.shape[0])
    feat_data = np.hstack((feat,feat_tgt.reshape(feat_tgt.shape[0], 1)))
    np_data_to_csv_file(feat_data, feat_csv_file)

    pred_all = []
    for cls in classes:
        wdir_bi = wdir + "/class_%s"%str(cls)
        feat_csv_file = wdir_bi + "/feat.csv"

        cmd = "./silasProExp predict -o %s/pred.csv %s/model/ %s/feat.csv"%(wdir_bi, wdir_bi, wdir)
        os.system(cmd)

        if os.path.exists("%s/pred.csv"%wdir_bi) == False: # or cls >= 35:# in [35, 49, 55, 56, 61]:
            pred = [0.0] * feat.shape[0]
            pred = np.asarray(pred)
            pred_all.append(pred)
            continue

        pred = []
        f = open("%s/pred.csv"%wdir_bi, "r")
        head_flag = False
        for x in f.readlines():
            if head_flag == False:
                head_flag = True
                continue
            p = x.split(",")
            p = float(p[0])
            #input(p)
            pred.append(p)
        pred_all.append(pred)
        input([cls,len(pred)])
    pred_all = np.asarray(pred_all).T
    input(pred_all.shape)
    pred_idx = pred_all.argmax(axis = 1)
    res = []
    for i in range(feat.shape[0]):
        res.append(classes[pred_idx[i]])
    return res
    
def Silas_score_multi(feat, tgt, wdir):
    pred = Silas_predict_multi(feat, wdir)
    acc = 0
    for i in range(len(tgt)):
        if pred[i] == tgt[i]:
            acc = acc + 1
    acc = acc * 1.0 / len(tgt)
    return acc


def Silas_predict_binary(feat, wdir):

    feat_csv_file = wdir + "/feat.csv"
    feat_tgt = np.asarray([0] * feat.shape[0])
    feat_data = np.hstack((feat,feat_tgt.reshape(feat_tgt.shape[0], 1)))
    np_data_to_csv_file(feat_data, feat_csv_file)
    cmd = "./silasProExp predict -o %s/pred.csv %s/model/ %s/feat.csv > %s/log.tmp"%(wdir, wdir, wdir, wdir)
    os.system(cmd)

    flag = False
    logf = open(wdir + "/log.tmp", "r")
    logdata = logf.readlines()
    logf.close()
    for x in logdata:
        if "Predictions have been saved to" in x:
            flag = True
    if flag == False:
        return []

    pred = []
    f = open("%s/pred.csv"%wdir, "r")
    head_flag = False
    for x in f.readlines():
        if head_flag == False:
            head_flag = True
            head_str = x.replace("\n","")
            continue
        p = x.split(",")
        p = float(p[0])
        pred.append(p)

    classes = head_str.split(",")
    res = []
    for i in range(feat.shape[0]):
        if pred[i] >= 0.5:
            res.append(classes[0])
        else:
            res.append(classes[1])

    res = np.array(res).astype(np.int)
    return res



def Silas_explain_binary(feat, wdir):

    feat_csv_file = wdir + "/feat.csv"
    feat_tgt = np.asarray([0] * feat.shape[0])
    feat_data = np.hstack((feat,feat_tgt.reshape(feat_tgt.shape[0], 1)))
    np_data_to_csv_file(feat_data, feat_csv_file)
    cmd = "./silasProExp explain -x -o %s/expl.csv %s/model/ %s/feat.csv > %s/log.tmp"%(wdir, wdir, wdir, wdir)
    os.system(cmd)

    return 0
     



def Silas_predict_binary_edu(feat, wdir):

    feat_csv_file = wdir + "/feat.csv"
    feat_tgt = np.asarray([0] * feat.shape[0])
    feat_data = np.hstack((feat,feat_tgt.reshape(feat_tgt.shape[0], 1)))
    np_data_to_csv_file(feat_data, feat_csv_file)
    cmd = "./silasProExp predict -o %s/pred.csv %s/model/ %s/feat.csv"%(wdir, wdir, wdir)
    os.system(cmd)

    pred = []
    f = open("%s/pred.csv"%wdir, "r")
    head_flag = False
    for x in f.readlines():
        if head_flag == False:
            head_flag = True
            #head_str = x.replace("\n","")
            continue
        p = x.split(",")
        p = 1 - int(p[0])
        pred.append(p)

    return pred
    


def Silas_score_binary(feat, tgt, wdir):
    pred = Silas_predict_binary(feat, wdir)
    if not(type(pred) is np.ndarray):
        return None
    acc = 0
    for i in range(len(tgt)):
        if str(pred[i]) == str(tgt[i]):
            acc = acc + 1
        """
        else:
            input([str(pred[i]),str(tgt[i])])
        """
    acc = acc * 1.0 / len(tgt)
    return acc
           
        
def binary_dataset_balancing(feat, tgt):
    num_0 = tgt.tolist().count(0)
    num_1 = tgt.tolist().count(1)
    add_feat = []
    add_tgt = []
    if num_0 > num_1:
        t = 1
        dn = num_0 - num_1
    else:
        t = 0
        dn = num_1 = num_0
    for i in range(tgt.shape[0]):
        if tgt[i] == t:
            add_feat.append(feat[i])
            add_tgt.append(tgt[i])

    new_feat = []
    new_tgt = []
    for k in range(dn):
        i = int(random.random() * len(add_tgt))
        new_feat.append(add_feat[i])
        new_tgt.append(add_tgt[i])

    new_feat = np.array(new_feat)
    new_tgt = np.array(new_tgt)

    res_feat = np.vstack((feat, new_feat))
    res_tgt = np.hstack((tgt, new_tgt))

    return [res_feat, res_tgt]

            

def train_binary_classifier(wdir, train_data_file, test_data_file, class_threshold, clf_type):

    cmd = "mkdir -vp %s"%wdir
    os.system(cmd)

    tr_feat = []
    tr_tgt = []

    f = open(train_data_file,"r")
    feat_header = f.readline()
    feat_header = feat_header.replace("Format:","").replace("\n","")
    feat_header = feat_header.replace("[","").replace("]","").replace(" ","")
    feat_header = feat_header.split(",")
    feat_header = feat_header[1:len(feat_header)-1]
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        tr_feat.append(feat)
        if tgt >= class_threshold:
            tgt = 0
        else:
            tgt = 1
        tr_tgt.append(tgt)
        

    print(len(tr_tgt))
    print(min(tr_tgt))
    print(max(tr_tgt))


    ev_feat = []
    ev_tgt = []

    f = open(test_data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        ev_feat.append(feat)
        if tgt >= class_threshold:
            tgt = 0
        else:
            tgt = 1
        ev_tgt.append(tgt)

    tgt_all = tr_tgt + ev_tgt

    tr_feat = np.asarray(tr_feat)
    tr_tgt = np.asarray(tr_tgt)

    ev_feat = np.asarray(ev_feat)
    ev_tgt = np.asarray(ev_tgt)
 
    print(tr_feat.shape)
    feat_exp = symbolic_feat_expansion(feat_header, tr_feat)
    tr_feat = feat_exp[0]
    feat_header_exp = feat_exp[1]
    feat_gexp = feat_exp[2]
   
    print(tr_feat.shape)


    print(ev_feat.shape)
    ev_feat = symbolic_feat_expansion_with_gexp(feat_gexp, ev_feat)
    print(ev_feat.shape)

    tr_feat_org = tr_feat
    tr_tgt_org = tr_tgt
    ev_feat_org = ev_feat
    ev_tgt_org = ev_tgt
    
    # Balancing the datasets.
    tr_data = binary_dataset_balancing(tr_feat, tr_tgt)
    tr_feat = tr_data[0]
    tr_tgt = tr_data[1]

    ev_data = binary_dataset_balancing(ev_feat, ev_tgt)
    ev_feat = ev_data[0]
    ev_tgt = ev_data[1]
    

    f = open(wdir + "/feat_header_and_gexp.txt", "w")
    f.write(str(feat_header_exp) + "\n")
    f.write(str(feat_gexp) + "\n")
    f.close()


    #input("Adfas")

    if clf_type == "Silas":
        tr_data = np.hstack((tr_feat,tr_tgt.reshape(tr_tgt.shape[0], 1)))
        ev_data = np.hstack((ev_feat,ev_tgt.reshape(ev_tgt.shape[0], 1)))

        tr_csv_file = "%s/train.csv"%wdir
        np_data_to_csv_file(tr_data, tr_csv_file)
        ev_csv_file = "%s/test.csv"%wdir
        np_data_to_csv_file(ev_data, ev_csv_file)

        flag = False
        while flag == False:
            Silas_train_binary(tr_csv_file, ev_csv_file, wdir)
            #input("sfd")
            Acc = Silas_score_binary(ev_feat, ev_tgt, wdir)
            if Acc == None:
                print("Training failed due to some unknown reasons. We retrain the model again.")
                continue
            print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

            # Try the explanation function.
            #ex_feat = ev_feat[0:1000,:]
            #ex_feat = tr_feat
            Silas_explain_binary(tr_feat_org, wdir)
            logf = open(wdir + "/log.tmp", "r")
            logdata = logf.readlines()
            logf.close()
            for x in logdata:
                if "Predictions have been saved to" in x:
                    flag = True
            if flag == False:
                print("Cannot perform the explaination function due to some unknown reasons. We retrain the model again.")


        Acc = Silas_score_binary(ev_feat_org, ev_tgt_org, wdir)
        pred_tgt_org = Silas_predict_binary(ev_feat_org, wdir)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)

        return [Acc, Prec, Reca]


    elif clf_type == "SKRF":

        clf = RandomForestClassifier(n_estimators=1000)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
        #clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr')
        #clf = RandomForestRegressor(n_estimators=100)
        clf.fit(tr_feat, tr_tgt)
        clf.apply_z_score = False

        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

        #input(ev_feat_org.shape)
        #input(ev_tgt_org.shape)
        Acc = clf.score(ev_feat_org, ev_tgt_org)
        pred_tgt_org = clf.predict(ev_feat_org)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)
        #input([Acc, Prec, Reca, tn, fp, fn, tp])

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file)

        return [Acc, Prec, Reca]

    elif clf_type in ["SKLR", "SKMLP", "SKKN", "SKSVM", "SKGP", "SKDT", "SKAB", "SKGNB", "SKBNB", "SKGB"]:

        u = np.mean(tr_feat, axis = 0)
        s = np.std(tr_feat, axis = 0)
        tr_feat = feat_z_score_normalisation(tr_feat, u, s)

        if clf_type == "SKLR":
            clf = LogisticRegression(solver='lbfgs',multi_class='ovr')
        elif clf_type == "SKMLP":
            clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(128, 64), max_iter=10000)
        elif clf_type == "SKKN":
            clf = KNeighborsClassifier(100)
        elif clf_type == "SKSVM":
            clf = SVC(kernel="rbf")
        elif clf_type == "SKGP":
            clf = GaussianProcessClassifier()
        elif clf_type == "SKDT":
            clf = DecisionTreeClassifier()
        elif clf_type == "SKAB":
            clf = AdaBoostClassifier(n_estimators = 100)
        elif clf_type == "SKGNB":
            clf = GaussianNB()
        elif clf_type == "SKBNB":
            clf = BernoulliNB()
        elif clf_type == "SKGB":
            clf = GradientBoostingClassifier()

        clf.fit(tr_feat, tr_tgt)

        clf.apply_z_score = True
        clf.feat_u = u
        clf.feat_s = s

        ev_feat = feat_z_score_normalisation(ev_feat, u, s)
        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

        ev_feat_org = feat_z_score_normalisation(ev_feat_org, u, s)
        Acc = clf.score(ev_feat_org, ev_tgt_org)
        pred_tgt_org = clf.predict(ev_feat_org)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)
        #input([Acc, Prec, Reca, tn, fp, fn, tp])

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file)

        return [Acc, Prec, Reca]


    elif clf_type in ["DenseNet", "ResNet18", "AlexNet", "VGG16", "SqueezeNet", "InceptionV3", "GoogLeNet", "ShuffleNet", "MobileNet", "MNASNet", "ResNeXt", "Wide ResNet"]:

        u = np.mean(tr_feat, axis = 0)
        s = np.std(tr_feat, axis = 0)
        tr_feat = feat_z_score_normalisation(tr_feat, u, s)

        clf = train_transfer_net(clf_type, tr_feat, tr_tgt)
        
        clf.apply_z_score = True
        clf.feat_u = u
        clf.feat_s = s

        ev_feat = feat_z_score_normalisation(ev_feat, u, s)
        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

        ev_feat_org = feat_z_score_normalisation(ev_feat_org, u, s)
        Acc = clf.score(ev_feat_org, ev_tgt_org)
        pred_tgt_org = clf.predict(ev_feat_org)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)
        #input([Acc, Prec, Reca, tn, fp, fn, tp])

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file)

        return [Acc, Prec, Reca]


    """
    elif clf_type == "SKLR":

        u = np.mean(tr_feat, axis = 0)
        s = np.std(tr_feat, axis = 0)
        tr_feat = feat_z_score_normalisation(tr_feat, u, s)

        clf = LogisticRegression(solver='lbfgs',multi_class='ovr')
        clf.fit(tr_feat, tr_tgt)

        clf.apply_z_score = True
        clf.feat_u = u
        clf.feat_s = s

        ev_feat = feat_z_score_normalisation(ev_feat, u, s)
        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

        ev_feat_org = feat_z_score_normalisation(ev_feat_org, u, s)
        Acc = clf.score(ev_feat_org, ev_tgt_org)
        pred_tgt_org = clf.predict(ev_feat_org)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)
        #input([Acc, Prec, Reca, tn, fp, fn, tp])

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file)

        return [Acc, Prec, Reca]

    elif clf_type == "SKMLP":

        u = np.mean(tr_feat, axis = 0)
        s = np.std(tr_feat, axis = 0)

        tr_feat = feat_z_score_normalisation(tr_feat, u, s)

        
        Acc_best = -10000.0
        l = int(tr_feat.shape[0] * 0.7)
        tr_feat_train = tr_feat[0:l]
        tr_tgt_train = tr_tgt[0:l]
        tr_feat_cross = tr_feat[l:tr_feat.shape[0]]
        tr_tgt_cross = tr_tgt[l:tr_feat.shape[0]]
        for i in range(10):
            h1 = int(random.random() * 256 + 1)
            h2 = int(random.random() * 256 + 1)
            clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(h1, h2), max_iter=10000)
            clf.fit(tr_feat_train, tr_tgt_train)
            Acc = clf.score(tr_feat_cross, tr_tgt_cross)
            print(Acc)
            if Acc > Acc_best:
                Acc_best = Acc
                best_hidden_layer_sizes = (h1, h2)
        
        #best_hidden_layer_sizes = (128, 128)

        #print("Best hidden layer sizes: %s.\n"%str(best_hidden_layer_sizes))
        #input("adsfas")

        #from sklearn.ensemble import AdaBoostRegressor
        #from sklearn.ensemble import GradientBoostingRegressor
        #regr = GradientBoostingRegressor(random_state=0)
        #regr = AdaBoostRegressor(random_state=0, n_estimators=100)
        clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=best_hidden_layer_sizes, max_iter=10000)
        clf.fit(tr_feat, tr_tgt)

        clf.apply_z_score = True
        clf.feat_u = u
        clf.feat_s = s

        ev_feat = feat_z_score_normalisation(ev_feat, u, s)
        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuracy on the balanced test set is %.3lf%%.\n"%(Acc*100))

        
        ev_feat_org = feat_z_score_normalisation(ev_feat_org, u, s)
        Acc = clf.score(ev_feat_org, ev_tgt_org)
        pred_tgt_org = clf.predict(ev_feat_org)
        tn, fp, fn, tp = confusion_matrix(ev_tgt_org, pred_tgt_org, labels=[0, 1]).ravel()
        Prec = tp * 1.0 / (tp + fp)
        Reca = tp * 1.0 / (tp + fn)
        #input([Acc, Prec, Reca, tn, fp, fn, tp])

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file)

        return [Acc, Prec, Reca]
        """


def train_ordered_classifier(wdir, train_data_file, test_data_file, class_threshold_list, clf_type):

    cmd = "mkdir -vp %s"%wdir
    os.system(cmd)

    f = open(wdir + "/model_info.txt", "w")
    f.write(str(class_threshold_list))
    f.write("\n")
    f.write(clf_type)
    f.write("\n")
    f.close()

    record_list = []

    for class_threshold in class_threshold_list:
        wdir_sub = wdir + "/%d"%class_threshold
        record = train_binary_classifier(wdir_sub, train_data_file, test_data_file, class_threshold, clf_type)
        record_list.append(record)
        
    return record_list
    

def ordered_classifier_prediction(wdir, data_file, tolerance):
    f = open(wdir + "/model_info.txt", "r")
    model_info = f.readlines()
    class_threshold_list = eval(model_info[0])
    clf_type = model_info[1].replace("\n","")
    f.close()

    pred = None
    pred_all = []
    for class_threshold in class_threshold_list:
        wdir_sub = wdir + "/%d"%class_threshold

        ev_feat = []

        f = open(data_file,"r")
        f.readline()
        for x in f.readlines():
            y = eval(x)
            feat = y[1:len(y)-1]
            ev_feat.append(feat)
        ev_feat = np.asarray(ev_feat)

        f = open(wdir_sub + "/feat_header_and_gexp.txt", "r")
        fhg = f.readlines()
        feat_header_exp = eval(fhg[0])
        feat_gexp = eval(fhg[1])
        ev_feat = symbolic_feat_expansion_with_gexp(feat_gexp, ev_feat)


        # format: pred[i][0] is class_threshold, pred[i][1] is tolerance.
        if pred == None:
            pred = []
            for i in range(ev_feat.shape[0]):
                pred.append([0, 0])

        if clf_type == "Silas":
            pred_sub = Silas_predict_binary(ev_feat, wdir_sub)
        else:
            clf_file = open(wdir_sub + "/classifier.mdl", "rb")
            clf = pickle.load(clf_file)
            if clf.apply_z_score == True:
                u = clf.feat_u
                s = clf.feat_s
                ev_feat = feat_z_score_normalisation(ev_feat, u, s) 
            pred_sub = clf.predict(ev_feat)

        #input(pred_sub.shape)
        #input(pred_sub)        
        """
        # Convert results of regression to classes.
        for i in range(pred_sub.shape[0]):
            if pred_sub[i] > 0.5:
                pred_sub[i] = 1
            else:
                pred_sub[i] = 0
        """

        #for x in pred_sub:
            #input(x)
        
        #pred_sub = np.array(pred_sub)
        pred_all.append(pred_sub)
        for i in range(pred_sub.shape[0]):
            #print(pred_sub[i])
            if pred_sub[i] == 1:
                pred[i][1] = pred[i][1] + 1
                #input(i)
            else: # pred_sub[i] == 0
                #print([i,pred_sub[i]])
                #input("Afdsa")
                if pred[i][1] > tolerance:
                    continue
                pred[i][0] = class_threshold
                #print(class_threshold)
    pred_all = np.array(pred_all).T

    """
    for x in pred_all:
        input(x)
    """

    res = []
    for x in pred:
        #input(x)
        res.append(x[0])
    #print(res)
    #input("PRED")
    res = np.array(res).astype(np.int)

    return res


def ordered_classifier_score(wdir, data_file, tolerance):

    f = open(wdir + "/model_info.txt", "r")
    model_info = f.readlines()
    class_threshold_list = eval(model_info[0])
    f.close()


    pred = ordered_classifier_prediction(wdir, data_file, tolerance)
    ev_tgt = []
    f = open(data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        tgt = y[-1]
        if tgt > class_threshold_list[-1]:
            tgt = class_threshold_list[-1]
        elif tgt < class_threshold_list[0]:
            tgt = 0
        else:
            i = len(class_threshold_list) - 1
            while tgt < class_threshold_list[i]:
                i = i - 1
            tgt = class_threshold_list[i]
        ev_tgt.append(tgt)
    print(ev_tgt)
    #input("PPPPP")

    Acc = 0
    for i in range(pred.shape[0]):
        if pred[i] == ev_tgt[i]:
            Acc = Acc + 1
        else:
            print([pred[i], ev_tgt[i]])
            #input("asdfa")

    Acc = Acc * 1.0 / pred.shape[0]

    R2 = r2_score(ev_tgt, pred)
    print("R2 is %lf.\n"%R2)
    input("asdfas")

    return Acc



def train_regressor(wdir, train_data_file, test_data_file, regr_type):

    cmd = "mkdir -vp %s"%wdir
    os.system(cmd)

    tr_feat = []
    tr_tgt = []

    f = open(train_data_file,"r")
    feat_header = f.readline()
    feat_header = feat_header.replace("Format:","").replace("\n","")
    feat_header = feat_header.replace("[","").replace("]","").replace(" ","")
    feat_header = feat_header.split(",")
    feat_header = feat_header[1:len(feat_header)-1]
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        tr_feat.append(feat)
        tr_tgt.append(tgt)

    ev_feat = []
    ev_tgt = []
    

    f = open(test_data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        ev_feat.append(feat)
        ev_tgt.append(tgt)

    tgt_all = tr_tgt + ev_tgt

    tr_feat = np.asarray(tr_feat)
    tr_tgt = np.asarray(tr_tgt)
    ev_feat = np.asarray(ev_feat)
    ev_tgt = np.asarray(ev_tgt)

    if regr_type == "SKRF":

        R2_best = -10000.0
        l = int(tr_feat.shape[0] * 0.7)
        tr_feat_train = tr_feat[0:l]
        tr_tgt_train = tr_tgt[0:l]
        tr_feat_cross = tr_feat[l:tr_feat.shape[0]]
        tr_tgt_cross = tr_tgt[l:tr_feat.shape[0]]
        """
        for i in range(100):
            min_samples_split = int(random.random() * 8 + 2)
            min_samples_leaf = int(random.random() * 8 + 1)
            #h2 = int(random.random() * 256 + 1)
            regr = RandomForestRegressor(n_estimators=100, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=True)
            regr.fit(tr_feat_train, tr_tgt_train)
            R2 = regr.score(tr_feat_cross, tr_tgt_cross)
            print(R2)
            if R2 > R2_best:
                R2_best = R2
                best_min_samples_split = min_samples_split
                best_min_samples_leaf = min_samples_leaf
        """
   
        """
        for min_samples_split in range(2, 8):
            for min_samples_leaf in range(1, 8):
                for max_depth in [22,23,24]:
                    regr = RandomForestRegressor(n_estimators=100, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=True, max_depth= max_depth)
                    regr.fit(tr_feat_train, tr_tgt_train)
                    R2 = regr.score(tr_feat_cross, tr_tgt_cross)
                    print(R2)
                    if R2 > R2_best:
                        R2_best = R2
                        best_min_samples_split = min_samples_split
                        best_min_samples_leaf = min_samples_leaf
                        best_max_depth = max_depth
        """
    
        best_min_samples_split = 2
        best_min_samples_leaf = 1
        best_max_depth = 32
        print([best_min_samples_split, best_min_samples_leaf, best_max_depth])
        input("asdfsa")

        

        regr = RandomForestRegressor(n_estimators=1000, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf, oob_score=True, max_depth = best_max_depth)
        regr.fit(tr_feat, tr_tgt)
        regr.apply_z_score = False
  
        R2 = regr.score(ev_feat, ev_tgt)
        print("R2 on the test set is %.3lf.\n"%(R2))

        regr_file = open(wdir + "/regressor.mdl", "wb")
        pickle.dump(regr, regr_file) 

    elif regr_type == "SKMLP":

        u = np.mean(tr_feat, axis = 0)
        s = np.std(tr_feat, axis = 0)

        tr_feat = feat_z_score_normalisation(tr_feat, u, s)

        """
        R2_best = -10000.0
        l = int(tr_feat.shape[0] * 0.5)
        tr_feat_train = tr_feat[0:l]
        tr_tgt_train = tr_tgt[0:l]
        tr_feat_cross = tr_feat[l:tr_feat.shape[0]]
        tr_tgt_cross = tr_tgt[l:tr_feat.shape[0]]
        for i in range(10):
            h1 = int(random.random() * 256 + 1)
            #h2 = int(random.random() * 256 + 1)
            regr = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(h1, ), max_iter=10000)
            regr.fit(tr_feat_train, tr_tgt_train)
            R2 = regr.score(tr_feat_cross, tr_tgt_cross)
            print(R2)
            if R2 > R2_best:
                R2_best = R2
                best_hidden_layer_sizes = (h1, )
        """
        best_hidden_layer_sizes = (256, 128)

        print("Best hidden layer sizes: %s.\n"%str(best_hidden_layer_sizes))
        input("adsfas")

        #from sklearn.ensemble import AdaBoostRegressor
        #from sklearn.ensemble import GradientBoostingRegressor
        #regr = GradientBoostingRegressor(random_state=0)
        #regr = AdaBoostRegressor(random_state=0, n_estimators=100)
        regr = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=best_hidden_layer_sizes, max_iter=10000)
        regr.fit(tr_feat, tr_tgt)

        regr.apply_z_score = True
        regr.feat_u = u
        regr.feat_s = s

        ev_feat = feat_z_score_normalisation(ev_feat, u, s)

        R2 = regr.score(ev_feat, ev_tgt)
        print("R2 on the test set is %.3lf.\n"%(R2))
        
        regr_file = open(wdir + "/regressor.mdl", "wb")
        pickle.dump(regr, regr_file)

    return regr



def regressor_accuracy_score(regr, test_data_file, class_threshold_list):


    ev_feat = []
    ev_tgt = []

    f = open(test_data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        ev_feat.append(feat)
        if tgt > class_threshold_list[-1]:
            tgt = class_threshold_list[-1]
        elif tgt < class_threshold_list[0]:
            tgt = 0
        else:
            i = len(class_threshold_list) - 1
            while tgt < class_threshold_list[i]:
                i = i - 1
            tgt = class_threshold_list[i]
        ev_tgt.append(tgt)


    ev_feat = np.asarray(ev_feat)
    ev_tgt = np.asarray(ev_tgt)

    if regr.apply_z_score == True:
        u = regr.feat_u
        s = regr.feat_s
        ev_feat = feat_z_score_normalisation(ev_feat, u, s)


    pred_num = regr.predict(ev_feat)

   
    pred = []
    for y in pred_num:
        if y > class_threshold_list[-1]:
            y = class_threshold_list[-1]
        elif y < class_threshold_list[0]:
            y = 0
        else:
            i = len(class_threshold_list) - 1
            while y < class_threshold_list[i]:
                i = i - 1
            y = class_threshold_list[i]
        pred.append(y)


    Acc = 0
    for i in range(len(pred)):
        if pred[i] == ev_tgt[i]:
            Acc = Acc + 1
        else:
            print([pred[i], ev_tgt[i]])
            input("asdfa")

    Acc = Acc * 1.0 / len(pred)

    return Acc




def train_multi_classifier(wdir, train_data_file, test_data_file, clf_type, class_threshold_list):

    cmd = "mkdir -vp %s"%wdir
    os.system(cmd)

    tr_feat = []
    tr_tgt = []

    f = open(train_data_file,"r")
    feat_header = f.readline()
    feat_header = feat_header.replace("Format:","").replace("\n","")
    feat_header = feat_header.replace("[","").replace("]","").replace(" ","")
    feat_header = feat_header.split(",")
    feat_header = feat_header[1:len(feat_header)-1]
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        tr_feat.append(feat)
        if tgt > class_threshold_list[-1]:
            tgt = class_threshold_list[-1]
        elif tgt < class_threshold_list[0]:
            tgt = 0
        else:
            i = len(class_threshold_list) - 1
            while tgt < class_threshold_list[i]:
                i = i - 1
            tgt = class_threshold_list[i]
        tr_tgt.append(tgt)

    ev_feat = []
    ev_tgt = []

    f = open(test_data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        ev_feat.append(feat)
        if tgt > class_threshold_list[-1]:
            tgt = class_threshold_list[-1]
        elif tgt < class_threshold_list[0]:
            tgt = 0
        else:
            i = len(class_threshold_list) - 1
            while tgt < class_threshold_list[i]:
                i = i - 1
            tgt = class_threshold_list[i]
        ev_tgt.append(tgt)


    tgt_all = tr_tgt + ev_tgt

    tr_feat = np.asarray(tr_feat)
    tr_tgt = np.asarray(tr_tgt)
    ev_feat = np.asarray(ev_feat)
    ev_tgt = np.asarray(ev_tgt)

    if clf_type == "SKRF":

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(tr_feat, tr_tgt)

        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuray on the test set is %.3lf%%.\n"%(Acc * 100))

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file) 

    elif clf_type == "SKMLP":

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, ), max_iter=10000, random_state=1)
        clf.fit(tr_feat, tr_tgt)

        Acc = clf.score(ev_feat, ev_tgt)
        print("Accuray on the test set is %.3lf%%.\n"%(Acc * 100))

        clf_file = open(wdir + "/classifier.mdl", "wb")
        pickle.dump(clf, clf_file) 

    return clf

def multi_classifier_accuracy_score(clf, test_data_file, class_threshold_list):

    ev_feat = []
    ev_tgt = []

    f = open(test_data_file,"r")
    f.readline()
    for x in f.readlines():
        y = eval(x)
        feat = y[1:len(y)-1]
        tgt = y[-1]
        ev_feat.append(feat)
        if tgt > class_threshold_list[-1]:
            tgt = class_threshold_list[-1]
        elif tgt < class_threshold_list[0]:
            tgt = 0
        else:
            i = len(class_threshold_list) - 1
            while tgt < class_threshold_list[i]:
                i = i - 1
            tgt = class_threshold_list[i]
        ev_tgt.append(tgt)


    ev_feat = np.asarray(ev_feat)
    ev_tgt = np.asarray(ev_tgt)



    Acc = clf.score(ev_feat, ev_tgt)
   
    return Acc


def experiment_binary_explanation():
    data_file = "UCI_data_all.txt"
    os.system("mkdir experiment_binary_explanation")
    wdir = "experiment_binary_explanation/exp"
    logdir = "experiment_binary_explanation/log"
    os.system("mkdir experiment_binary_explanation/exp")
    os.system("mkdir experiment_binary_explanation/log")
    train_data_file = wdir + "/train_random_split.txt"
    test_data_file = wdir + "/test_random_split.txt"
    """
    strength_list = []
    for i in range(15, 80):
        strength_list.append(i)
    """
    strength_list = [65, 70, 75]# [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    #strength_list = [30, 40, 50]
    random.seed(666)

    clf_list = ["Silas"]
    for t in range(1):
        f = open(logdir + "/" + str(t) + "_Silas.log", "w")
        f.write(str(clf_list) + "\n")
        f.write(str(strength_list) + "\n")
        f.write("Positive = Type 1: the actual strength is SMALLER than the requirement." + "\n")
        f.write(str(["Accuracy", "Precision", "Recall"]) + "\n")
        for clf_type in clf_list:
            split_data_and_add_noise(wdir, data_file, 0.2, train_data_file, test_data_file, 0, False)
            res = train_ordered_classifier(wdir, train_data_file, test_data_file, class_threshold_list = strength_list, clf_type = clf_type)
            print(res)
            f.write(str(res) + "\n")
            #input("dsaf")
        f.close()


experiment_binary_explanation()

