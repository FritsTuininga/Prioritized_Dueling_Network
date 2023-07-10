import copy
import math
import torch
import numpy as np
import time as time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from joblib import load, dump
from tqdm import tqdm

#set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def plot(x, title="MSE Loss", label="Error", xlabel="Epochs", ylabel="Error"):
    
    #define figure and axis
    fig, ax = plt.subplots()

    #define x- and y-axis (+label)          
    ax.plot([i for i in range(1,len(x)+1)], x)
    
    #define title
    plt.title(title)
    
    #define labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    #show plot
    plt.show()

class CNN(nn.Module):
    def __init__(self,
                 img_size,
                 conv_layers=(1,8,16,32),
                 conv_kernel=4,
                 conv_padding=0,
                 conv_stride=1,
                 pool_kernel=2,
                 pool_padding=0,
                 pool_stride=2):
        super(CNN, self).__init__()
        
        #initialisation
        self.img_size = img_size
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.pool_kernel = pool_kernel
        self.pool_padding = pool_padding
        self.pool_stride = pool_stride
        self.conv_len = len(conv_layers) - 1
        self.conv_output = conv_layers[-1]
        
        #declare convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=conv_layers[i], out_channels=conv_layers[i + 1], kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding) for i in range(len(conv_layers) - 1)])
        
        #value stream for states
        self.fc_h_v = nn.Linear(self.conv_to_lin(), 3)
        self.fc_z_v = nn.Linear(3, 1)
        
        #advantage stream for actions
        self.fc_h_a = nn.Linear(self.conv_to_lin(), 3)
        self.fc_z_a = nn.Linear(3, 3)
        
    def conv_to_lin(self):
        
        #initialise size
        size = self.img_size
        
        #iterate over all convolutional layers
        for i in range(self.conv_len):

            #calculate size after convolutional layer (i.e. rounddown((size - k + 2p)/stride) + 1)
            size = math.floor((size-self.conv_kernel+2*self.conv_padding)/self.conv_stride) + 1

            #calculate size after pooling layer (i.e. rounddown((size - k + 2p)/stride) + 1)
            size = math.floor((size-self.pool_kernel+2*self.pool_padding)/self.pool_stride) + 1

        #calculate number of required neurons (i.e. width*height*channels)
        size = (size*size)*self.conv_output
        
        return size
    
    def forward(self, x):
        
        #pass data through conv layers and apply max. pooling
        for layer in self.conv_layers:
            x = F.max_pool2d(input=F.relu(layer(x)), kernel_size=self.pool_kernel, stride=self.pool_stride, padding=self.pool_padding)

        #ensure correct dimension is used
        if len(x.shape) == 3:
            s = 0
        else:
            s = 1

        #flatten image for linear layers
        x = torch.flatten(x, start_dim=s)
        
        #get state layer values
        v = self.fc_h_v(x)
        v = self.fc_z_v(v)
        
        #get action layer values
        a = self.fc_h_a(x)
        a = self.fc_z_a(a)
        
        #combine streams
        q = v + a - a.mean()
        
        #get final prediction
        return q

class ExperienceReplay():
    def __init__(self):
        
        #initialization
        self.alpha = 0.95
        self.quintuple = ['states', 'actions', 'rewards', 'next_states', 'priorities']
        self.buffer = {key:[] for key in self.quintuple}
    
    def size(self):
        return len(self.buffer['actions'])
    
    def add(self, x):  

        #if buffer size exceeds maximum, then remove oldest items
        if self.size() >= 1763:
            
            #get batch size to delete old items
            del_len = len(x[0])
            
            #remove all oldest batches with size del_len from buffer
            for key in self.quintuple:
                self.buffer[key] = self.buffer[key][del_len:]
               
        #add new items
        for i, key in enumerate(self.quintuple):
            self.buffer[key].extend(x[i])

    def sample(self, probs):
        
        #initialise output
        output = []
        
        #sample replay indices
        idx = torch.multinomial(probs, 64, replacement=True)
        
        #apply priority sampling and select 64 quintuples
        for elem in self.quintuple[:-1]:
            output.append(torch.stack([self.buffer[elem][i] for i in idx], dim=0))

        return output
    
    def get_probs(self):
        
        #calculate priority probabilities
        prob = [el**self.alpha for el in replay.buffer['priorities']]
        probsum = sum(prob)
        probs = [el/probsum for el in prob]
        
        #convert to tensor
        return torch.stack(probs, dim=0)

class Agent():
    def __init__(self):
        
        #initialization
        self.gamma = 0.95
    
    def take_actions(self, states_batch):
        
        #initialization
        batch_size = len(states_batch)
        actions = torch.randint(0,3,(batch_size,)).to(device)
        p = torch.rand((batch_size,))
        
        #get state indices to exploit based on epsilon
        exploit_indices = (p>epsilon).nonzero().flatten()
        
        #get actions
        actions[exploit_indices] = self.predict(primary_model, states_batch[exploit_indices]).argmax(dim=1)
        
        return actions
    
    def get_preds(self, states_batch, actions_batch, rewards_batch, next_states_batch):
        
        #get batch size
        batch_size = len(states_batch)
        
        #get primary model predictions for next states
        primary_pred = self.predict(primary_model, next_states_batch)
        
        #identify best actions
        primary_best_actions = primary_pred.argmax(dim=1)
        
        #get target model predictions for next states
        target_pred = self.predict(target_model, next_states_batch)
        
        #select Q-values based on previously calculated actions
        target_best_qs = target_pred[torch.arange(batch_size), primary_best_actions]
        
        #calculate Q* (target)
        q_star = rewards_batch + self.gamma*target_best_qs
        
        #get primary model predictions for current states and current actions
        q_pred = self.predict(primary_model, states_batch)[torch.arange(batch_size), actions_batch]
        
        return [q_pred, q_star]
    
    def predict(self, model, states_batch):
        
        #get prediction and do not accumulate gradients
        with torch.no_grad():
            return model(states_batch)
    
class Fit():
    def __init__(self):
        
        #intialization
        self.nrm = 1
        self.optimizer = optim.Adam(primary_model.parameters(), lr=1e-3)
    
    def environment_step(self, y, primary_model, target_model):
        
        for i in range(0, 1763, 64):
            
            #get states batch
            states_batch = states[i:i+64]
            
            #get next states batch
            next_states_batch = next_states[i:i+64]
            
            #get y batch
            y_batch = y[i:i+64]
            
            #get actions
            actions_batch = agent.take_actions(states_batch)
            
            #get rewards
            rewards_batch = y_batch*(actions_batch-1)

            #multiply rewards by Negative Rewards Multiplier (NRM)
            rewards_batch[rewards_batch<0] = rewards_batch[rewards_batch<0]*self.nrm
            
            #get predictions and targets
            y_pred, target = agent.get_preds(states_batch, actions_batch, rewards_batch, next_states_batch)
            
            #calculate loss
            priorities = abs(target - y_pred) + (1-epsilon)
            
            #add quintuple to replay buffer
            replay.add([states_batch, actions_batch, rewards_batch, next_states_batch, priorities])

    def update_step(self, primary_model, target_model):
        
        #initialization
        error = []
        
        #get probabilities
        probs = replay.get_probs()
                
        for update in range(UPDATES):
        
            #sample from replay buffer
            states_batch, actions_batch, rewards_batch, next_states_batch = replay.sample(probs)
            
            #get primary model predictions for next states
            primary_pred = agent.predict(primary_model, next_states_batch)
            
            #identify best actions
            primary_best_actions = primary_pred.argmax(dim=1)
            
            #get target model predictions for next states
            target_pred = target_model(next_states_batch)
            
            #select Q-values based on previously calculated actions
            target_best_qs = target_pred[torch.arange(64), primary_best_actions]
            
            #calculate Q* (target)
            q_star = rewards_batch + agent.gamma*target_best_qs
            
            #get primary model predictions for current states
            q_pred = primary_model(states_batch)
            
            #get Q* in the correct form
            q_star_matrix = q_pred.clone()
            q_star_matrix[torch.arange(64), actions_batch] = q_star
            
            #set gradient equal to zero to prevent unwanted accumulation of gradients
            self.optimizer.zero_grad()
            
            #get loss
            loss = loss_function(q_pred, q_star_matrix)
            
            #append error
            error.append(loss)
            
            #calculate gradients for each layer in model (backwards fashion)
            loss.backward()
            
            #update model with specified update method
            self.optimizer.step()
        
            #update target network
            if update%99 == 0:
                
                #copy primary model to target model
                target_model = copy.deepcopy(primary_model)
        
        #return average error as tensor and as a percentage
        return torch.stack(error, dim=0).mean()*100

#PARAMETER SELECTION-----------------------------------------------------------

RUNS = 1
EPOCHS = 10
UPDATES = 100

EPSEXP = 0.15

USE_CUDA = True

#PARAMETER SELECTION-----------------------------------------------------------
    
#use either GPU or CPU
if USE_CUDA and torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is used!")
else:
    device = torch.device('cpu')
    print("CPU is used!")

#import X and y
X_train = load("C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Train\\^GSPC_X.joblib").float().to(device)
y_train = torch.squeeze(load("C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Train\\^GSPC_y.joblib")).float().to(device)*100
X_test = load("C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Test\\^GSPC_X.joblib").float().to(device)
y_test = torch.squeeze(load("C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Test\\^GSPC_y.joblib")).float().to(device)*100

#initialize loss function
loss_function = nn.MSELoss()

#calculate base used for exponential decay
eps_base = EPSEXP**(1/math.floor(EPOCHS/2))

#truncate X- and y-train for calculation (the last state can never transition into a new state)
states = X_train[:-1]
next_states = X_train[1:]
y = y_train[:-1]

#initialize dataframe to store rewards
df_rewards = pd.DataFrame(np.arange(len(y_test)), index=np.arange(len(y_test)), columns=["Index"])

#register start time for iteration
stime = time.time()

for run in range(RUNS):
    
    #initialization
    primary_model = CNN(84).to(device)
    target_model = copy.deepcopy(primary_model)
    replay = ExperienceReplay()
    agent = Agent()
    fit = Fit()
    train_errors = []
    test_rewards = []
        
    for epoch in tqdm(range(EPOCHS)):
        
        #get exponentially decaying epsilon with minimum exploration rate of 10%
        epsilon = max(eps_base**epoch, 0.1)
    
        #take environment step
        fit.environment_step(y, primary_model, target_model)
        
        #take update step and store train errors
        train_errors.append(fit.update_step(primary_model, target_model).cpu().detach().numpy())

        #initialization
        rewards = []
        
        #do not accumulate gradients
        with torch.no_grad():    
            for x_, y_ in zip(X_test, y_test):
                
                #get action given X-test
                action = primary_model(x_).argmax(dim=0) - 1
                
                #get reward
                reward = y_*action
                    
                #append rewards
                rewards.append(reward.cpu().detach().item())

        #clear memory
        torch.cuda.empty_cache()
        
        #store test rewards
        test_rewards.append(np.mean(rewards))
                
    #store model
    dump(primary_model, f'C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Models\\model_{run}.joblib')

    #store rewards
    df_rewards[f"RUN_{run}"] = rewards

#plot results
plot(train_errors)
plot(test_rewards, title="Average Rewards", label="Average Reward", xlabel="Epochs", ylabel="Average Rewards (%)")
plot(rewards, title="Rewards (01-01-2020, 30-06-2020)", label="Reward", xlabel="Days", ylabel="Rewards (%)")

#write df to excel
df_rewards.to_excel('C:\\Users\\Frits\\PycharmProjects\\Prioritized_Dueling_Network\\Excel\\df_PDN_Rewards.xlsx')

print(f"Total computation time: {time.time()-stime}s")
