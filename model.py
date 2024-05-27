import torch
import torch.nn as nn
import torch.optim as optim  # neural network layers used to compose into our model
import torch.nn.functional as f  # activation  and max polling functions, used to connect the layers
import os  # used to save the model


# feed forward model with an input layer, hidden layer, and output layer.
class Linear_QNet(nn.Module):
    # init used to construct layers used in computation
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()

        self.linear1 = nn.Linear(
            input_size, hidden_size
        )  # applies linear transformation to the incoming data using weights and biases
        self.linear2 = nn.Linear(hidden_size, output_size)

    # each model has a forward function where computation happens. An input it passed through the network layers to generate an output (a prediction)
    def forward(self, tensor):
        tensor = f.relu(
            self.linear1(tensor)
        )  # Apply the linear layer and use activation function
        tensor = self.linear2(tensor)
        return tensor

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# class that does the actual training and optiomization
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # optimizer is what drives the learning
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # #loss is a measument of how far from our ideal output the models prediction was
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        # can take in a tuple, list, or single value so must convert to pytorch tensor
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # handle multiple sizes
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over,)

        # 1: predicted Q values with current state
        prediction = self.model(state)

        # 2: Q_new = r y * max(next_predicted Q value)
        # pred.clone()
        # pred[argmax(action)] = Q_new
        target = prediction.clone()

        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()  # empty the gradients

        # loss is a measument of how far from our ideal output the models prediction was
        loss = self.criterion(target, prediction)
        loss.backward()  # apply back propagation and calculate gradients that will direct the learning

        # optimizer preforms one learning step.
        # Uses gradients from backward() to nudge learning weights in direction it thinks will reduce loss
        self.optimizer.step()
