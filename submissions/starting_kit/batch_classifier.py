import sys
import git
import torch


root_path = git.Repo(".", search_parent_directories=True).working_tree_dir

sys.path.append(root_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = "data/private/"


class BatchClassifier:
    def __init__(self):
        self.correct = 0
        self.total_correct = 0
        self.total = 0
        self.print_every = 10
        self.model = self.build_model()

    def build_model(self):
        model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(model.last_channel, 3)

        for param in model.classifier.parameters():
            param.requires_grad = True

        model = model.to(device)
        return model

    def fit(self, gen_builder):
        batch_size = 32
        gen_train, gen_valid, nb_train, nb_valid = (
            gen_builder.get_train_valid_generators(
                batch_size=batch_size, valid_ratio=0.1
            )
        )

        # for i, data in enumerate(gen_train):
        #     inputs, labels = data
        #     inputs, labels = inputs.to(device), labels.to(device)

        #     self.optimizer.zero_grad()

        #     outputs = self.model(inputs)
        #     self.loss = self.criterion(outputs, labels)
        #     self.loss.backward()
        #     self.optimizer.step()

    def predict_proba(self, images):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for _, img in enumerate(images):
                img = torch.Tensor(img.reshape((-1, 3, 400, 400))).to(device)
                outputs = self.model_(img)
                predictions.append(outputs.squeeze().detach().cpu().numpy())
        return predictions
